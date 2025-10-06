import wgpu
import wgpu.backends.wgpu_native
import numpy as np
import json
import math

# ==============================================================================
# WGPU INITIALIZATION
# ==============================================================================

try:
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    print(f"INFO: wgpu_math backend using GPU: {adapter.info.device}")
except Exception as e:
    print(f"FATAL: Could not initialize WGPU device: {e}")
    print("Please ensure you have working Vulkan drivers.")
    exit(1)

# ==============================================================================
# OPTIMIZED COMPUTE SHADERS
# ==============================================================================

# --- KERNEL 1: Forward pass RNN step (same as before) ---
RNN_STEP_SHADER = """
@group(0) @binding(0) var<storage, read> w_embed: array<f32>;
@group(0) @binding(1) var<storage, read> w_xh: array<f32>;
@group(0) @binding(2) var<storage, read> w_hh: array<f32>;
@group(0) @binding(3) var<storage, read> b_h: array<f32>;
@group(0) @binding(4) var<storage, read> h_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> h_out: array<f32>;

struct Uniforms {
    hidden_size: u32,
    embedding_dim: u32,
    vocab_size: u32,
    input_idx: u32,
};
@group(0) @binding(6) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size;
    let E = uniforms.embedding_dim;
    let i = global_id.x;
    if (i >= H) { return; }

    var xh_dot = 0.0;
    for (var j = 0u; j < E; j = j + 1u) {
        xh_dot += w_xh[i * E + j] * w_embed[j * uniforms.vocab_size + uniforms.input_idx];
    }

    var hh_dot = 0.0;
    for (var j = 0u; j < H; j = j + 1u) {
        hh_dot += w_hh[i * H + j] * h_in[j];
    }
    
    h_out[i] = tanh(xh_dot + hh_dot + b_h[i]);
}
"""

# --- KERNEL 2: Final logits calculation (same as before) ---
FINAL_LOGITS_SHADER = """
@group(0) @binding(0) var<storage, read> w_hy: array<f32>;
@group(0) @binding(1) var<storage, read> b_y: array<f32>;
@group(0) @binding(2) var<storage, read> h_final: array<f32>;
@group(0) @binding(3) var<storage, read_write> logits_out: array<f32>;

struct Uniforms { hidden_size: u32, vocab_size: u32 };
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size;
    let V = uniforms.vocab_size;
    let i = global_id.x;
    if (i >= V) { return; }
    
    var sum = 0.0;
    for (var j = 0u; j < H; j = j + 1u) {
        sum += w_hy[i * H + j] * h_final[j];
    }
    logits_out[i] = sum + b_y[i];
}
"""

# --- KERNEL 3: Backward pass - calculating gradients for output layer (W_hy, b_y) ---
BACKWARD_HY_SHADER = """
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> h_final: array<f32>;
@group(0) @binding(2) var<storage, read_write> dW_hy: array<f32>;
@group(0) @binding(3) var<storage, read_write> db_y: array<f32>;

struct Uniforms { hidden_size: u32, vocab_size: u32 };
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size;
    let V = uniforms.vocab_size;
    let v = global_id.x;
    let h = global_id.y;

    if (v >= V || h >= H) { return; }
    
    dW_hy[v * H + h] = dy[v] * h_final[h];

    if (h == 0u) {
        db_y[v] = dy[v];
    }
}
"""

# --- KERNEL 4: Backward pass - one step of BPTT ---
BPTT_STEP_SHADER = """
@group(0) @binding(0) var<storage, read> W_hy: array<f32>;
@group(0) @binding(1) var<storage, read> W_hh: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read> dh_next: array<f32>;
@group(0) @binding(4) var<storage, read> h_t: array<f32>;
@group(0) @binding(5) var<storage, read> h_prev: array<f32>;
@group(0) @binding(6) var<storage, read> x_embed_t: array<f32>;
@group(0) @binding(7) var<storage, read_write> dW_xh: array<f32>;
@group(0) @binding(8) var<storage, read_write> dW_hh: array<f32>;
@group(0) @binding(9) var<storage, read_write> db_h: array<f32>;
@group(0) @binding(10) var<storage, read_write> dh_next_out: array<f32>;
@group(0) @binding(11) var<storage, read_write> dh_raw_out: array<f32>;

struct Uniforms { hidden_size: u32, embedding_dim: u32, vocab_size: u32 };
@group(0) @binding(12) var<uniform> uniforms: Uniforms;

fn dtanh(y: f32) -> f32 { return 1.0 - y * y; }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size;
    let E = uniforms.embedding_dim;
    let V = uniforms.vocab_size;
    let i = global_id.x;
    if (i >= H) { return; }

    var dh_from_y = 0.0;
    for (var j = 0u; j < V; j = j + 1u) {
        dh_from_y += W_hy[j * H + i] * dy[j];
    }
    let dh = dh_from_y + dh_next[i];
    let dh_raw = dtanh(h_t[i]) * dh;

    db_h[i] += dh_raw;
    for (var j = 0u; j < E; j = j + 1u) {
        dW_xh[i * E + j] += dh_raw * x_embed_t[j];
    }
    for (var j = 0u; j < H; j = j + 1u) {
        dW_hh[i * H + j] += dh_raw * h_prev[j];
    }

    var dh_next_val = 0.0;
    for (var j = 0u; j < H; j = j + 1u) {
        dh_next_val += W_hh[j * H + i] * dh_raw;
    }
    dh_next_out[i] = dh_next_val;
    dh_raw_out[i] = dh_raw;
}
"""

# --- KERNEL 5: Calculates dx_embed and updates dW_embed ---
DX_EMBED_AND_UPDATE_SHADER = """
@group(0) @binding(0) var<storage, read> W_xh: array<f32>;
@group(0) @binding(1) var<storage, read> dh_raw: array<f32>;
@group(0) @binding(2) var<storage, read_write> dW_embed: array<f32>;

struct Uniforms {
    hidden_size: u32,
    embedding_dim: u32,
    vocab_size: u32,
    input_idx: u32,
};
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size;
    let E = uniforms.embedding_dim;
    let V = uniforms.vocab_size;
    let j = global_id.x;
    if (j >= E) { return; }

    var dx_embed_j = 0.0;
    for (var i = 0u; i < H; i = i + 1u) {
        dx_embed_j += W_xh[i * E + j] * dh_raw[i];
    }
    
    dW_embed[j * V + uniforms.input_idx] += dx_embed_j;
}
"""

# --- KERNEL 6: Update weights using calculated gradients (same as before) ---
UPDATE_WEIGHTS_SHADER = """
@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;

struct Uniforms { learning_rate: f32 };
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

fn clip(val: f32, min_val: f32, max_val: f32) -> f32 {
    return max(min_val, min(val, max_val));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&param)) { return; }
    let clipped_grad = clip(grad[i], -5.0, 5.0);
    param[i] = param[i] - uniforms.learning_rate * clipped_grad;
}
"""

def create_compute_pipeline(shader_code, bind_group_layouts):
    shader_module = device.create_shader_module(code=shader_code)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=bind_group_layouts)
    return device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "main"}
    )

# ==============================================================================
# CPU-BASED PRIMITIVES (Softmax and Sampling)
# ==============================================================================

def softmax(logits):
    if not isinstance(logits, np.ndarray): logits = np.array(logits, dtype=np.float32)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=-1, keepdims=True)

def sample_from_dist(probs):
    return np.random.choice(len(probs), p=probs)

# ==============================================================================
# OPTIMIZED MODEL CLASS
# ==============================================================================

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

        self.params = {
            'W_embed': np.random.uniform(-0.01, 0.01, (embedding_dim, vocab_size)).astype(np.float32),
            'W_xh': np.random.uniform(-0.01, 0.01, (hidden_size, embedding_dim)).astype(np.float32),
            'W_hh': np.random.uniform(-0.01, 0.01, (hidden_size, hidden_size)).astype(np.float32),
            'W_hy': np.random.uniform(-0.01, 0.01, (vocab_size, hidden_size)).astype(np.float32),
            'b_h': np.zeros(hidden_size, dtype=np.float32),
            'b_y': np.zeros(vocab_size, dtype=np.float32),
        }
        
        self.params_gpu = {}
        self.grads_gpu = {}
        param_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        grad_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        
        for name, arr in self.params.items():
            self.params_gpu[name] = device.create_buffer_with_data(data=arr, usage=param_usage)
            self.grads_gpu[name] = device.create_buffer(size=arr.nbytes, usage=grad_usage)
            
        self._create_pipelines()

    def _create_pipelines(self):
        b = wgpu.BufferBindingType
        s = wgpu.ShaderStage
        
        self.bgls = {
            'rnn_step': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 3, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 4, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 5, "visibility": s.COMPUTE, "buffer": {"type": b.storage}},
                {"binding": 6, "visibility": s.COMPUTE, "buffer": {"type": b.uniform}},
            ]),
            'final_logits': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 3, "visibility": s.COMPUTE, "buffer": {"type": b.storage}},
                {"binding": 4, "visibility": s.COMPUTE, "buffer": {"type": b.uniform}},
            ]),
            'backward_hy': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": {"type": b.storage}},
                {"binding": 3, "visibility": s.COMPUTE, "buffer": {"type": b.storage}},
                {"binding": 4, "visibility": s.COMPUTE, "buffer": {"type": b.uniform}},
            ]),
            'bptt_step': device.create_bind_group_layout(entries=[
                *([{"binding": i, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}} for i in range(7)]),
                *([{"binding": i, "visibility": s.COMPUTE, "buffer": {"type": b.storage}} for i in range(7, 12)]),
                {"binding": 12, "visibility": s.COMPUTE, "buffer": {"type": b.uniform}},
            ]),
            'dx_embed_update': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": {"type": b.storage}},
                {"binding": 3, "visibility": s.COMPUTE, "buffer": {"type": b.uniform}},
            ]),
            'update_weights': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": {"type": b.storage}},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": {"type": b.read_only_storage}},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": {"type": b.uniform}},
            ]),
        }
        self.pipelines = {
            'rnn_step': create_compute_pipeline(RNN_STEP_SHADER, [self.bgls['rnn_step']]),
            'final_logits': create_compute_pipeline(FINAL_LOGITS_SHADER, [self.bgls['final_logits']]),
            'backward_hy': create_compute_pipeline(BACKWARD_HY_SHADER, [self.bgls['backward_hy']]),
            'bptt_step': create_compute_pipeline(BPTT_STEP_SHADER, [self.bgls['bptt_step']]),
            'dx_embed_update': create_compute_pipeline(DX_EMBED_AND_UPDATE_SHADER, [self.bgls['dx_embed_update']]),
            'update_weights': create_compute_pipeline(UPDATE_WEIGHTS_SHADER, [self.bgls['update_weights']]),
        }

    def get_initial_hidden_state(self):
        return np.zeros(self.hidden_size, dtype=np.float32)

    def _get_params(self):
        cpu_params = {}
        for name, gpu_buf in self.params_gpu.items():
            data = device.queue.read_buffer(gpu_buf)
            arr = np.frombuffer(data, dtype=np.float32).reshape(self.params[name].shape)
            cpu_params[name] = arr.tolist()
        return cpu_params
        
    def _set_params(self, params):
        for name, data in params.items():
            arr = np.array(data, dtype=np.float32)
            self.params[name] = arr
            device.queue.write_buffer(self.params_gpu[name], 0, arr)
            
    def save_model(self, filepath: str):
        model_data = {
            'config': {'vocab_size': self.vocab_size, 'hidden_size': self.hidden_size, 'embedding_dim': self.embedding_dim},
            'params': self._get_params()
        }
        with open(filepath, 'w') as f: json.dump(model_data, f)
        print(f"INFO: Model saved to '{filepath}'")

    @classmethod
    def load_model(cls, filepath: str):
        print(f"INFO: Loading model from '{filepath}'...")
        with open(filepath, 'r') as f: model_data = json.load(f)
        config = model_data['config']
        model = cls(config['vocab_size'], config['hidden_size'], config['embedding_dim'])
        model._set_params(model_data['params'])
        return model

    def forward_pass(self, inputs, h_prev):
        S, H, E, V = len(inputs), self.hidden_size, self.embedding_dim, self.vocab_size
        
        h_history_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        h_history_buffer = device.create_buffer(size=S * H * 4, usage=h_history_usage)
        
        h_buffer_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        h_buffer = device.create_buffer_with_data(data=h_prev, usage=h_buffer_usage)
        
        command_encoder = device.create_command_encoder()
        for t, x_idx in enumerate(inputs):
            uniform_data = np.array([H, E, V, x_idx], dtype=np.uint32)
            uniform_buffer = device.create_buffer_with_data(data=uniform_data, usage=wgpu.BufferUsage.UNIFORM)
            
            h_out_buffer = device.create_buffer(size=H * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

            bind_group = device.create_bind_group(layout=self.bgls['rnn_step'], entries=[
                {"binding": 0, "resource": {"buffer": self.params_gpu['W_embed']}},
                {"binding": 1, "resource": {"buffer": self.params_gpu['W_xh']}},
                {"binding": 2, "resource": {"buffer": self.params_gpu['W_hh']}},
                {"binding": 3, "resource": {"buffer": self.params_gpu['b_h']}},
                {"binding": 4, "resource": {"buffer": h_buffer}},
                {"binding": 5, "resource": {"buffer": h_out_buffer}},
                {"binding": 6, "resource": {"buffer": uniform_buffer}},
            ])
            
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.pipelines['rnn_step'])
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(math.ceil(H / 64))
            compute_pass.end()
            
            command_encoder.copy_buffer_to_buffer(h_out_buffer, 0, h_buffer, 0, h_buffer.size)
            command_encoder.copy_buffer_to_buffer(h_out_buffer, 0, h_history_buffer, t * H * 4, h_out_buffer.size)

        final_logits_buffer = device.create_buffer(size=self.params['b_y'].nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        uniform_data = np.array([H, V], dtype=np.uint32)
        uniform_buffer = device.create_buffer_with_data(data=uniform_data, usage=wgpu.BufferUsage.UNIFORM)
        
        bind_group = device.create_bind_group(layout=self.bgls['final_logits'], entries=[
            {"binding": 0, "resource": {"buffer": self.params_gpu['W_hy']}},
            {"binding": 1, "resource": {"buffer": self.params_gpu['b_y']}},
            {"binding": 2, "resource": {"buffer": h_buffer}},
            {"binding": 3, "resource": {"buffer": final_logits_buffer}},
            {"binding": 4, "resource": {"buffer": uniform_buffer}},
        ])

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['final_logits'])
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(math.ceil(V / 64))
        compute_pass.end()
        device.queue.submit([command_encoder.finish()])

        final_logits = np.frombuffer(device.queue.read_buffer(final_logits_buffer), dtype=np.float32)
        all_h_np = np.frombuffer(device.queue.read_buffer(h_history_buffer), dtype=np.float32).reshape(S, H)
        
        cache = {'h': {-1: h_prev}}
        for t in range(S): cache['h'][t] = all_h_np[t, :]
        
        return final_logits, all_h_np[-1, :], cache
        
    def backward_pass(self, inputs, target_idx, cache):
        S, H, E, V = len(inputs), self.hidden_size, self.embedding_dim, self.vocab_size
        
        logits, _, _ = self.forward_pass(inputs, cache['h'][-1])
        probs = softmax(logits)
        dy = probs; dy[target_idx] -= 1
        dy_gpu = device.create_buffer_with_data(data=dy, usage=wgpu.BufferUsage.STORAGE)

        command_encoder = device.create_command_encoder()
        for grad_buf in self.grads_gpu.values():
            command_encoder.clear_buffer(grad_buf)

        h_final_gpu = device.create_buffer_with_data(data=cache['h'][len(inputs)-1], usage=wgpu.BufferUsage.STORAGE)
        uniform_data_hy = np.array([H, V], dtype=np.uint32)
        uniform_buffer_hy = device.create_buffer_with_data(data=uniform_data_hy, usage=wgpu.BufferUsage.UNIFORM)
        
        bg_hy = device.create_bind_group(layout=self.bgls['backward_hy'], entries=[
            {"binding": 0, "resource": {"buffer": dy_gpu}},
            {"binding": 1, "resource": {"buffer": h_final_gpu}},
            {"binding": 2, "resource": {"buffer": self.grads_gpu['W_hy']}},
            {"binding": 3, "resource": {"buffer": self.grads_gpu['b_y']}},
            {"binding": 4, "resource": {"buffer": uniform_buffer_hy}},
        ])
        
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['backward_hy'])
        compute_pass.set_bind_group(0, bg_hy)
        compute_pass.dispatch_workgroups(math.ceil(V / 8), math.ceil(H / 8))
        compute_pass.end()

        dh_next_gpu = device.create_buffer(size=H*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
        command_encoder.clear_buffer(dh_next_gpu)
        dh_raw_gpu = device.create_buffer(size=H*4, usage=wgpu.BufferUsage.STORAGE)

        for t in reversed(range(len(inputs))):
            h_t = device.create_buffer_with_data(data=cache['h'][t], usage=wgpu.BufferUsage.STORAGE)
            h_prev = device.create_buffer_with_data(data=cache['h'][t-1], usage=wgpu.BufferUsage.STORAGE)
            
            x_embed_t_data = np.ascontiguousarray(self.params['W_embed'][:, inputs[t]])
            x_embed_t = device.create_buffer_with_data(data=x_embed_t_data, usage=wgpu.BufferUsage.STORAGE)
            
            dh_next_out_gpu = device.create_buffer(size=dh_next_gpu.size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
            uniform_data_bptt = np.array([H, E, V], dtype=np.uint32)
            uniform_buffer_bptt = device.create_buffer_with_data(data=uniform_data_bptt, usage=wgpu.BufferUsage.UNIFORM)

            bg_bptt = device.create_bind_group(layout=self.bgls['bptt_step'], entries=[
                {"binding": 0, "resource": {"buffer": self.params_gpu['W_hy']}},
                {"binding": 1, "resource": {"buffer": self.params_gpu['W_hh']}},
                {"binding": 2, "resource": {"buffer": dy_gpu}},
                {"binding": 3, "resource": {"buffer": dh_next_gpu}},
                {"binding": 4, "resource": {"buffer": h_t}},
                {"binding": 5, "resource": {"buffer": h_prev}},
                {"binding": 6, "resource": {"buffer": x_embed_t}},
                {"binding": 7, "resource": {"buffer": self.grads_gpu['W_xh']}},
                {"binding": 8, "resource": {"buffer": self.grads_gpu['W_hh']}},
                {"binding": 9, "resource": {"buffer": self.grads_gpu['b_h']}},
                {"binding": 10, "resource": {"buffer": dh_next_out_gpu}},
                {"binding": 11, "resource": {"buffer": dh_raw_gpu}},
                {"binding": 12, "resource": {"buffer": uniform_buffer_bptt}},
            ])
            
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.pipelines['bptt_step'])
            compute_pass.set_bind_group(0, bg_bptt)
            compute_pass.dispatch_workgroups(math.ceil(H / 64))
            compute_pass.end()
            
            command_encoder.copy_buffer_to_buffer(dh_next_out_gpu, 0, dh_next_gpu, 0, dh_next_gpu.size)

            uniform_data_embed = np.array([H, E, V, inputs[t]], dtype=np.uint32)
            uniform_buffer_embed = device.create_buffer_with_data(data=uniform_data_embed, usage=wgpu.BufferUsage.UNIFORM)
            bg_embed = device.create_bind_group(layout=self.bgls['dx_embed_update'], entries=[
                {"binding": 0, "resource": {"buffer": self.params_gpu['W_xh']}},
                {"binding": 1, "resource": {"buffer": dh_raw_gpu}},
                {"binding": 2, "resource": {"buffer": self.grads_gpu['W_embed']}},
                {"binding": 3, "resource": {"buffer": uniform_buffer_embed}},
            ])
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.pipelines['dx_embed_update'])
            compute_pass.set_bind_group(0, bg_embed)
            compute_pass.dispatch_workgroups(math.ceil(E / 64))
            compute_pass.end()

        device.queue.submit([command_encoder.finish()])
        return {}

    def update_weights(self, grads, learning_rate):
        command_encoder = device.create_command_encoder()
        uniform_data = np.array([learning_rate], dtype=np.float32)
        uniform_buffer = device.create_buffer_with_data(data=uniform_data, usage=wgpu.BufferUsage.UNIFORM)
        
        for name, param_buf in self.params_gpu.items():
            grad_buf = self.grads_gpu[name]
            bind_group = device.create_bind_group(layout=self.bgls['update_weights'], entries=[
                {"binding": 0, "resource": {"buffer": param_buf}},
                {"binding": 1, "resource": {"buffer": grad_buf}},
                {"binding": 2, "resource": {"buffer": uniform_buffer}},
            ])
            
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.pipelines['update_weights'])
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(math.ceil(param_buf.size / 4 / 256))
            compute_pass.end()
            
        device.queue.submit([command_encoder.finish()])