import wgpu
import wgpu.backends.wgpu_native
import numpy as np
import json
import math
import time

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

@compute @workgroup_size(8, 8, 1) // 2D workgroup
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size;
    let V = uniforms.vocab_size;
    let v = global_id.x; // vocab index
    let h = global_id.y; // hidden index

    if (v >= V || h >= H) { return; }
    
    // Outer product: dW_hy = dy outer h_final
    dW_hy[v * H + h] = dy[v] * h_final[h];

    // Parallel update for db_y, only run by the first column of threads
    if (h == 0u) {
        db_y[v] = dy[v];
    }
}
"""

# --- KERNEL 4: Backward pass - one step of Backpropagation Through Time (BPTT) ---
BPTT_STEP_SHADER = """
// Input parameters
@group(0) @binding(0) var<storage, read> W_hy: array<f32>;
@group(0) @binding(1) var<storage, read> W_hh: array<f32>;
@group(0) @binding(2) var<storage, read> W_xh: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read> dh_next: array<f32>;
@group(0) @binding(5) var<storage, read> h_t: array<f32>;
@group(0) @binding(6) var<storage, read> h_prev: array<f32>;
@group(0) @binding(7) var<storage, read> x_embed_t: array<f32>;

// Gradients to accumulate
@group(0) @binding(8) var<storage, read_write> dW_xh: array<f32>;
@group(0) @binding(9) var<storage, read_write> dW_hh: array<f32>;
@group(0) @binding(10) var<storage, read_write> db_h: array<f32>;
@group(0) @binding(11) var<storage, read_write> dW_embed: array<f32>;
@group(0) @binding(12) var<storage, read_write> dh_next_out: array<f32>; // To be used in the next BPTT step

struct Uniforms {
    hidden_size: u32,
    embedding_dim: u32,
    vocab_size: u32,
    input_idx: u32,
};
@group(0) @binding(13) var<uniform> uniforms: Uniforms;

fn dtanh(y: f32) -> f32 { return 1.0 - y * y; }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size;
    let E = uniforms.embedding_dim;
    let V = uniforms.vocab_size;
    let i = global_id.x; // Corresponds to hidden_size index

    if (i >= H) { return; }

    // Calculate dh = W_hy.T @ dy + dh_next
    var dh_from_y = 0.0;
    for (var j = 0u; j < V; j = j + 1u) {
        dh_from_y += W_hy[j * H + i] * dy[j];
    }
    let dh = dh_from_y + dh_next[i];
    let dh_raw = dtanh(h_t[i]) * dh;

    // Accumulate gradients for b_h, W_xh, W_hh
    db_h[i] += dh_raw;
    for (var j = 0u; j < E; j = j + 1u) {
        dW_xh[i * E + j] += dh_raw * x_embed_t[j];
    }
    for (var j = 0u; j < H; j = j + 1u) {
        dW_hh[i * H + j] += dh_raw * h_prev[j];
    }

    // Calculate dx_embed and accumulate embedding gradients
    // Note: atomicAdd would be better, but this is a reasonable approximation
    // that works by having each thread update its corresponding part of the embedding.
    var dx_embed = 0.0;
     for (var j = 0u; j < H; j = j+1u) {
        dx_embed += W_xh[j * E + i] * dh_raw[j];
    }
    dW_embed[i * V + uniforms.input_idx] += dx_embed;


    // Calculate dh_next for the previous time step
    var dh_next_val = 0.0;
    for (var j = 0u; j < H; j = j + 1u) {
        dh_next_val += W_hh[j * H + i] * dh_raw[j];
    }
    dh_next_out[i] = dh_next_val;
}
"""

# --- KERNEL 5: Update weights using calculated gradients ---
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
    // Note: A check to see if i is out of bounds is implicitly handled by buffer size
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

# Placeholder for pipelines, to be created in the class __init__
# This avoids creating global pipelines with hardcoded layouts.

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

        # --- CPU Parameters (as numpy arrays) ---
        self.params = {
            'W_embed': np.random.uniform(-0.01, 0.01, (embedding_dim, vocab_size)).astype(np.float32),
            'W_xh': np.random.uniform(-0.01, 0.01, (hidden_size, embedding_dim)).astype(np.float32),
            'W_hh': np.random.uniform(-0.01, 0.01, (hidden_size, hidden_size)).astype(np.float32),
            'W_hy': np.random.uniform(-0.01, 0.01, (vocab_size, hidden_size)).astype(np.float32),
            'b_h': np.zeros(hidden_size, dtype=np.float32),
            'b_y': np.zeros(vocab_size, dtype=np.float32),
        }
        
        # --- GPU Buffers for Parameters and Gradients ---
        self.params_gpu = {}
        self.grads_gpu = {}
        param_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        grad_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        
        for name, arr in self.params.items():
            self.params_gpu[name] = device.create_buffer_with_data(data=arr, usage=param_usage)
            self.grads_gpu[name] = device.create_buffer(size=arr.nbytes, usage=grad_usage)
            
        self._create_pipelines()

    def _create_pipelines(self):
        # --- Create Bind Group Layouts ---
        self.bgls = {
            'rnn_step': device.create_bind_group_layout(entries=[
                {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}} for i in range(6)]
                + [{"binding": 6, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}}
            ]),
            'final_logits': device.create_bind_group_layout(entries=[
                {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}} for i in range(3)]
                + [{"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
                   {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}}
            ]),
            'backward_hy': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}}
            ]),
            'bptt_step': device.create_bind_group_layout(entries=[
                {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}} for i in range(8)]
                + [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}} for i in range(8, 13)]
                + [{"binding": 13, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}}
            ]),
            'update_weights': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}}
            ]),
        }
        # --- Create Pipelines ---
        self.pipelines = {
            'rnn_step': create_compute_pipeline(RNN_STEP_SHADER, [self.bgls['rnn_step']]),
            'final_logits': create_compute_pipeline(FINAL_LOGITS_SHADER, [self.bgls['final_logits']]),
            'backward_hy': create_compute_pipeline(BACKWARD_HY_SHADER, [self.bgls['backward_hy']]),
            'bptt_step': create_compute_pipeline(BPTT_STEP_SHADER, [self.bgls['bptt_step']]),
            'update_weights': create_compute_pipeline(UPDATE_WEIGHTS_SHADER, [self.bgls['update_weights']]),
        }

    def get_initial_hidden_state(self):
        return np.zeros(self.hidden_size, dtype=np.float32)

    def _get_params(self):
        # Read all parameters from GPU to CPU for saving
        command_encoder = device.create_command_encoder()
        cpu_params = {}
        for name, gpu_buf in self.params_gpu.items():
            output_buffer = device.create_buffer(size=gpu_buf.size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
            command_encoder.copy_buffer_to_buffer(gpu_buf, 0, output_buffer, 0, gpu_buf.size)
            cpu_params[name] = output_buffer
        device.queue.submit([command_encoder.finish()])
        
        final_params = {}
        for name, arr_shape in self.params.items():
            buf = cpu_params[name]
            data = buf.read_mapped()
            arr = np.frombuffer(data, dtype=np.float32).reshape(self.params[name].shape)
            final_params[name] = arr.tolist()
            buf.unmap()
        return final_params
        
    def _set_params(self, params):
        for name, data in params.items():
            arr = np.array(data, dtype=np.float32)
            self.params[name] = arr
            # Write new data to existing GPU buffers
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
        
        # Create GPU buffer to store the entire history of hidden states
        h_history_buffer = device.create_buffer(size=S * H * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        h_buffer = device.create_buffer_with_data(data=h_prev, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
        
        command_encoder = device.create_command_encoder()
        for t, x_idx in enumerate(inputs):
            uniform_data = np.array([H, E, V, x_idx], dtype=np.uint32)
            uniform_buffer = device.create_buffer_with_data(data=uniform_data, usage=wgpu.BufferUsage.UNIFORM)
            
            # The output buffer is a slice of the history buffer
            h_out_view = device.create_buffer(size=H * 4, usage=wgpu.BufferUsage.STORAGE) # This feels wrong, can't create view. Let's copy.
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
            compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
            compute_pass.dispatch_workgroups(math.ceil(H / 64))
            compute_pass.end()
            
            command_encoder.copy_buffer_to_buffer(h_out_buffer, 0, h_buffer, 0, h_buffer.size)
            command_encoder.copy_buffer_to_buffer(h_out_buffer, 0, h_history_buffer, t * H * 4, h_out_buffer.size)

        # Calculate final logits
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
        compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
        compute_pass.dispatch_workgroups(math.ceil(V / 64))
        compute_pass.end()
        device.queue.submit([command_encoder.finish()])

        # Read results back to CPU once
        final_logits = np.frombuffer(device.queue.read_buffer(final_logits_buffer), dtype=np.float32)
        all_h_np = np.frombuffer(device.queue.read_buffer(h_history_buffer), dtype=np.float32).reshape(S, H)
        
        # Populate cache for backward pass
        cache = {'h': {-1: h_prev}, 'h_history_buffer': h_history_buffer}
        for t in range(S): cache['h'][t] = all_h_np[t, :]
        
        return final_logits, all_h_np[-1, :], cache
        
    def backward_pass(self, inputs, target_idx, cache):
        S, H, E, V = len(inputs), self.hidden_size, self.embedding_dim, self.vocab_size
        
        # --- 1. Calculate dy (on CPU, it's small) ---
        logits, _, _ = self.forward_pass(inputs, cache['h'][-1])
        probs = softmax(logits)
        dy = probs; dy[target_idx] -= 1
        dy_gpu = device.create_buffer_with_data(data=dy, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)

        # --- 2. Zero out gradients ---
        command_encoder = device.create_command_encoder()
        for grad_buf in self.grads_gpu.values():
            command_encoder.clear_buffer(grad_buf, 0, grad_buf.size)

        # --- 3. Backward pass for output layer (W_hy, b_y) ---
        h_final_gpu = device.create_buffer_with_data(data=cache['h'][len(inputs)-1], usage=wgpu.BufferUsage.STORAGE)
        uniform_data = np.array([H, V], dtype=np.uint32)
        uniform_buffer = device.create_buffer_with_data(data=uniform_data, usage=wgpu.BufferUsage.UNIFORM)
        
        bg_hy = device.create_bind_group(layout=self.bgls['backward_hy'], entries=[
            {"binding": 0, "resource": {"buffer": dy_gpu}},
            {"binding": 1, "resource": {"buffer": h_final_gpu}},
            {"binding": 2, "resource": {"buffer": self.grads_gpu['W_hy']}},
            {"binding": 3, "resource": {"buffer": self.grads_gpu['b_y']}},
            {"binding": 4, "resource": {"buffer": uniform_buffer}},
        ])
        
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['backward_hy'])
        compute_pass.set_bind_group(0, bg_hy, [], 0, 999999)
        compute_pass.dispatch_workgroups(math.ceil(V / 8), math.ceil(H / 8))
        compute_pass.end()

        # --- 4. BPTT Loop ---
        dh_next_gpu = device.create_buffer(size=self.params['b_h'].nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
        
        for t in reversed(range(len(inputs))):
            h_t = device.create_buffer_with_data(data=cache['h'][t], usage=wgpu.BufferUsage.STORAGE)
            h_prev = device.create_buffer_with_data(data=cache['h'][t-1], usage=wgpu.BufferUsage.STORAGE)
            x_embed_t = device.create_buffer_with_data(data=self.params['W_embed'][:, inputs[t]], usage=wgpu.BufferUsage.STORAGE)
            
            dh_next_out_gpu = device.create_buffer(size=dh_next_gpu.size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
            
            uniform_data = np.array([H, E, V, inputs[t]], dtype=np.uint32)
            uniform_buffer = device.create_buffer_with_data(data=uniform_data, usage=wgpu.BufferUsage.UNIFORM)

            bg_bptt = device.create_bind_group(layout=self.bgls['bptt_step'], entries=[
                {"binding": i, "resource": {"buffer": buf}} for i, buf in enumerate([
                    self.params_gpu['W_hy'], self.params_gpu['W_hh'], self.params_gpu['W_xh'], dy_gpu, dh_next_gpu, h_t, h_prev, x_embed_t,
                    self.grads_gpu['W_xh'], self.grads_gpu['W_hh'], self.grads_gpu['b_h'], self.grads_gpu['W_embed'], dh_next_out_gpu
                ])] + [{"binding": 13, "resource": {"buffer": uniform_buffer}}]
            )
            
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.pipelines['bptt_step'])
            compute_pass.set_bind_group(0, bg_bptt, [], 0, 999999)
            compute_pass.dispatch_workgroups(math.ceil(H / 64))
            compute_pass.end()
            
            command_encoder.copy_buffer_to_buffer(dh_next_out_gpu, 0, dh_next_gpu, 0, dh_next_gpu.size)

        device.queue.submit([command_encoder.finish()])
        return {} # Grads are already on the GPU

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
            compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
            compute_pass.dispatch_workgroups(math.ceil(param_buf.size / 4 / 256))
            compute_pass.end()
            
        device.queue.submit([command_encoder.finish()])