import wgpu
import wgpu.backends.wgpu_native
import numpy as np
import json
import math
import random

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

# --- KERNEL 1: Forward pass RNN step (Modified for corpus buffer) ---
RNN_STEP_SHADER = """
@group(0) @binding(0) var<storage, read> w_embed: array<f32>;
@group(0) @binding(1) var<storage, read> w_xh: array<f32>;
@group(0) @binding(2) var<storage, read> w_hh: array<f32>;
@group(0) @binding(3) var<storage, read> b_h: array<f32>;
@group(0) @binding(4) var<storage, read> h_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> h_out: array<f32>;
@group(0) @binding(6) var<storage, read> corpus: array<u32>;

struct Uniforms {
    hidden_size: u32,
    embedding_dim: u32,
    vocab_size: u32,
    input_offset: u32,
};
@group(0) @binding(7) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size;
    let E = uniforms.embedding_dim;
    let i = global_id.x;
    if (i >= H) { return; }

    let input_idx = corpus[uniforms.input_offset];

    var xh_dot = 0.0;
    for (var j = 0u; j < E; j = j + 1u) {
        xh_dot += w_xh[i * E + j] * w_embed[j * uniforms.vocab_size + input_idx];
    }

    var hh_dot = 0.0;
    for (var j = 0u; j < H; j = j + 1u) {
        hh_dot += w_hh[i * H + j] * h_in[j];
    }
    
    h_out[i] = tanh(xh_dot + hh_dot + b_h[i]);
}
"""

# --- KERNEL 2: Final logits calculation (Unchanged) ---
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

# --- KERNEL 3: Softmax and dy calculation (Unchanged) ---
SOFTMAX_DY_SHADER = """
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> temp_storage: array<f32>; // size 2: [max, sum]

struct Uniforms { vocab_size: u32, target_idx: u32 };
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(1, 1, 1)
fn compute_reductions() {
    var max_val = -3.4e38; // -inf
    for (var i = 0u; i < uniforms.vocab_size; i = i + 1u) {
        max_val = max(max_val, logits[i]);
    }
    
    var sum_val = 0.0;
    for (var i = 0u; i < uniforms.vocab_size; i = i + 1u) {
        sum_val += exp(logits[i] - max_val);
    }
    
    temp_storage[0] = max_val;
    temp_storage[1] = sum_val;
}

@compute @workgroup_size(256, 1, 1)
fn compute_dy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= uniforms.vocab_size) { return; }

    let max_val = temp_storage[0];
    let sum_val = temp_storage[1];

    let prob = exp(logits[i] - max_val) / sum_val;
    dy[i] = prob;
    if (i == uniforms.target_idx) {
        dy[i] = dy[i] - 1.0;
    }
}
"""

# --- KERNEL 4: Backward pass - calculating gradients for output layer ---
BACKWARD_HY_SHADER = """
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> h_final: array<f32>;
@group(0) @binding(2) var<storage, read_write> dW_hy: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> db_y: array<atomic<u32>>;

struct Uniforms { hidden_size: u32, vocab_size: u32 };
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size;
    let V = uniforms.vocab_size;
    let v = global_id.x;
    let h = global_id.y;

    if (v >= V || h >= H) { return; }
    
    let dW_hy_val = dy[v] * h_final[h];
    
    // Inlined atomicAdd_f32 for dW_hy
    {
        var old_val: u32;
        var new_val: u32;
        loop {
            old_val = atomicLoad(&dW_hy[v * H + h]);
            let new_f32 = bitcast<f32>(old_val) + dW_hy_val;
            new_val = bitcast<u32>(new_f32);
            let result = atomicCompareExchangeWeak(&dW_hy[v * H + h], old_val, new_val);
            if (result.exchanged) { break; }
        }
    }

    if (h == 0u) {
        // Inlined atomicAdd_f32 for db_y
        {
            var old_val: u32;
            var new_val: u32;
            loop {
                old_val = atomicLoad(&db_y[v]);
                let new_f32 = bitcast<f32>(old_val) + dy[v];
                new_val = bitcast<u32>(new_f32);
                let result = atomicCompareExchangeWeak(&db_y[v], old_val, new_val);
                if (result.exchanged) { break; }
            }
        }
    }
}
"""

# --- KERNEL 5: Initial dh calculation for BPTT (Unchanged) ---
INITIAL_DH_SHADER = """
@group(0) @binding(0) var<storage, read> W_hy: array<f32>;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dh_out: array<f32>;

struct Uniforms { hidden_size: u32, vocab_size: u32 };
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size;
    let V = uniforms.vocab_size;
    let i = global_id.x;
    if (i >= H) { return; }

    var dh_from_y = 0.0;
    for (var j = 0u; j < V; j = j + 1u) {
        dh_from_y += W_hy[j * H + i] * dy[j];
    }
    dh_out[i] = dh_from_y;
}
"""

# --- KERNEL 6: Backward pass - one step of BPTT ---
BPTT_STEP_SHADER = """
@group(0) @binding(0) var<storage, read> W_hh: array<f32>;
@group(0) @binding(1) var<storage, read> W_embed: array<f32>;
@group(0) @binding(2) var<storage, read> dh_next_in: array<f32>;
@group(0) @binding(3) var<storage, read> h_history: array<f32>;
@group(0) @binding(4) var<storage, read_write> dW_xh: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> dW_hh: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> db_h: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> dh_next_out: array<f32>;
@group(0) @binding(8) var<storage, read_write> dh_raw_out: array<f32>;
@group(0) @binding(9) var<storage, read> corpus: array<u32>;

struct Uniforms {
    hidden_size: u32,
    embedding_dim: u32,
    vocab_size: u32,
    t: u32,
    input_offset: u32,
};
@group(0) @binding(10) var<uniform> uniforms: Uniforms;

fn dtanh(y: f32) -> f32 { return 1.0 - y * y; }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size;
    let E = uniforms.embedding_dim;
    let V = uniforms.vocab_size;
    let t = uniforms.t;
    let i = global_id.x;
    if (i >= H) { return; }
    
    let h_t_offset = (t + 1u) * H;
    let h_prev_offset = t * H;
    let input_idx = corpus[uniforms.input_offset + t];

    let dh = dh_next_in[i];
    let dh_raw = dtanh(h_history[h_t_offset + i]) * dh;

    // Inlined atomicAdd_f32 for db_h
    {
        var old_val: u32;
        loop {
            old_val = atomicLoad(&db_h[i]);
            let new_f32 = bitcast<f32>(old_val) + dh_raw;
            let new_val = bitcast<u32>(new_f32);
            let result = atomicCompareExchangeWeak(&db_h[i], old_val, new_val);
            if (result.exchanged) { break; }
        }
    }

    for (var j = 0u; j < E; j = j + 1u) {
        let x_embed_t_j = W_embed[j * V + input_idx];
        // Inlined atomicAdd_f32 for dW_xh
        {
            var old_val: u32;
            loop {
                old_val = atomicLoad(&dW_xh[i * E + j]);
                let new_f32 = bitcast<f32>(old_val) + dh_raw * x_embed_t_j;
                let new_val = bitcast<u32>(new_f32);
                let result = atomicCompareExchangeWeak(&dW_xh[i * E + j], old_val, new_val);
                if (result.exchanged) { break; }
            }
        }
    }
    for (var j = 0u; j < H; j = j + 1u) {
        // Inlined atomicAdd_f32 for dW_hh
        {
            var old_val: u32;
            loop {
                old_val = atomicLoad(&dW_hh[i * H + j]);
                let new_f32 = bitcast<f32>(old_val) + dh_raw * h_history[h_prev_offset + j];
                let new_val = bitcast<u32>(new_f32);
                let result = atomicCompareExchangeWeak(&dW_hh[i * H + j], old_val, new_val);
                if (result.exchanged) { break; }
            }
        }
    }

    var dh_next_val = 0.0;
    for (var j = 0u; j < H; j = j + 1u) {
        dh_next_val += W_hh[j * H + i] * dh_raw;
    }
    dh_next_out[i] = dh_next_val;
    dh_raw_out[i] = dh_raw;
}
"""

# --- KERNEL 7: Calculates dx_embed and updates dW_embed ---
DX_EMBED_AND_UPDATE_SHADER = """
@group(0) @binding(0) var<storage, read> W_xh: array<f32>;
@group(0) @binding(1) var<storage, read> dh_raw: array<f32>;
@group(0) @binding(2) var<storage, read_write> dW_embed: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> corpus: array<u32>;

struct Uniforms {
    hidden_size: u32,
    embedding_dim: u32,
    vocab_size: u32,
    input_offset: u32,
};
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size;
    let E = uniforms.embedding_dim;
    let V = uniforms.vocab_size;
    let j = global_id.x;
    if (j >= E) { return; }

    let input_idx = corpus[uniforms.input_offset];

    var dx_embed_j = 0.0;
    for (var i = 0u; i < H; i = i + 1u) {
        dx_embed_j += W_xh[i * E + j] * dh_raw[i];
    }
    
    // Inlined atomicAdd_f32 for dW_embed
    {
        var old_val: u32;
        loop {
            old_val = atomicLoad(&dW_embed[j * V + input_idx]);
            let new_f32 = bitcast<f32>(old_val) + dx_embed_j;
            let new_val = bitcast<u32>(new_f32);
            let result = atomicCompareExchangeWeak(&dW_embed[j * V + input_idx], old_val, new_val);
            if (result.exchanged) { break; }
        }
    }
}
"""

# --- KERNEL 8: Update weights using calculated gradients (Corrected for atomics) ---
UPDATE_WEIGHTS_SHADER = """
@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<atomic<u32>>;

struct Uniforms { learning_rate: f32 };
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

fn clip(val: f32, min_val: f32, max_val: f32) -> f32 {
    return max(min_val, min(val, max_val));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&param)) { return; }
    
    let grad_f32 = bitcast<f32>(atomicLoad(&grad[i]));
    let clipped_grad = clip(grad_f32, -5.0, 5.0);
    param[i] = param[i] - uniforms.learning_rate * clipped_grad;
}
"""

# --- KERNEL 9: Calculate Cross-Entropy Loss (NEW) ---
LOSS_SHADER = """
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> loss_out: array<f32>;

struct Uniforms { vocab_size: u32, target_idx: u32 };
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(1, 1, 1)
fn main() {
    var max_val = -3.4e38; // -inf
    for (var i = 0u; i < uniforms.vocab_size; i = i + 1u) {
        max_val = max(max_val, logits[i]);
    }
    
    var sum_val = 0.0;
    for (var i = 0u; i < uniforms.vocab_size; i = i + 1u) {
        sum_val += exp(logits[i] - max_val);
    }

    let log_prob = (logits[uniforms.target_idx] - max_val) - log(sum_val);
    loss_out[0] = -log_prob;
}
"""

# --- KERNEL 10: Full Softmax for generation (NEW) ---
SOFTMAX_SHADER = """
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> temp_storage: array<f32>; // size 2: [max, sum]

struct Uniforms { vocab_size: u32 };
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(1, 1, 1)
fn compute_reductions() {
    var max_val = -3.4e38;
    for (var i = 0u; i < uniforms.vocab_size; i = i + 1u) {
        max_val = max(max_val, logits[i]);
    }
    
    var sum_val = 0.0;
    for (var i = 0u; i < uniforms.vocab_size; i = i + 1u) {
        sum_val += exp(logits[i] - max_val);
    }
    
    temp_storage[0] = max_val;
    temp_storage[1] = sum_val;
}

@compute @workgroup_size(256, 1, 1)
fn compute_probs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= uniforms.vocab_size) { return; }

    let max_val = temp_storage[0];
    let sum_val = temp_storage[1];
    
    probs_out[i] = exp(logits[i] - max_val) / sum_val;
}
"""

# --- KERNEL 11: GPU Sampling (NEW) ---
SAMPLING_SHADER = """
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_idx: array<u32>;

struct Uniforms { vocab_size: u32, rand_u: f32 };
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(1, 1, 1)
fn main() {
    var cum_prob = 0.0;
    for (var i = 0u; i < uniforms.vocab_size; i = i + 1u) {
        cum_prob += probs[i];
        if (uniforms.rand_u < cum_prob) {
            out_idx[0] = i;
            return;
        }
    }
    out_idx[0] = uniforms.vocab_size - 1u;
}
"""

def create_compute_pipeline(shader_code, bind_group_layouts, entry_point="main"):
    shader_module = device.create_shader_module(code=shader_code)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=bind_group_layouts)
    return device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": entry_point}
    )

# ==============================================================================
# OPTIMIZED MODEL CLASS
# ==============================================================================

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        self.device = device
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

        # --- Resources for optimized generation/warmup ---
        self.gen_h_out = device.create_buffer(size=hidden_size * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        self.gen_input = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
            
        self._create_pipelines()

    def _create_pipelines(self):
        b = wgpu.BufferBindingType
        s = wgpu.ShaderStage
        
        ro_storage = {"type": b.read_only_storage}
        rw_storage = {"type": b.storage}
        uniform = {"type": b.uniform}
        
        self.bgls = {
            'rnn_step': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 3, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 4, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 5, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 6, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 7, "visibility": s.COMPUTE, "buffer": uniform},
            ]),
            'final_logits': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 3, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 4, "visibility": s.COMPUTE, "buffer": uniform},
            ]),
            'softmax_dy': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 3, "visibility": s.COMPUTE, "buffer": uniform},
            ]),
            'backward_hy': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 3, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 4, "visibility": s.COMPUTE, "buffer": uniform},
            ]),
             'initial_dh': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 3, "visibility": s.COMPUTE, "buffer": uniform},
            ]),
            'bptt_step': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 3, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 4, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 5, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 6, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 7, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 8, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 9, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 10, "visibility": s.COMPUTE, "buffer": uniform},
            ]),
            'dx_embed_update': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 3, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 4, "visibility": s.COMPUTE, "buffer": uniform},
            ]),
            'update_weights': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": uniform},
            ]),
            'loss': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": uniform},
            ]),
            'softmax': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 3, "visibility": s.COMPUTE, "buffer": uniform},
            ]),
            'sampling': device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": s.COMPUTE, "buffer": ro_storage},
                {"binding": 1, "visibility": s.COMPUTE, "buffer": rw_storage},
                {"binding": 2, "visibility": s.COMPUTE, "buffer": uniform},
            ]),
        }
        self.pipelines = {
            'rnn_step': create_compute_pipeline(RNN_STEP_SHADER, [self.bgls['rnn_step']]),
            'final_logits': create_compute_pipeline(FINAL_LOGITS_SHADER, [self.bgls['final_logits']]),
            'softmax_dy_reductions': create_compute_pipeline(SOFTMAX_DY_SHADER, [self.bgls['softmax_dy']], entry_point="compute_reductions"),
            'softmax_dy_compute': create_compute_pipeline(SOFTMAX_DY_SHADER, [self.bgls['softmax_dy']], entry_point="compute_dy"),
            'backward_hy': create_compute_pipeline(BACKWARD_HY_SHADER, [self.bgls['backward_hy']]),
            'initial_dh': create_compute_pipeline(INITIAL_DH_SHADER, [self.bgls['initial_dh']]),
            'bptt_step': create_compute_pipeline(BPTT_STEP_SHADER, [self.bgls['bptt_step']]),
            'dx_embed_update': create_compute_pipeline(DX_EMBED_AND_UPDATE_SHADER, [self.bgls['dx_embed_update']]),
            'update_weights': create_compute_pipeline(UPDATE_WEIGHTS_SHADER, [self.bgls['update_weights']]),
            'loss': create_compute_pipeline(LOSS_SHADER, [self.bgls['loss']]),
            'softmax_reductions': create_compute_pipeline(SOFTMAX_SHADER, [self.bgls['softmax']], entry_point="compute_reductions"),
            'softmax_compute': create_compute_pipeline(SOFTMAX_SHADER, [self.bgls['softmax']], entry_point="compute_probs"),
            'sampling': create_compute_pipeline(SAMPLING_SHADER, [self.bgls['sampling']]),
        }

    def get_initial_hidden_state_gpu(self):
        H = self.hidden_size
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        h_buffer = device.create_buffer(size=H * 4, usage=usage)
        self.zero_buffer(h_buffer)
        return h_buffer
        
    def zero_buffer(self, buffer):
        command_encoder = device.create_command_encoder()
        command_encoder.clear_buffer(buffer, 0, buffer.size)
        device.queue.submit([command_encoder.finish()])

    def update_hidden_state(self, h_source, h_dest):
        H = self.hidden_size
        S = (h_source.size // (H * 4)) - 1
        
        command_encoder = device.create_command_encoder()
        command_encoder.copy_buffer_to_buffer(h_source, S * H * 4, h_dest, 0, h_dest.size)
        device.queue.submit([command_encoder.finish()])

    def _get_params(self):
        cpu_params = {}
        for name, gpu_buf in self.params_gpu.items():
            data = device.queue.read_buffer(gpu_buf).cast("f")
            arr = np.array(data).reshape(self.params[name].shape)
            cpu_params[name] = arr.tolist()
        return cpu_params
        
    def _set_params(self, params):
        for name, data in params.items():
            arr = np.array(data, dtype=np.float32)
            self.params[name] = arr
            device.queue.write_buffer(self.params_gpu[name], 0, arr.tobytes())
            
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

    def forward_sequence(self, corpus_gpu, h_prev_gpu, offset, seq_length):
        S, H, E, V = seq_length, self.hidden_size, self.embedding_dim, self.vocab_size
        
        # FIX: The h_history_gpu buffer is a destination for copies, so it needs COPY_DST.
        h_history_gpu = device.create_buffer(
            size=(S + 1) * H * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        )
        h_ping = h_prev_gpu
        h_pong = device.create_buffer(size=H * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        
        command_encoder = device.create_command_encoder()
        command_encoder.copy_buffer_to_buffer(h_ping, 0, h_history_gpu, 0, h_ping.size)

        for t in range(seq_length):
            uniform_data = np.array([H, E, V, offset + t], dtype=np.uint32)
            uniform_buffer = device.create_buffer_with_data(data=uniform_data, usage=wgpu.BufferUsage.UNIFORM)
            
            bind_group = device.create_bind_group(layout=self.bgls['rnn_step'], entries=[
                {"binding": 0, "resource": {"buffer": self.params_gpu['W_embed']}},
                {"binding": 1, "resource": {"buffer": self.params_gpu['W_xh']}},
                {"binding": 2, "resource": {"buffer": self.params_gpu['W_hh']}},
                {"binding": 3, "resource": {"buffer": self.params_gpu['b_h']}},
                {"binding": 4, "resource": {"buffer": h_ping}},
                {"binding": 5, "resource": {"buffer": h_pong}},
                {"binding": 6, "resource": {"buffer": corpus_gpu}},
                {"binding": 7, "resource": {"buffer": uniform_buffer}},
            ])
            
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.pipelines['rnn_step'])
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(math.ceil(H / 64))
            compute_pass.end()
            
            command_encoder.copy_buffer_to_buffer(h_pong, 0, h_history_gpu, (t + 1) * H * 4, h_pong.size)
            h_ping, h_pong = h_pong, h_ping
        
        final_h_gpu = h_ping
        logits_gpu = device.create_buffer(size=V * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        
        uniform_data_logits = np.array([H, V], dtype=np.uint32)
        uniform_buffer_logits = device.create_buffer_with_data(data=uniform_data_logits, usage=wgpu.BufferUsage.UNIFORM)
        
        bind_group_logits = device.create_bind_group(layout=self.bgls['final_logits'], entries=[
            {"binding": 0, "resource": {"buffer": self.params_gpu['W_hy']}},
            {"binding": 1, "resource": {"buffer": self.params_gpu['b_y']}},
            {"binding": 2, "resource": {"buffer": final_h_gpu}},
            {"binding": 3, "resource": {"buffer": logits_gpu}},
            {"binding": 4, "resource": {"buffer": uniform_buffer_logits}},
        ])

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['final_logits'])
        compute_pass.set_bind_group(0, bind_group_logits)
        compute_pass.dispatch_workgroups(math.ceil(V / 64))
        compute_pass.end()
        device.queue.submit([command_encoder.finish()])
        
        return logits_gpu, h_history_gpu
        
    def calculate_loss_gpu(self, logits_gpu, target_idx):
        V = self.vocab_size
        loss_out_gpu = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

        uniform_data = np.array([V, target_idx], dtype=np.uint32)
        uniform_buffer = device.create_buffer_with_data(data=uniform_data, usage=wgpu.BufferUsage.UNIFORM)

        bind_group = device.create_bind_group(layout=self.bgls['loss'], entries=[
            {"binding": 0, "resource": {"buffer": logits_gpu}},
            {"binding": 1, "resource": {"buffer": loss_out_gpu}},
            {"binding": 2, "resource": {"buffer": uniform_buffer}},
        ])

        command_encoder = device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['loss'])
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        device.queue.submit([command_encoder.finish()])

        loss_data = device.queue.read_buffer(loss_out_gpu).cast("f")
        return loss_data[0]

    def backward_sequence(self, corpus_gpu, offset, seq_length, target_idx, logits_gpu, h_history_gpu):
        S, H, E, V = seq_length, self.hidden_size, self.embedding_dim, self.vocab_size
        
        command_encoder = device.create_command_encoder()
        for grad_buf in self.grads_gpu.values():
            command_encoder.clear_buffer(grad_buf)

        # 1. Softmax and dy calculation
        dy_gpu = device.create_buffer(size=V*4, usage=wgpu.BufferUsage.STORAGE)
        temp_softmax_gpu = device.create_buffer(size=2*4, usage=wgpu.BufferUsage.STORAGE)
        uniform_data_softmax = np.array([V, target_idx], dtype=np.uint32)
        uniform_buffer_softmax = device.create_buffer_with_data(data=uniform_data_softmax, usage=wgpu.BufferUsage.UNIFORM)
        bg_softmax = device.create_bind_group(layout=self.bgls['softmax_dy'], entries=[
            {"binding": 0, "resource": {"buffer": logits_gpu}},
            {"binding": 1, "resource": {"buffer": dy_gpu}},
            {"binding": 2, "resource": {"buffer": temp_softmax_gpu}},
            {"binding": 3, "resource": {"buffer": uniform_buffer_softmax}},
        ])
        
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['softmax_dy_reductions'])
        compute_pass.set_bind_group(0, bg_softmax)
        compute_pass.dispatch_workgroups(1)
        compute_pass.set_pipeline(self.pipelines['softmax_dy_compute'])
        compute_pass.set_bind_group(0, bg_softmax)
        compute_pass.dispatch_workgroups(math.ceil(V / 256))
        compute_pass.end()
        
        # 2. Gradients for output layer (W_hy, b_y)
        uniform_data_hy = np.array([H, V], dtype=np.uint32)
        uniform_buffer_hy = device.create_buffer_with_data(data=uniform_data_hy, usage=wgpu.BufferUsage.UNIFORM)
        bg_hy = device.create_bind_group(layout=self.bgls['backward_hy'], entries=[
            {"binding": 0, "resource": {"buffer": dy_gpu}},
            {"binding": 1, "resource": {"buffer": h_history_gpu, "offset": S * H * 4, "size": H * 4}},
            {"binding": 2, "resource": {"buffer": self.grads_gpu['W_hy']}},
            {"binding": 3, "resource": {"buffer": self.grads_gpu['b_y']}},
            {"binding": 4, "resource": {"buffer": uniform_buffer_hy}},
        ])
        
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['backward_hy'])
        compute_pass.set_bind_group(0, bg_hy)
        compute_pass.dispatch_workgroups(math.ceil(V / 8), math.ceil(H / 8))
        compute_pass.end()
        
        # 3. Initial dh for BPTT
        dh_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        dh_ping = device.create_buffer(size=H*4, usage=dh_usage)
        dh_pong = device.create_buffer(size=H*4, usage=dh_usage)

        bg_initial_dh = device.create_bind_group(layout=self.bgls['initial_dh'], entries=[
            {"binding": 0, "resource": {"buffer": self.params_gpu['W_hy']}},
            {"binding": 1, "resource": {"buffer": dy_gpu}},
            {"binding": 2, "resource": {"buffer": dh_ping}},
            {"binding": 3, "resource": {"buffer": uniform_buffer_hy}},
        ])
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['initial_dh'])
        compute_pass.set_bind_group(0, bg_initial_dh)
        compute_pass.dispatch_workgroups(math.ceil(H / 64))
        compute_pass.end()

        # 4. BPTT loop
        dh_raw_gpu = device.create_buffer(size=H*4, usage=wgpu.BufferUsage.STORAGE)
        for t in reversed(range(seq_length)):
            uniform_data_bptt = np.array([H, E, V, t, offset], dtype=np.uint32)
            uniform_buffer_bptt = device.create_buffer_with_data(data=uniform_data_bptt, usage=wgpu.BufferUsage.UNIFORM)
            
            bg_bptt = device.create_bind_group(layout=self.bgls['bptt_step'], entries=[
                {"binding": 0, "resource": {"buffer": self.params_gpu['W_hh']}},
                {"binding": 1, "resource": {"buffer": self.params_gpu['W_embed']}},
                {"binding": 2, "resource": {"buffer": dh_ping}},
                {"binding": 3, "resource": {"buffer": h_history_gpu}},
                {"binding": 4, "resource": {"buffer": self.grads_gpu['W_xh']}},
                {"binding": 5, "resource": {"buffer": self.grads_gpu['W_hh']}},
                {"binding": 6, "resource": {"buffer": self.grads_gpu['b_h']}},
                {"binding": 7, "resource": {"buffer": dh_pong}},
                {"binding": 8, "resource": {"buffer": dh_raw_gpu}},
                {"binding": 9, "resource": {"buffer": corpus_gpu}},
                {"binding": 10, "resource": {"buffer": uniform_buffer_bptt}},
            ])
            
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.pipelines['bptt_step'])
            compute_pass.set_bind_group(0, bg_bptt)
            compute_pass.dispatch_workgroups(math.ceil(H / 64))
            compute_pass.end()
            
            # 5. Update dW_embed
            uniform_data_embed = np.array([H, E, V, offset + t], dtype=np.uint32)
            uniform_buffer_embed = device.create_buffer_with_data(data=uniform_data_embed, usage=wgpu.BufferUsage.UNIFORM)
            bg_embed = device.create_bind_group(layout=self.bgls['dx_embed_update'], entries=[
                {"binding": 0, "resource": {"buffer": self.params_gpu['W_xh']}},
                {"binding": 1, "resource": {"buffer": dh_raw_gpu}},
                {"binding": 2, "resource": {"buffer": self.grads_gpu['W_embed']}},
                {"binding": 3, "resource": {"buffer": corpus_gpu}},
                {"binding": 4, "resource": {"buffer": uniform_buffer_embed}},
            ])
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.pipelines['dx_embed_update'])
            compute_pass.set_bind_group(0, bg_embed)
            compute_pass.dispatch_workgroups(math.ceil(E / 64))
            compute_pass.end()

            dh_ping, dh_pong = dh_pong, dh_ping

        device.queue.submit([command_encoder.finish()])

    def update_weights(self, learning_rate):
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

    def forward_step(self, input_idx, h_gpu):
        """Runs an optimized single forward step, reusing GPU resources."""
        H, E, V = self.hidden_size, self.embedding_dim, self.vocab_size
        
        device.queue.write_buffer(self.gen_input, 0, np.array([input_idx], dtype=np.uint32))
        
        command_encoder = device.create_command_encoder()
        uniform_data = np.array([H, E, V, 0], dtype=np.uint32)
        uniform_buffer = device.create_buffer_with_data(data=uniform_data, usage=wgpu.BufferUsage.UNIFORM)
        
        bind_group = device.create_bind_group(layout=self.bgls['rnn_step'], entries=[
            {"binding": 0, "resource": {"buffer": self.params_gpu['W_embed']}},
            {"binding": 1, "resource": {"buffer": self.params_gpu['W_xh']}},
            {"binding": 2, "resource": {"buffer": self.params_gpu['W_hh']}},
            {"binding": 3, "resource": {"buffer": self.params_gpu['b_h']}},
            {"binding": 4, "resource": {"buffer": h_gpu}},
            {"binding": 5, "resource": {"buffer": self.gen_h_out}},
            {"binding": 6, "resource": {"buffer": self.gen_input}},
            {"binding": 7, "resource": {"buffer": uniform_buffer}},
        ])
        
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['rnn_step'])
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(math.ceil(H / 64))
        compute_pass.end()
        
        command_encoder.copy_buffer_to_buffer(self.gen_h_out, 0, h_gpu, 0, h_gpu.size)
        device.queue.submit([command_encoder.finish()])
        return h_gpu

    def generate_step(self, input_idx, h_gpu):
        """Performs a full generation step in a single batched GPU submission."""
        H, V = self.hidden_size, self.vocab_size

        device.queue.write_buffer(self.gen_input, 0, np.array([input_idx], dtype=np.uint32))
        command_encoder = device.create_command_encoder()
        
        # === 1. Forward Step ===
        uniform_fwd_data = np.array([H, self.embedding_dim, V, 0], dtype=np.uint32)
        uniform_fwd_buf = device.create_buffer_with_data(data=uniform_fwd_data, usage=wgpu.BufferUsage.UNIFORM)
        bg_fwd = device.create_bind_group(layout=self.bgls['rnn_step'], entries=[
            {"binding": 0, "resource": {"buffer": self.params_gpu['W_embed']}},
            {"binding": 1, "resource": {"buffer": self.params_gpu['W_xh']}},
            {"binding": 2, "resource": {"buffer": self.params_gpu['W_hh']}},
            {"binding": 3, "resource": {"buffer": self.params_gpu['b_h']}},
            {"binding": 4, "resource": {"buffer": h_gpu}},
            {"binding": 5, "resource": {"buffer": self.gen_h_out}},
            {"binding": 6, "resource": {"buffer": self.gen_input}},
            {"binding": 7, "resource": {"buffer": uniform_fwd_buf}},
        ])
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['rnn_step'])
        compute_pass.set_bind_group(0, bg_fwd)
        compute_pass.dispatch_workgroups(math.ceil(H / 64))
        compute_pass.end()
        command_encoder.copy_buffer_to_buffer(self.gen_h_out, 0, h_gpu, 0, h_gpu.size)
        
        # === 2. Calculate Logits ===
        logits_gpu = device.create_buffer(size=V*4, usage=wgpu.BufferUsage.STORAGE)
        uniform_logits_data = np.array([H, V], dtype=np.uint32)
        uniform_logits_buf = device.create_buffer_with_data(data=uniform_logits_data, usage=wgpu.BufferUsage.UNIFORM)
        bg_logits = device.create_bind_group(layout=self.bgls['final_logits'], entries=[
            {"binding": 0, "resource": {"buffer": self.params_gpu['W_hy']}},
            {"binding": 1, "resource": {"buffer": self.params_gpu['b_y']}},
            {"binding": 2, "resource": {"buffer": h_gpu}},
            {"binding": 3, "resource": {"buffer": logits_gpu}},
            {"binding": 4, "resource": {"buffer": uniform_logits_buf}},
        ])
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['final_logits'])
        compute_pass.set_bind_group(0, bg_logits)
        compute_pass.dispatch_workgroups(math.ceil(V / 64))
        compute_pass.end()
        
        # === 3. Compute Softmax ===
        probs_gpu = device.create_buffer(size=V*4, usage=wgpu.BufferUsage.STORAGE)
        temp_softmax_gpu = device.create_buffer(size=2*4, usage=wgpu.BufferUsage.STORAGE)
        uniform_softmax_data = np.array([V], dtype=np.uint32)
        uniform_softmax_buf = device.create_buffer_with_data(data=uniform_softmax_data, usage=wgpu.BufferUsage.UNIFORM)
        bg_softmax = device.create_bind_group(layout=self.bgls['softmax'], entries=[
            {"binding": 0, "resource": {"buffer": logits_gpu}},
            {"binding": 1, "resource": {"buffer": probs_gpu}},
            {"binding": 2, "resource": {"buffer": temp_softmax_gpu}},
            {"binding": 3, "resource": {"buffer": uniform_softmax_buf}},
        ])
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['softmax_reductions'])
        compute_pass.set_bind_group(0, bg_softmax)
        compute_pass.dispatch_workgroups(1)
        compute_pass.set_pipeline(self.pipelines['softmax_compute'])
        compute_pass.set_bind_group(0, bg_softmax)
        compute_pass.dispatch_workgroups(math.ceil(V / 256))
        compute_pass.end()
        
        # === 4. Sample from Distribution ===
        rand_u = random.random()
        out_idx_gpu = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        
        uniform_data_sample = np.zeros(2, dtype=np.uint32)
        uniform_data_sample[0] = V
        uniform_data_sample.view(np.float32)[1] = rand_u
        
        uniform_sample_buf = device.create_buffer_with_data(data=uniform_data_sample, usage=wgpu.BufferUsage.UNIFORM)
        bg_sample = device.create_bind_group(layout=self.bgls['sampling'], entries=[
            {"binding": 0, "resource": {"buffer": probs_gpu}},
            {"binding": 1, "resource": {"buffer": out_idx_gpu}},
            {"binding": 2, "resource": {"buffer": uniform_sample_buf}},
        ])
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines['sampling'])
        compute_pass.set_bind_group(0, bg_sample)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()
        
        # === 5. Submit All Commands and Prepare for Readback ===
        device.queue.submit([command_encoder.finish()])
        
        # === 6. Read Result Directly from GPU Source Buffer ===
        idx_data = device.queue.read_buffer(out_idx_gpu).cast("I")
        return idx_data[0]
        
        # === 7. Read Result Back to CPU ===
        idx_data = device.queue.read_buffer(staging_buffer).cast("I")
        return idx_data[0]