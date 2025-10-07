import wgpu

# ==============================================================================
# OPTIMIZED COMPUTE SHADERS (CHANGED)
# ==============================================================================

# --- KERNEL 1: Forward pass RNN step (Workgroup size increased) ---
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

// Changed workgroup size from 64 to 256 for better occupancy
@compute @workgroup_size(256, 1, 1)
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

# --- KERNEL 2: Final logits calculation (Workgroup size increased) ---
FINAL_LOGITS_SHADER = """
@group(0) @binding(0) var<storage, read> w_hy: array<f32>;
@group(0) @binding(1) var<storage, read> b_y: array<f32>;
@group(0) @binding(2) var<storage, read> h_final: array<f32>;
@group(0) @binding(3) var<storage, read_write> logits_out: array<f32>;

struct Uniforms { hidden_size: u32, vocab_size: u32 };
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

// Changed workgroup size from 64 to 256 for better occupancy
@compute @workgroup_size(256, 1, 1)
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

# --- KERNEL 3: Softmax and dy calculation (Rewritten for Parallel Reduction) ---
SOFTMAX_DY_SHADER = """
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> temp_storage: array<f32>;

struct Uniforms { vocab_size: u32, target_idx: u32 };
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

const WG_SIZE = 256u;
var<workgroup> wg_buffer: array<f32, WG_SIZE>; // Shared memory for reduction

@compute @workgroup_size(WG_SIZE, 1, 1)
fn compute_reductions(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(global_invocation_id) global_id: vec3<u32>) {
    let V = uniforms.vocab_size;
    let local_x = local_id.x;

    // Phase 1: Find Max Value (Parallel Reduction)
    var max_val_local = -3.4e38;
    for (var i = global_id.x; i < V; i = i + WG_SIZE) {
        max_val_local = max(max_val_local, logits[i]);
    }
    wg_buffer[local_x] = max_val_local;
    workgroupBarrier();

    // Workgroup reduction for max
    var s = WG_SIZE / 2u;
    while (s > 0u) {
        if (local_x < s) {
            wg_buffer[local_x] = max(wg_buffer[local_x], wg_buffer[local_x + s]);
        }
        s = s / 2u;
        workgroupBarrier();
    }

    let max_val = wg_buffer[0];

    // Phase 2: Find Sum of Exponentials (Parallel Reduction)
    var sum_val_local = 0.0;
    for (var i = global_id.x; i < V; i = i + WG_SIZE) {
        sum_val_local += exp(logits[i] - max_val);
    }
    wg_buffer[local_x] = sum_val_local;
    workgroupBarrier();

    // Workgroup reduction for sum
    s = WG_SIZE / 2u;
    while (s > 0u) {
        if (local_x < s) {
            wg_buffer[local_x] += wg_buffer[local_x + s];
        }
        s = s / 2u;
        workgroupBarrier();
    }

    // Phase 3: Final Write
    if (local_x == 0u) {
        temp_storage[0] = max_val;
        temp_storage[1] = wg_buffer[0];
    }
}

@compute @workgroup_size(256, 1, 1)
fn compute_dy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= uniforms.vocab_size) { return; }
    let max_val = temp_storage[0];
    let sum_val = temp_storage[1];
    let prob = exp(logits[i] - max_val) / sum_val;
    dy[i] = prob;
    if (i == uniforms.target_idx) { dy[i] = dy[i] - 1.0; }
}
"""

# --- KERNEL 4: Backward pass - output layer (Workgroup size increased) ---
BACKWARD_HY_SHADER = """
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> h_final: array<f32>;
@group(0) @binding(2) var<storage, read_write> dW_hy: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> db_y: array<atomic<u32>>;

struct Uniforms { hidden_size: u32, vocab_size: u32 };
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

// Changed workgroup size from (8, 8, 1) to (16, 16, 1) for better occupancy in the outer product
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let V = uniforms.vocab_size; let H = uniforms.hidden_size;
    let v = global_id.x; let h = global_id.y;
    if (v >= V || h >= H) { return; }
    let dW_hy_val = dy[v] * h_final[h];
    var old_val: u32; loop { old_val = atomicLoad(&dW_hy[v * H + h]); let new_f32 = bitcast<f32>(old_val) + dW_hy_val; let new_val = bitcast<u32>(new_f32); let result = atomicCompareExchangeWeak(&dW_hy[v * H + h], old_val, new_val); if (result.exchanged) { break; } }
    if (h == 0u) { loop { old_val = atomicLoad(&db_y[v]); let new_f32 = bitcast<f32>(old_val) + dy[v]; let new_val = bitcast<u32>(new_f32); let result = atomicCompareExchangeWeak(&db_y[v], old_val, new_val); if (result.exchanged) { break; } } }
}
"""

# --- KERNEL 5: Initial dh calculation for BPTT (Workgroup size increased) ---
INITIAL_DH_SHADER = """
@group(0) @binding(0) var<storage, read> W_hy: array<f32>;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dh_out: array<f32>;

struct Uniforms { hidden_size: u32, vocab_size: u32 };
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// Changed workgroup size from 64 to 256 for better occupancy
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size; let V = uniforms.vocab_size;
    let i = global_id.x; if (i >= H) { return; }
    var dh_from_y = 0.0;
    for (var j = 0u; j < V; j = j + 1u) { dh_from_y += W_hy[j * H + i] * dy[j]; }
    dh_out[i] = dh_from_y;
}
"""

# --- KERNEL 6: Backward pass - one step of BPTT (Workgroup size increased for better occupancy/latency hiding) ---
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
    padded_h_size_bytes: u32,
};
@group(0) @binding(10) var<uniform> uniforms: Uniforms;

fn dtanh(y: f32) -> f32 { return 1.0 - y * y; }

// Increased workgroup size from 64 to 256
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size; let E = uniforms.embedding_dim; let V = uniforms.vocab_size;
    let t = uniforms.t; let i = global_id.x; if (i >= H) { return; }

    let h_t_offset = ((t + 1u) * uniforms.padded_h_size_bytes) / 4u;
    let h_prev_offset = (t * uniforms.padded_h_size_bytes) / 4u;
    let input_idx = corpus[uniforms.input_offset + t];

    let dh = dh_next_in[i];
    let dh_raw = dtanh(h_history[h_t_offset + i]) * dh;

    var old_val: u32;
    // dW_xh accumulation (loop over E=128)
    loop { old_val = atomicLoad(&db_h[i]); let new_f32 = bitcast<f32>(old_val) + dh_raw; let new_val = bitcast<u32>(new_f32); let result = atomicCompareExchangeWeak(&db_h[i], old_val, new_val); if (result.exchanged) { break; } }
    for (var j = 0u; j < E; j = j + 1u) { let x_embed_t_j = W_embed[j * V + input_idx]; loop { old_val = atomicLoad(&dW_xh[i * E + j]); let new_f32 = bitcast<f32>(old_val) + dh_raw * x_embed_t_j; let new_val = bitcast<u32>(new_f32); let result = atomicCompareExchangeWeak(&dW_xh[i * E + j], old_val, new_val); if (result.exchanged) { break; } } }
    // dW_hh accumulation (loop over H=256)
    for (var j = 0u; j < H; j = j + 1u) { loop { old_val = atomicLoad(&dW_hh[i * H + j]); let new_f32 = bitcast<f32>(old_val) + dh_raw * h_history[h_prev_offset + j]; let new_val = bitcast<u32>(new_f32); let result = atomicCompareExchangeWeak(&dW_hh[i * H + j], old_val, new_val); if (result.exchanged) { break; } } }

    var dh_next_val = 0.0;
    for (var j = 0u; j < H; j = j + 1u) { dh_next_val += W_hh[j * H + i] * dh_raw; }
    dh_next_out[i] = dh_next_val;
    dh_raw_out[i] = dh_raw;
}
"""

# --- KERNEL 7: Calculates dx_embed and updates dW_embed (Workgroup size increased) ---
DX_EMBED_AND_UPDATE_SHADER = """
@group(0) @binding(0) var<storage, read> W_xh: array<f32>;
@group(0) @binding(1) var<storage, read> dh_raw: array<f32>;
@group(0) @binding(2) var<storage, read_write> dW_embed: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> corpus: array<u32>;

struct Uniforms { hidden_size: u32, embedding_dim: u32, vocab_size: u32, input_offset: u32 };
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

// Changed workgroup size from 64 to 256 for better occupancy
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let H = uniforms.hidden_size; let E = uniforms.embedding_dim; let V = uniforms.vocab_size;
    let j = global_id.x; if (j >= E) { return; }
    let input_idx = corpus[uniforms.input_offset];
    var dx_embed_j = 0.0;
    for (var i = 0u; i < H; i = i + 1u) { dx_embed_j += W_xh[i * E + j] * dh_raw[i]; }
    var old_val: u32; loop { old_val = atomicLoad(&dW_embed[j * V + input_idx]); let new_f32 = bitcast<f32>(old_val) + dx_embed_j; let new_val = bitcast<u32>(new_f32); let result = atomicCompareExchangeWeak(&dW_embed[j * V + input_idx], old_val, new_val); if (result.exchanged) { break; } }
}
"""
# --- KERNEL 8: Update weights using calculated gradients (Unchanged) ---
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

# --- KERNEL 9: Calculate Cross-Entropy Loss (FIXED: Parallel Reduction) ---
LOSS_SHADER = """
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> loss_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> temp_storage: array<f32>; // Use a temporary buffer for reduction

struct Uniforms { vocab_size: u32, target_idx: u32 };
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

const WG_SIZE = 256u;
var<workgroup> wg_buffer: array<f32, WG_SIZE>; // Shared memory for reduction

// FIX: Rewritten to use parallel reduction, same logic as SOFTMAX_DY_SHADER's compute_reductions
@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(global_invocation_id) global_id: vec3<u32>) {
    let V = uniforms.vocab_size;
    let local_x = local_id.x;

    // Phase 1: Find Max Value (Parallel Reduction)
    var max_val_local = -3.4e38;
    for (var i = global_id.x; i < V; i = i + WG_SIZE) {
        max_val_local = max(max_val_local, logits[i]);
    }
    wg_buffer[local_x] = max_val_local;
    workgroupBarrier();

    var s = WG_SIZE / 2u;
    while (s > 0u) {
        if (local_x < s) {
            wg_buffer[local_x] = max(wg_buffer[local_x], wg_buffer[local_x + s]);
        }
        s = s / 2u;
        workgroupBarrier();
    }

    let max_val = wg_buffer[0];

    // Phase 2: Find Sum of Exponentials (Parallel Reduction)
    var sum_val_local = 0.0;
    for (var i = global_id.x; i < V; i = i + WG_SIZE) {
        sum_val_local += exp(logits[i] - max_val);
    }
    wg_buffer[local_x] = sum_val_local;
    workgroupBarrier();

    s = WG_SIZE / 2u;
    while (s > 0u) {
        if (local_x < s) {
            wg_buffer[local_x] += wg_buffer[local_x + s];
        }
        s = s / 2u;
        workgroupBarrier();
    }

    // Phase 3: Final Loss Calculation (only thread 0)
    if (local_x == 0u) {
        let sum_val = wg_buffer[0];
        let log_prob = (logits[uniforms.target_idx] - max_val) - log(sum_val);
        loss_out[0] = -log_prob;
    }
}
"""

# --- KERNEL 10: Full Softmax for generation (Rewritten for Parallel Reduction) ---
SOFTMAX_SHADER = """
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> temp_storage: array<f32>;

struct Uniforms { vocab_size: u32 };
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

const WG_SIZE = 256u;
var<workgroup> wg_buffer: array<f32, WG_SIZE>; // Shared memory for reduction

@compute @workgroup_size(WG_SIZE, 1, 1)
fn compute_reductions(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(global_invocation_id) global_id: vec3<u32>) {
    let V = uniforms.vocab_size;
    let local_x = local_id.x;

    // Phase 1: Find Max Value (Parallel Reduction)
    var max_val_local = -3.4e38;
    for (var i = global_id.x; i < V; i = i + WG_SIZE) {
        max_val_local = max(max_val_local, logits[i]);
    }
    wg_buffer[local_x] = max_val_local;
    workgroupBarrier();

    var s = WG_SIZE / 2u;
    while (s > 0u) {
        if (local_x < s) {
            wg_buffer[local_x] = max(wg_buffer[local_x], wg_buffer[local_x + s]);
        }
        s = s / 2u;
        workgroupBarrier();
    }

    let max_val = wg_buffer[0];

    // Phase 2: Find Sum of Exponentials (Parallel Reduction)
    var sum_val_local = 0.0;
    for (var i = global_id.x; i < V; i = i + WG_SIZE) {
        sum_val_local += exp(logits[i] - max_val);
    }
    wg_buffer[local_x] = sum_val_local;
    workgroupBarrier();

    s = WG_SIZE / 2u;
    while (s > 0u) {
        if (local_x < s) {
            wg_buffer[local_x] += wg_buffer[local_x + s];
        }
        s = s / 2u;
        workgroupBarrier();
    }

    // Phase 3: Final Write
    if (local_x == 0u) {
        temp_storage[0] = max_val;
        temp_storage[1] = wg_buffer[0];
    }
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

# --- KERNEL 11: GPU Sampling (Unchanged) ---
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

# --- KERNEL 12: Gradient Reset (NEW) ---
GRADIENT_RESET_SHADER = """
// Gradient buffer to reset (atomic<u32> buffer, which stores bitcast f32)
@group(0) @binding(0) var<storage, read_write> grad: array<atomic<u32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&grad)) { return; }

    // Writing 0u (the bitcast for 0.0f) is an atomic store, which is faster
    // and more scalable than a large CPU-to-GPU memory copy.
    atomicStore(&grad[i], 0u);
}
"""


# ==============================================================================
# WGPU INITIALIZATION
# ==============================================================================
def gpu_init():
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        device = adapter.request_device_sync()
        print(f"INFO: wgpu_math backend using GPU: {adapter.info.device}")
        return adapter, device
    except Exception as e:
        print(f"FATAL: Could not initialize WGPU device: {e}")
        print("Please ensure you have working Vulkan drivers.")
        exit(1)


def create_compute_pipeline(shader_code, bind_group_layouts, entry_point="main"):
    shader_module = device.create_shader_module(code=shader_code)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=bind_group_layouts
    )
    return device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": entry_point},
    )


adapter, device = gpu_init()
