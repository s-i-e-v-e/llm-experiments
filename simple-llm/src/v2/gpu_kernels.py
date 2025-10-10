# ============================================================================
# WGSL KERNELS
# ============================================================================

TILED_MATMUL_KERNEL = """
// Tiled matrix multiplication: C = A @ B
// Uses shared memory to reduce global memory accesses

struct MatmulParams {
    M: u32,  // Rows of A
    K: u32,  // Cols of A, Rows of B
    N: u32,  // Cols of B
}

@group(0) @binding(0) var<uniform> params: MatmulParams;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TILE_SIZE: u32 = 16u;
var<workgroup> tile_A: array<f32, 256>;  // 16x16 tile
var<workgroup> tile_B: array<f32, 256>;  // 16x16 tile

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum = 0.0;

    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load tile of A into shared memory
        let a_row = row;
        let a_col = t * TILE_SIZE + local_col;
        if (a_row < params.M && a_col < params.K) {
            tile_A[local_row * TILE_SIZE + local_col] = A[a_row * params.K + a_col];
        } else {
            tile_A[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = t * TILE_SIZE + local_row;
        let b_col = col;
        if (b_row < params.K && b_col < params.N) {
            tile_B[local_row * TILE_SIZE + local_col] = B[b_row * params.N + b_col];
        } else {
            tile_B[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Synchronize to ensure all threads have loaded their data
        workgroupBarrier();

        // Compute partial dot product using shared memory
        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tile_A[local_row * TILE_SIZE + k] * tile_B[k * TILE_SIZE + local_col];
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result
    if (row < params.M && col < params.N) {
        C[row * params.N + col] = sum;
    }
}
"""

LAYERNORM_KERNEL = """
struct NormParams {
    size: u32,
    n_elements: u32,
}

@group(0) @binding(0) var<uniform> params: NormParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const EPS: f32 = 1e-5;
const BLOCK_SIZE: u32 = 256u;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;
    let tid = local_id.x;
    let elem_idx = idx / params.size;

    if (elem_idx >= params.n_elements) {
        return;
    }

    let offset = elem_idx * params.size;

    // Compute mean using parallel reduction
    var sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        sum += input[offset + i];
    }
    shared_data[tid] = sum;
    workgroupBarrier();

    // Reduction
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }

    let mean = shared_data[0] / f32(params.size);

    // Compute variance
    var var_sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        let diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    shared_data[tid] = var_sum;
    workgroupBarrier();

    // Reduction
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }

    let variance = shared_data[0] / f32(params.size);
    let inv_std = 1.0 / sqrt(variance + EPS);

    // Normalize and apply affine transformation
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        let normalized = (input[offset + i] - mean) * inv_std;
        output[offset + i] = normalized * gamma[i] + beta[i];
    }
}
"""

GELU_KERNEL = """
struct GeluParams {
    size: u32,
}

@group(0) @binding(0) var<uniform> params: GeluParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let x = input[idx];
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}
"""

RESIDUAL_ADD_KERNEL = """
struct AddParams {
    size: u32,
}

@group(0) @binding(0) var<uniform> params: AddParams;
@group(0) @binding(1) var<storage, read> input_a: array<f32>;
@group(0) @binding(2) var<storage, read> input_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    output[idx] = input_a[idx] + input_b[idx];
}
"""

EMBEDDING_KERNEL = """
struct EmbedParams {
    batch_size: u32,
    seq_len: u32,
    embedding_dim: u32,
}

@group(0) @binding(0) var<uniform> params: EmbedParams;
@group(0) @binding(1) var<storage, read> embedding_table: array<f32>;
@group(0) @binding(2) var<storage, read> pos_encoding: array<f32>;
@group(0) @binding(3) var<storage, read> input_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.seq_len;

    if (idx >= total) {
        return;
    }

    let seq_idx = idx % params.seq_len;
    let token_id = input_ids[idx];
    let D = params.embedding_dim;

    let emb_offset = token_id * D;
    let pos_offset = seq_idx * D;
    let out_offset = idx * D;

    for (var d = 0u; d < D; d++) {
        output[out_offset + d] = embedding_table[emb_offset + d] + pos_encoding[pos_offset + d];
    }
}
"""

BIAS_ADD_KERNEL = """
struct BiasParams {
    size: u32,
    dim: u32,
}

@group(0) @binding(0) var<uniform> params: BiasParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let row = idx / params.dim;
    let col = idx % params.dim;

    output[idx] = input[idx] + bias[col];
}
"""


# ============================================================================
# BACKWARD PASS KERNELS
# ============================================================================

MATMUL_BACKWARD_A_KERNEL = """
// Backward pass for matmul: compute gradient w.r.t. A
// Given: dL/dC, B
// Compute: dL/dA = dL/dC @ B^T

struct MatmulParams {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<uniform> params: MatmulParams;
@group(0) @binding(1) var<storage, read> grad_C: array<f32>;  // [M, N]
@group(0) @binding(2) var<storage, read> B: array<f32>;       // [K, N]
@group(0) @binding(3) var<storage, read_write> grad_A: array<f32>;  // [M, K]

const TILE_SIZE: u32 = 16u;
var<workgroup> tile_grad: array<f32, 256>;
var<workgroup> tile_B: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;  // M dimension
    let col = global_id.x;  // K dimension
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum = 0.0;
    let num_tiles = (params.N + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load grad_C tile
        let g_row = row;
        let g_col = t * TILE_SIZE + local_col;
        if (g_row < params.M && g_col < params.N) {
            tile_grad[local_row * TILE_SIZE + local_col] = grad_C[g_row * params.N + g_col];
        } else {
            tile_grad[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load B^T tile (transpose on-the-fly)
        let b_row = t * TILE_SIZE + local_row;
        let b_col = col;
        if (b_row < params.N && b_col < params.K) {
            tile_B[local_row * TILE_SIZE + local_col] = B[b_col * params.N + b_row];
        } else {
            tile_B[local_row * TILE_SIZE + local_col] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tile_grad[local_row * TILE_SIZE + k] * tile_B[k * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.K) {
        grad_A[row * params.K + col] = sum;
    }
}
"""

MATMUL_BACKWARD_B_KERNEL = """
// Backward pass for matmul: compute gradient w.r.t. B
// Given: A, dL/dC
// Compute: dL/dB = A^T @ dL/dC

struct MatmulParams {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<uniform> params: MatmulParams;
@group(0) @binding(1) var<storage, read> A: array<f32>;       // [M, K]
@group(0) @binding(2) var<storage, read> grad_C: array<f32>;  // [M, N]
@group(0) @binding(3) var<storage, read_write> grad_B: array<f32>;  // [K, N]

const TILE_SIZE: u32 = 16u;
var<workgroup> tile_A: array<f32, 256>;
var<workgroup> tile_grad: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;  // K dimension
    let col = global_id.x;  // N dimension
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum = 0.0;
    let num_tiles = (params.M + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load A^T tile (transpose on-the-fly)
        let a_row = t * TILE_SIZE + local_row;
        let a_col = row;
        if (a_row < params.M && a_col < params.K) {
            tile_A[local_row * TILE_SIZE + local_col] = A[a_row * params.K + a_col];
        } else {
            tile_A[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load grad_C tile
        let g_row = t * TILE_SIZE + local_row;
        let g_col = col;
        if (g_row < params.M && g_col < params.N) {
            tile_grad[local_row * TILE_SIZE + local_col] = grad_C[g_row * params.N + g_col];
        } else {
            tile_grad[local_row * TILE_SIZE + local_col] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tile_A[k * TILE_SIZE + local_col] * tile_grad[k * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    if (row < params.K && col < params.N) {
        grad_B[row * params.N + col] = sum;
    }
}
"""

LAYERNORM_BACKWARD_KERNEL = """
// Backward pass for layer normalization (without atomics)
// Each workgroup handles one complete element

struct NormParams {
    size: u32,
    n_elements: u32,
}

@group(0) @binding(0) var<uniform> params: NormParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_gamma: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_beta: array<f32>;

const EPS: f32 = 1e-5;
const BLOCK_SIZE: u32 = 256u;
var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let elem_idx = workgroup_id.x;
    let tid = local_id.x;

    if (elem_idx >= params.n_elements) {
        return;
    }

    let offset = elem_idx * params.size;

    // Recompute mean and variance from forward pass
    var sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        sum += input[offset + i];
    }
    shared_data[tid] = sum;
    workgroupBarrier();

    // Reduction for mean
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }
    let mean = shared_data[0] / f32(params.size);

    // Compute variance
    var var_sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        let diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    shared_data[tid] = var_sum;
    workgroupBarrier();

    // Reduction for variance
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }
    let variance = shared_data[0] / f32(params.size);
    let inv_std = 1.0 / sqrt(variance + EPS);

    // Compute gradient contributions for gamma and beta (per element)
    var grad_gamma_sum = 0.0;
    var grad_beta_sum = 0.0;

    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        let x_norm = (input[offset + i] - mean) * inv_std;
        grad_gamma_sum += grad_output[offset + i] * x_norm;
        grad_beta_sum += grad_output[offset + i];
    }

    shared_data[tid] = grad_gamma_sum;
    workgroupBarrier();

    // Reduce gamma gradient
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }

    // Thread 0 writes gamma gradient for each dimension
    if (tid == 0u) {
        for (var i = 0u; i < params.size; i++) {
            let x_norm = (input[offset + i] - mean) * inv_std;

            // Accumulate to global (sequential, but only one thread)
            grad_gamma[i] += grad_output[offset + i] * x_norm;
            grad_beta[i] += grad_output[offset + i];
        }
    }
    workgroupBarrier();

    // Compute gradient w.r.t. input using chain rule
    var dxhat_sum = 0.0;
    var dxhat_xhat_sum = 0.0;

    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        let x_hat = (input[offset + i] - mean) * inv_std;
        let d_xhat = grad_output[offset + i] * gamma[i];
        dxhat_sum += d_xhat;
        dxhat_xhat_sum += d_xhat * x_hat;
    }

    shared_data[tid] = dxhat_sum;
    workgroupBarrier();
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }
    dxhat_sum = shared_data[0];

    shared_data[tid] = dxhat_xhat_sum;
    workgroupBarrier();
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }
    dxhat_xhat_sum = shared_data[0];

    // Write gradient w.r.t. input
    let N = f32(params.size);
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        let x_hat = (input[offset + i] - mean) * inv_std;
        let d_xhat = grad_output[offset + i] * gamma[i];
        grad_input[offset + i] = (d_xhat - (dxhat_sum / N) - (x_hat * dxhat_xhat_sum / N)) * inv_std;
    }
}
"""


GELU_BACKWARD_KERNEL = """
// Backward pass for GELU activation
// GELU(x) = x * Φ(x) where Φ is standard normal CDF
// GELU'(x) ≈ Φ(x) + x * φ(x) where φ is standard normal PDF

struct GeluParams {
    size: u32,
}

@group(0) @binding(0) var<uniform> params: GeluParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let x = input[idx];
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    let tanh_inner = tanh(inner);

    // Derivative of GELU
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEFF * x * x);
    let gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;

    grad_input[idx] = grad_output[idx] * gelu_grad;
}
"""

BIAS_BACKWARD_KERNEL = """
// Backward pass for bias addition
// Gradient w.r.t. bias is sum over batch dimension

struct BiasParams {
    size: u32,
    dim: u32,
}

@group(0) @binding(0) var<uniform> params: BiasParams;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_bias: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;

    if (col >= params.dim) {
        return;
    }

    let n_rows = params.size / params.dim;
    var sum = 0.0;

    for (var row = 0u; row < n_rows; row++) {
        sum += grad_output[row * params.dim + col];
    }

    grad_bias[col] = sum;
}
"""

CROSS_ENTROPY_LOSS_KERNEL = """
// Combined cross-entropy loss and gradient computation
// More efficient than separate loss + backward kernels

struct LossParams {
    batch_size: u32,
    seq_len: u32,
    vocab_size: u32,
}

@group(0) @binding(0) var<uniform> params: LossParams;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> loss_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_logits: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pred_idx = global_id.x;
    let total = params.batch_size * params.seq_len;

    if (pred_idx >= total) {
        return;
    }

    let target_idx = targets[pred_idx];
    let logit_offset = pred_idx * params.vocab_size;

    // Numerically stable softmax
    var max_logit = logits[logit_offset];
    for (var i = 1u; i < params.vocab_size; i++) {
        max_logit = max(max_logit, logits[logit_offset + i]);
    }

    var sum_exp = 0.0;
    for (var i = 0u; i < params.vocab_size; i++) {
        sum_exp += exp(logits[logit_offset + i] - max_logit);
    }

    // Loss: -log(softmax[target])
    let target_logit = logits[logit_offset + target_idx];
    loss_output[pred_idx] = -(target_logit - max_logit - log(sum_exp));

    // Gradient: softmax - one_hot
    for (var i = 0u; i < params.vocab_size; i++) {
        let prob = exp(logits[logit_offset + i] - max_logit) / sum_exp;
        var grad = prob;
        if (i == target_idx) {
            grad -= 1.0;
        }
        grad_logits[logit_offset + i] = grad / f32(total);
    }
}
"""

ADAMW_OPTIMIZER_KERNEL = """
// Fused AdamW optimizer update

struct OptimizerParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    eps: f32,
    step: f32,
}

@group(0) @binding(0) var<uniform> params: OptimizerParams;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= size) {
        return;
    }

    let grad = gradients[idx];
    let weight = weights[idx];

    // Update biased first moment estimate
    let m_new = params.beta1 * m[idx] + (1.0 - params.beta1) * grad;
    m[idx] = m_new;

    // Update biased second raw moment estimate
    let v_new = params.beta2 * v[idx] + (1.0 - params.beta2) * grad * grad;
    v[idx] = v_new;

    // Compute bias-corrected first moment estimate
    let m_hat = m_new / (1.0 - pow(params.beta1, params.step));

    // Compute bias-corrected second raw moment estimate
    let v_hat = v_new / (1.0 - pow(params.beta2, params.step));

    // Update weights with AdamW (decoupled weight decay)
    let update = m_hat / (sqrt(v_hat) + params.eps);
    weights[idx] = weight - params.lr * (update + params.weight_decay * weight);
}
"""


# ============================================================================
# PROPER ATTENTION KERNEL (Simplified Multi-Head)
# ============================================================================

MULTIHEAD_ATTENTION_KERNEL = """
// Simplified multi-head self-attention with causal masking
// Processes one query position across all heads

struct AttentionParams {
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<uniform> params: AttentionParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;  // [B*S, n_heads*head_dim]
@group(0) @binding(2) var<storage, read> K: array<f32>;  // [B*S, n_heads*head_dim]
@group(0) @binding(3) var<storage, read> V: array<f32>;  // [B*S, n_heads*head_dim]
@group(0) @binding(4) var<storage, read_write> output: array<f32>;  // [B*S, n_heads*head_dim]

var<workgroup> shared_scores: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let batch_idx = group_id.z;
    let head_idx = group_id.y;
    let q_pos = group_id.x;
    let tid = local_id.x;

    if (batch_idx >= params.batch_size || q_pos >= params.seq_len) {
        return;
    }

    let head_dim = params.head_dim;
    let seq_len = params.seq_len;
    let embedding_dim = params.n_heads * head_dim;
    let scale = 1.0 / sqrt(f32(head_dim));

    // Calculate offset for this query
    let q_offset = batch_idx * seq_len * embedding_dim +
                   q_pos * embedding_dim +
                   head_idx * head_dim;

    // Load query into registers (small enough for head_dim <= 64)
    var q_local: array<f32, 64>;
    for (var d = 0u; d < head_dim; d++) {
        q_local[d] = Q[q_offset + d];
    }

    // Compute attention scores for positions up to q_pos (causal mask)
    var max_score = -1e9;
    var scores: array<f32, 512>;  // Max seq_len

    // Each thread computes a subset of scores
    for (var k_pos = tid; k_pos <= q_pos; k_pos += 256u) {
        let k_offset = batch_idx * seq_len * embedding_dim +
                      k_pos * embedding_dim +
                      head_idx * head_dim;

        var score = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            score += q_local[d] * K[k_offset + d];
        }
        score *= scale;
        scores[k_pos] = score;
        max_score = max(max_score, score);
    }

    // Reduce max across threads
    shared_scores[tid] = max_score;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s && tid + s < 256u) {
            shared_scores[tid] = max(shared_scores[tid], shared_scores[tid + s]);
        }
        workgroupBarrier();
    }
    max_score = shared_scores[0];

    // Compute exp and sum for softmax
    var sum_exp = 0.0;
    for (var k_pos = tid; k_pos <= q_pos; k_pos += 256u) {
        let exp_score = exp(scores[k_pos] - max_score);
        scores[k_pos] = exp_score;
        sum_exp += exp_score;
    }

    // Reduce sum across threads
    shared_scores[tid] = sum_exp;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s && tid + s < 256u) {
            shared_scores[tid] += shared_scores[tid + s];
        }
        workgroupBarrier();
    }
    sum_exp = shared_scores[0];

    // Compute weighted sum of values
    var output_local: array<f32, 64>;
    for (var d = 0u; d < head_dim; d++) {
        output_local[d] = 0.0;
    }

    for (var k_pos = tid; k_pos <= q_pos; k_pos += 256u) {
        let attn_weight = scores[k_pos] / sum_exp;
        let v_offset = batch_idx * seq_len * embedding_dim +
                      k_pos * embedding_dim +
                      head_idx * head_dim;

        for (var d = 0u; d < head_dim; d++) {
            output_local[d] += attn_weight * V[v_offset + d];
        }
    }

    // Reduce across threads for each dimension
    for (var d = 0u; d < head_dim; d++) {
        shared_scores[tid] = output_local[d];
        workgroupBarrier();

        for (var s = 128u; s > 0u; s >>= 1u) {
            if (tid < s && tid + s < 256u) {
                shared_scores[tid] += shared_scores[tid + s];
            }
            workgroupBarrier();
        }

        if (tid == 0u) {
            let out_offset = batch_idx * seq_len * embedding_dim +
                           q_pos * embedding_dim +
                           head_idx * head_dim;
            output[out_offset + d] = shared_scores[0];
        }
        workgroupBarrier();
    }
}
"""

TRANSPOSE_KERNEL = """
// Matrix transpose: B = A^T

struct TransposeParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> params: TransposeParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const TILE_SIZE: u32 = 16u;
var<workgroup> tile: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    // Load into shared memory with coalescing
    if (row < params.rows && col < params.cols) {
        tile[local_row * TILE_SIZE + local_col] = input[row * params.cols + col];
    }

    workgroupBarrier();

    // Write transposed (swap row and col)
    let out_row = col;
    let out_col = row;

    if (out_row < params.cols && out_col < params.rows) {
        output[out_row * params.rows + out_col] = tile[local_col * TILE_SIZE + local_row];
    }
}
"""

EXTRACT_LAST_TOKENS_KERNEL = """
// Extract last token from each sequence in batch

struct ExtractParams {
    batch_size: u32,
    seq_len: u32,
    embedding_dim: u32,
}

@group(0) @binding(0) var<uniform> params: ExtractParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;  // [B*S, D]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;  // [B, D]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.y;
    let dim_idx = global_id.x;

    if (batch_idx >= params.batch_size || dim_idx >= params.embedding_dim) {
        return;
    }

    // Extract from last position in sequence
    let last_pos = params.seq_len - 1u;
    let input_offset = batch_idx * params.seq_len * params.embedding_dim +
                       last_pos * params.embedding_dim + dim_idx;
    let output_offset = batch_idx * params.embedding_dim + dim_idx;

    output[output_offset] = input[input_offset];
}
"""


# ============================================================================
# FLASHATTENTION-STYLE TILED ATTENTION KERNEL
# ============================================================================

FLASHATTENTION_FORWARD_KERNEL = """
// FlashAttention: Memory-efficient attention using tiling and online softmax
// Based on: Dao et al. 2022 - "FlashAttention: Fast and Memory-Efficient Exact Attention"
//
// Key innovations:
// 1. Tile Q, K, V to fit in shared memory (SRAM)
// 2. Online softmax: maintain running max and sum for numerical stability
// 3. Fused operations: compute attention without materializing full matrix
// 4. Minimize HBM accesses: O(N^2/M) instead of O(N^2)

struct FlashAttentionParams {
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    Bc: u32,  // Block size for K/V (columns)
    Br: u32,  // Block size for Q (rows)
}

@group(0) @binding(0) var<uniform> params: FlashAttentionParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@group(0) @binding(5) var<storage, read_write> L: array<f32>;  // Softmax denominator (for backward)
@group(0) @binding(6) var<storage, read_write> M: array<f32>;  // Max values (for backward)

// Shared memory tiles
// Bc = 32, Br = 32, head_dim = 64 max
const Bc: u32 = 32u;
const Br: u32 = 32u;
const HEAD_DIM: u32 = 64u;

var<workgroup> Qi: array<f32, 2048>;  // Br x head_dim (32 x 64)
var<workgroup> Kj: array<f32, 2048>;  // Bc x head_dim (32 x 64)
var<workgroup> Vj: array<f32, 2048>;  // Bc x head_dim (32 x 64)
var<workgroup> Sij: array<f32, 1024>; // Br x Bc (32 x 32) - attention scores
var<workgroup> Pij: array<f32, 1024>; // Br x Bc (32 x 32) - attention weights

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.z;
    let head_idx = workgroup_id.y;
    let block_row = workgroup_id.x;  // Which block of Q we're processing
    let tid = local_id.x;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let d = params.head_dim;
    let N = params.seq_len;
    let embedding_dim = params.n_heads * d;
    let scale = 1.0 / sqrt(f32(d));

    // Calculate block bounds for Q
    let q_start = block_row * Br;
    let q_end = min(q_start + Br, N);
    let actual_Br = q_end - q_start;

    if (actual_Br == 0u) {
        return;
    }

    // Load Q block into shared memory (cooperatively)
    // Each thread loads multiple elements
    for (var i = tid; i < actual_Br * d; i += 32u) {
        let local_row = i / d;
        let local_col = i % d;
        let global_row = q_start + local_row;

        let q_offset = batch_idx * N * embedding_dim +
                      global_row * embedding_dim +
                      head_idx * d + local_col;

        Qi[local_row * HEAD_DIM + local_col] = Q[q_offset];
    }
    workgroupBarrier();

    // Initialize output accumulators and online softmax statistics
    var Oi: array<f32, 64>;  // Per-thread output accumulator
    var mi: array<f32, 32>;  // Per-row running max
    var li: array<f32, 32>;  // Per-row running sum

    // Thread 0 initializes for all rows in this block
    if (tid == 0u) {
        for (var i = 0u; i < actual_Br; i++) {
            mi[i] = -1e9;
            li[i] = 0.0;
            for (var d_idx = 0u; d_idx < d; d_idx++) {
                Oi[i * HEAD_DIM + d_idx] = 0.0;
            }
        }
    }
    workgroupBarrier();

    // Number of K/V blocks to process
    let num_kv_blocks = (N + Bc - 1u) / Bc;

    // Iterate over K/V blocks
    for (var block_col = 0u; block_col < num_kv_blocks; block_col++) {
        let kv_start = block_col * Bc;
        let kv_end = min(kv_start + Bc, N);
        let actual_Bc = kv_end - kv_start;

        // Causal masking: only process blocks where KV positions <= Q positions
        if (kv_start > q_end) {
            break;
        }

        // Load K and V blocks into shared memory
        for (var i = tid; i < actual_Bc * d; i += 32u) {
            let local_row = i / d;
            let local_col = i % d;
            let global_row = kv_start + local_row;

            let kv_offset = batch_idx * N * embedding_dim +
                          global_row * embedding_dim +
                          head_idx * d + local_col;

            Kj[local_row * HEAD_DIM + local_col] = K[kv_offset];
            Vj[local_row * HEAD_DIM + local_col] = V[kv_offset];
        }
        workgroupBarrier();

        // Compute Sij = Qi @ Kj^T (attention scores for this block)
        for (var i = tid; i < actual_Br * actual_Bc; i += 32u) {
            let row = i / actual_Bc;
            let col = i % actual_Bc;

            var score = 0.0;
            for (var k = 0u; k < d; k++) {
                score += Qi[row * HEAD_DIM + k] * Kj[col * HEAD_DIM + k];
            }

            // Apply scaling and causal mask
            let q_pos = q_start + row;
            let kv_pos = kv_start + col;
            if (kv_pos <= q_pos) {
                Sij[row * Bc + col] = score * scale;
            } else {
                Sij[row * Bc + col] = -1e9;  // Masked position
            }
        }
        workgroupBarrier();

        // Online softmax update (per row)
        if (tid == 0u) {
            for (var row = 0u; row < actual_Br; row++) {
                // Find new max for this row
                var mij_new = mi[row];
                for (var col = 0u; col < actual_Bc; col++) {
                    mij_new = max(mij_new, Sij[row * Bc + col]);
                }

                // Compute exp(scores - max) and new sum
                var lij_new = 0.0;
                for (var col = 0u; col < actual_Bc; col++) {
                    let p = exp(Sij[row * Bc + col] - mij_new);
                    Pij[row * Bc + col] = p;
                    lij_new += p;
                }

                // Update running statistics
                let mi_old = mi[row];
                let li_old = li[row];

                mi[row] = mij_new;
                li[row] = li_old * exp(mi_old - mij_new) + lij_new;
            }
        }
        workgroupBarrier();

        // Update output: Oi = diag(exp(mi_old - mi_new)) @ Oi + Pij @ Vj
        for (var row = tid; row < actual_Br; row += 32u) {
            let correction = exp(mi[row] - mi[row]);  // Will be 1.0 for first block

            // Scale previous output
            for (var d_idx = 0u; d_idx < d; d_idx++) {
                Oi[row * HEAD_DIM + d_idx] *= correction;
            }

            // Add Pij @ Vj contribution
            for (var d_idx = 0u; d_idx < d; d_idx++) {
                var sum = 0.0;
                for (var col = 0u; col < actual_Bc; col++) {
                    sum += Pij[row * Bc + col] * Vj[col * HEAD_DIM + d_idx];
                }
                Oi[row * HEAD_DIM + d_idx] += sum;
            }
        }
        workgroupBarrier();
    }

    // Final normalization and write output
    for (var row = tid; row < actual_Br; row += 32u) {
        let global_row = q_start + row;

        for (var d_idx = 0u; d_idx < d; d_idx++) {
            let o_offset = batch_idx * N * embedding_dim +
                          global_row * embedding_dim +
                          head_idx * d + d_idx;

            O[o_offset] = Oi[row * HEAD_DIM + d_idx] / li[row];
        }

        // Store softmax statistics for backward pass
        let stats_offset = batch_idx * N * params.n_heads +
                          global_row * params.n_heads + head_idx;
        L[stats_offset] = li[row];
        M[stats_offset] = mi[row];
    }
}
"""

FLASHATTENTION_BACKWARD_KERNEL = """
// FlashAttention backward pass
// Recomputes attention on-the-fly using saved statistics

struct FlashAttentionParams {
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<uniform> params: FlashAttentionParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read> O: array<f32>;
@group(0) @binding(5) var<storage, read> dO: array<f32>;  // Gradient of output
@group(0) @binding(6) var<storage, read> L: array<f32>;   // Saved softmax denominators
@group(0) @binding(7) var<storage, read> M: array<f32>;   // Saved max values
@group(0) @binding(8) var<storage, read_write> dQ: array<f32>;
@group(0) @binding(9) var<storage, read_write> dK: array<f32>;
@group(0) @binding(10) var<storage, read_write> dV: array<f32>;

const Bc: u32 = 32u;
const Br: u32 = 32u;
const HEAD_DIM: u32 = 64u;

var<workgroup> Qi: array<f32, 2048>;
var<workgroup> Kj: array<f32, 2048>;
var<workgroup> Vj: array<f32, 2048>;
var<workgroup> dOi: array<f32, 2048>;
var<workgroup> Pij: array<f32, 1024>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.z;
    let head_idx = workgroup_id.y;
    let block_row = workgroup_id.x;
    let tid = local_id.x;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let d = params.head_dim;
    let N = params.seq_len;
    let embedding_dim = params.n_heads * d;
    let scale = 1.0 / sqrt(f32(d));

    let q_start = block_row * Br;
    let q_end = min(q_start + Br, N);
    let actual_Br = q_end - q_start;

    if (actual_Br == 0u) {
        return;
    }

    // Load Q and dO blocks
    for (var i = tid; i < actual_Br * d; i += 32u) {
        let local_row = i / d;
        let local_col = i % d;
        let global_row = q_start + local_row;

        let offset = batch_idx * N * embedding_dim +
                    global_row * embedding_dim +
                    head_idx * d + local_col;

        Qi[local_row * HEAD_DIM + local_col] = Q[offset];
        dOi[local_row * HEAD_DIM + local_col] = dO[offset];
    }
    workgroupBarrier();

    // Initialize gradient accumulators
    var dQi: array<f32, 2048>;
    for (var i = 0u; i < actual_Br * d; i++) {
        dQi[i] = 0.0;
    }

    let num_kv_blocks = (N + Bc - 1u) / Bc;

    // Iterate over K/V blocks (same as forward)
    for (var block_col = 0u; block_col < num_kv_blocks; block_col++) {
        let kv_start = block_col * Bc;
        let kv_end = min(kv_start + Bc, N);
        let actual_Bc = kv_end - kv_start;

        if (kv_start > q_end) {
            break;
        }

        // Load K and V
        for (var i = tid; i < actual_Bc * d; i += 32u) {
            let local_row = i / d;
            let local_col = i % d;
            let global_row = kv_start + local_row;

            let offset = batch_idx * N * embedding_dim +
                       global_row * embedding_dim +
                       head_idx * d + local_col;

            Kj[local_row * HEAD_DIM + local_col] = K[offset];
            Vj[local_row * HEAD_DIM + local_col] = V[offset];
        }
        workgroupBarrier();

        // Recompute attention weights Pij
        for (var i = tid; i < actual_Br * actual_Bc; i += 32u) {
            let row = i / actual_Bc;
            let col = i % actual_Bc;

            var score = 0.0;
            for (var k = 0u; k < d; k++) {
                score += Qi[row * HEAD_DIM + k] * Kj[col * HEAD_DIM + k];
            }
            score *= scale;

            let q_pos = q_start + row;
            let kv_pos = kv_start + col;

            if (kv_pos <= q_pos) {
                let stats_offset = batch_idx * N * params.n_heads +
                                  q_pos * params.n_heads + head_idx;
                let m = M[stats_offset];
                let l = L[stats_offset];
                Pij[row * Bc + col] = exp(score - m) / l;
            } else {
                Pij[row * Bc + col] = 0.0;
            }
        }
        workgroupBarrier();

        // Compute gradients (simplified version)
        // In full implementation: dQ += (dP @ K), dK += (dP^T @ Q), dV += (P^T @ dO)
        // where dP is gradient w.r.t. attention weights

        // This is a placeholder - full backward pass requires more complex logic
        // For now, we'll accumulate basic gradients

        workgroupBarrier();
    }

    // Write dQ output
    for (var i = tid; i < actual_Br * d; i += 32u) {
        let local_row = i / d;
        let local_col = i % d;
        let global_row = q_start + local_row;

        let offset = batch_idx * N * embedding_dim +
                    global_row * embedding_dim +
                    head_idx * d + local_col;

        atomicAdd(&dQ[offset], dQi[i]);
    }
}
"""
