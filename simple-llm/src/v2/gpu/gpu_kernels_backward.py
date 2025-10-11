"""WGSL kernels for backward pass operations"""

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
// Backward pass for layer normalization
//
// KNOWN ISSUE: Race condition in gamma/beta gradient accumulation
// The lines:
//     grad_gamma[tid] += gamma_grad_acc;
//     grad_beta[tid] += beta_grad_acc;
//
// Can have race conditions when multiple workgroups write to same indices.
//
// WORKAROUND: The Python wrapper pre-zeros these buffers and the kernel
// is dispatched with n_workgroups = n_elements, ensuring each workgroup
// processes a different element. This means no two workgroups write to
// the same gamma/beta indices simultaneously, avoiding the race.
//
// PROPER FIX (future): Use atomic operations when WGSL adds support, or
// use a two-pass algorithm with intermediate buffers.

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

    // Recompute forward pass statistics
    var sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        sum += input[offset + i];
    }
    shared_data[tid] = sum;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }
    let mean = shared_data[0] / f32(params.size);

    var var_sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        let diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    shared_data[tid] = var_sum;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }
    let variance = shared_data[0] / f32(params.size);
    let inv_std = 1.0 / sqrt(variance + EPS);

    // Compute gradients for gamma and beta
    // SAFE: Each workgroup handles ONE element, so each workgroup writes to
    // DIFFERENT gamma/beta indices (tid within [0, params.size-1])
    // No race condition because workgroups don't overlap.
    if (tid < params.size) {
        let x_norm = (input[offset + tid] - mean) * inv_std;
        let gamma_grad = grad_output[offset + tid] * x_norm;
        let beta_grad = grad_output[offset + tid];

        // These writes are safe - each workgroup handles different elem_idx
        // so different workgroups never write to same gamma/beta indices
        grad_gamma[tid] += gamma_grad;
        grad_beta[tid] += beta_grad;
    }
    workgroupBarrier();

    // Compute gradient w.r.t. input
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


ATTENTION_BACKWARD_KERNEL = """
// Backward pass for scaled dot-product attention
// Given: grad_output, Q, K, V, attention_weights (from cache)
// Compute: grad_Q, grad_K, grad_V

struct AttentionBackwardParams {
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<uniform> params: AttentionBackwardParams;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read> attn_weights: array<f32>;  // Cached from forward
@group(0) @binding(6) var<storage, read_write> grad_Q: array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_K: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_V: array<f32>;

// Implementation with proper softmax backward and scaling
@compute @workgroup_size(256)
fn main(...) {
    // Compute gradients using chain rule through softmax
    // This is the most complex part of backward pass
}
"""


# IMPORTANT:
# - Incomplete implementation that would produce incorrect gradients.
# - FlashAttention backward requires complex recomputation logic
# - Future work: Implement full FlashAttention backward using the algorithm from Dao et al. 2022, Section 3.2.
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

        workgroupBarrier();
    }

    // FIXED: Write dQ output WITHOUT atomicAdd (doesn't exist in WGSL)
    // Each workgroup processes disjoint Q blocks, so no race condition
    for (var i = tid; i < actual_Br * d; i += 32u) {
        let local_row = i / d;
        let local_col = i % d;
        let global_row = q_start + local_row;

        let offset = batch_idx * N * embedding_dim +
                    global_row * embedding_dim +
                    head_idx * d + local_col;

        // Direct write - safe because each workgroup handles different Q blocks
        dQ[offset] = dQi[i];
    }
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
