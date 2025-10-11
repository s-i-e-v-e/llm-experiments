"""WGSL kernels for backward pass operations"""

# ============================================================================
# BACKWARD PASS KERNELS
# ============================================================================

# ============================================================================
# MATMUL BACKWARD KERNELS
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
@group(0) @binding(1) var<storage, read> grad_C: array<f32>;  // (M, N)
@group(0) @binding(2) var<storage, read> B: array<f32>;       // (K, N)
@group(0) @binding(3) var<storage, read_write> grad_A: array<f32>;  // (M, K)

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
@group(0) @binding(1) var<storage, read> A: array<f32>;       // (M, K)
@group(0) @binding(2) var<storage, read> grad_C: array<f32>;  // (M, N)
@group(0) @binding(3) var<storage, read_write> grad_B: array<f32>;  // (K, N)

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

# ============================================================================
# LAYER NORM BACKWARD KERNELS
# ============================================================================

LAYERNORM_BACKWARD_KERNEL = """
// Backward pass for layer normalization - STAGE 1
// Compute per-element contributions
//
// FIXED: Two-stage algorithm eliminates race condition:
// - Stage 1 (this kernel): Each workgroup computes partial gradients for gamma/beta
// - Stage 2 (separate kernel): Reduction combines partial gradients
//
// This kernel outputs PARTIAL gradients that must be reduced in stage 2.

struct NormParams {
    size: u32,
    n_elements: u32,
}

@group(0) @binding(0) var<uniform> params: NormParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(5) var<storage, read_write> partial_grad_gamma: array<f32>;  // (n_elements, size)
@group(0) @binding(6) var<storage, read_write> partial_grad_beta: array<f32>;   // (n_elements, size)

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

    // Compute PARTIAL gradients for gamma and beta (no accumulation, just write)
    // Each workgroup handles ONE element, writes to its own slice of partial buffers
    // SAFE: No race condition because each workgroup writes to different memory locations
    if (tid < params.size) {
        let x_norm = (input[offset + tid] - mean) * inv_std;
        let gamma_grad = grad_output[offset + tid] * x_norm;
        let beta_grad = grad_output[offset + tid];

        // Write partial gradients (will be reduced in stage 2)
        let partial_offset = elem_idx * params.size + tid;
        partial_grad_gamma[partial_offset] = gamma_grad;
        partial_grad_beta[partial_offset] = beta_grad;
    }
    workgroupBarrier();

    // Compute gradient w.r.t. input (unchanged from original)
    var d_xhat_sum = 0.0;
    var d_xhat_xhat_sum = 0.0;

    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        let xhat = (input[offset + i] - mean) * inv_std;
        let d_xhat = grad_output[offset + i] * gamma[i];
        d_xhat_sum += d_xhat;
        d_xhat_xhat_sum += d_xhat * xhat;
    }

    shared_data[tid] = d_xhat_sum;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }
    d_xhat_sum = shared_data[0];

    shared_data[tid] = d_xhat_xhat_sum;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }
    d_xhat_xhat_sum = shared_data[0];

    let N = f32(params.size);
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        let xhat = (input[offset + i] - mean) * inv_std;
        let d_xhat = grad_output[offset + i] * gamma[i];
        grad_input[offset + i] = (d_xhat - d_xhat_sum / N - xhat * d_xhat_xhat_sum / N) * inv_std;
    }
}
"""

LAYERNORM_BACKWARD_REDUCE_KERNEL = """
// Backward pass for layer normalization - STAGE 2
// Reduce partial gradients
//
// Reduces partial gamma/beta gradients from stage 1 across all elements.
// Each thread handles one feature dimension, summing across all batch elements.

struct ReduceParams {
    size: u32,
    n_elements: u32,
}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> partial_grad_gamma: array<f32>;  // (n_elements, size)
@group(0) @binding(2) var<storage, read> partial_grad_beta: array<f32>;   // (n_elements, size)
@group(0) @binding(3) var<storage, read_write> grad_gamma: array<f32>;    // (size,)
@group(0) @binding(4) var<storage, read_write> grad_beta: array<f32>;     // (size,)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim_idx = global_id.x;

    if (dim_idx >= params.size) {
        return;
    }

    // Sum partial gradients across all elements for this dimension
    var gamma_sum = 0.0;
    var beta_sum = 0.0;

    for (var elem_idx = 0u; elem_idx < params.n_elements; elem_idx++) {
        let partial_offset = elem_idx * params.size + dim_idx;
        gamma_sum += partial_grad_gamma[partial_offset];
        beta_sum += partial_grad_beta[partial_offset];
    }

    // Write final gradients
    grad_gamma[dim_idx] = gamma_sum;
    grad_beta[dim_idx] = beta_sum;
}
"""

LAYERNORM_BACKWARD_REDUCE_ACCUMULATE_KERNEL = """
// Backward pass for layer normalization - STAGE 2 (ACCUMULATE MODE)
// Reduce partial gradients and accumulate into existing values
//
// This variant atomically adds to existing grad_gamma and grad_beta values
// instead of overwriting them. Used for gradient accumulation across mini-batches.

struct ReduceParams {
    size: u32,
    n_elements: u32,
}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> partial_grad_gamma: array<f32>;  // (n_elements, size)
@group(0) @binding(2) var<storage, read> partial_grad_beta: array<f32>;   // (n_elements, size)
@group(0) @binding(3) var<storage, read_write> grad_gamma: array<f32>;    // (size,)
@group(0) @binding(4) var<storage, read_write> grad_beta: array<f32>;     // (size,)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim_idx = global_id.x;

    if (dim_idx >= params.size) {
        return;
    }

    // Sum partial gradients across all elements for this dimension
    var gamma_sum = 0.0;
    var beta_sum = 0.0;

    for (var elem_idx = 0u; elem_idx < params.n_elements; elem_idx++) {
        let partial_offset = elem_idx * params.size + dim_idx;
        gamma_sum += partial_grad_gamma[partial_offset];
        beta_sum += partial_grad_beta[partial_offset];
    }

    // Atomically add to existing gradients (instead of overwriting)
    atomicAdd(&grad_gamma[dim_idx], gamma_sum);
    atomicAdd(&grad_beta[dim_idx], beta_sum);
}
"""

# ============================================================================
# GELU BACKWARD KERNEL
# ============================================================================

GELU_BACKWARD_KERNEL = """
// Backward pass for GELU activation
//
// GELU(x) = x * Φ(x), where Φ is standard normal CDF
// GELU'(x) = Φ(x) + x * φ(x), where φ is standard normal PDF

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

# ============================================================================
# BIAS BACKWARD KERNEL
# ============================================================================

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

# ============================================================================
# ATTENTION BACKWARD KERNEL - FULL IMPLEMENTATION
# ============================================================================

ATTENTION_BACKWARD_KERNEL = """
// Backward pass for scaled dot-product attention
//
// Forward: O = softmax(Q @ K^T / sqrt(d)) @ V
// Backward: Compute dQ, dK, dV from dO
//
// Algorithm:
//   1. dV = P^T @ dO, where P = attention weights
//   2. dP = dO @ V^T
//   3. dS = softmax_backward(dP, P), where S = Q @ K^T / sqrt(d)
//   4. dQ = dS @ K / sqrt(d)
//   5. dK = dS^T @ Q / sqrt(d)

struct AttentionBackwardParams {
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<uniform> params: AttentionBackwardParams;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;  // dO
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read> O: array<f32>;  // Forward output (for softmax backward)
@group(0) @binding(6) var<storage, read_write> grad_Q: array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_K: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_V: array<f32>;

const BLOCK_SIZE: u32 = 256u;

var<workgroup> shared_mem: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.z;
    let head_idx = workgroup_id.y;
    let q_pos = workgroup_id.x;
    let tid = local_id.x;

    if (batch_idx >= params.batch_size || q_pos >= params.seq_len) {
        return;
    }

    let head_dim = params.head_dim;
    let seq_len = params.seq_len;
    let embedding_dim = params.n_heads * head_dim;
    let scale = 1.0 / sqrt(f32(head_dim));

    // Offsets for this query position
    let q_offset = batch_idx * seq_len * embedding_dim + q_pos * embedding_dim + head_idx * head_dim;
    let o_offset = q_offset;
    let do_offset = q_offset;

    // ========================================================================
    // STEP 1: Compute attention weights P = softmax(Q @ K^T / sqrt(d))
    // We need to recompute these from forward pass
    // ========================================================================

    // Compute scores and find max (for numerical stability)
    var max_score = -1e9;
    for (var k_pos = tid; k_pos <= q_pos; k_pos += BLOCK_SIZE) {  // Causal mask
        let k_offset = batch_idx * seq_len * embedding_dim + k_pos * embedding_dim + head_idx * head_dim;
        var score = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }
        score *= scale;
        max_score = max(max_score, score);
    }

    // Reduce max across threads
    shared_mem[tid] = max_score;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + s]);
        }
        workgroupBarrier();
    }
    max_score = shared_mem[0];
    workgroupBarrier();

    // Compute exp(scores - max) and sum
    var sum_exp = 0.0;
    for (var k_pos = tid; k_pos <= q_pos; k_pos += BLOCK_SIZE) {
        let k_offset = batch_idx * seq_len * embedding_dim + k_pos * embedding_dim + head_idx * head_dim;
        var score = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }
        score = score * scale - max_score;
        sum_exp += exp(score);
    }

    // Reduce sum
    shared_mem[tid] = sum_exp;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        workgroupBarrier();
    }
    sum_exp = shared_mem[0];
    workgroupBarrier();

    // ========================================================================
    // STEP 2: Compute dV = P^T @ dO
    // Each thread processes subset of V dimensions
    // ========================================================================

    for (var k_pos = 0u; k_pos <= q_pos; k_pos++) {
        // Compute attention weight P[q_pos, k_pos]
        let k_offset = batch_idx * seq_len * embedding_dim + k_pos * embedding_dim + head_idx * head_dim;
        var score = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }
        let attn_weight = exp(score * scale - max_score) / sum_exp;

        // Accumulate dV[k_pos] += attn_weight * dO[q_pos]
        let dv_offset = batch_idx * seq_len * embedding_dim + k_pos * embedding_dim + head_idx * head_dim;
        for (var d = tid; d < head_dim; d += BLOCK_SIZE) {
            atomicAdd(&grad_V[dv_offset + d], attn_weight * grad_output[do_offset + d]);
        }
    }

    workgroupBarrier();

    // ========================================================================
    // STEP 3: Compute dP (gradient w.r.t. attention weights)
    // dP[q_pos, k_pos] = dO[q_pos] dot V[k_pos]
    // ========================================================================

    // ========================================================================
    // STEP 4: Backward through softmax
    // dS = P * (dP - sum_j(P_j * dP_j))
    // where sum_j(P_j * dP_j) is computed as dot(O, dO)
    // ========================================================================

    // Compute dot(O[q_pos], dO[q_pos]) - this is sum_j(P_j * dP_j) due to chain rule
    var o_dot_do = 0.0;
    for (var d = tid; d < head_dim; d += BLOCK_SIZE) {
        o_dot_do += O[o_offset + d] * grad_output[do_offset + d];
    }

    shared_mem[tid] = o_dot_do;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        workgroupBarrier();
    }
    o_dot_do = shared_mem[0];
    workgroupBarrier();

    // ========================================================================
    // STEP 5: Compute dQ and dK
    // dQ[q_pos] += sum_k dS[q_pos, k] * K[k] * scale
    // dK[k_pos] += sum_q dS[q, k_pos] * Q[q] * scale
    // ========================================================================

    // Compute dQ for this query position
    for (var d = tid; d < head_dim; d += BLOCK_SIZE) {
        var dq_sum = 0.0;

        for (var k_pos = 0u; k_pos <= q_pos; k_pos++) {
            let k_offset = batch_idx * seq_len * embedding_dim + k_pos * embedding_dim + head_idx * head_dim;
            let v_offset = k_offset;

            // Compute P[q_pos, k_pos]
            var score = 0.0;
            for (var dd = 0u; dd < head_dim; dd++) {
                score += Q[q_offset + dd] * K[k_offset + dd];
            }
            let P = exp(score * scale - max_score) / sum_exp;

            // Compute dP[q_pos, k_pos] = dot(dO[q_pos], V[k_pos])
            var dP = 0.0;
            for (var dd = 0u; dd < head_dim; dd++) {
                dP += grad_output[do_offset + dd] * V[v_offset + dd];
            }

            // Softmax backward: dS = P * (dP - o_dot_do)
            let dS = P * (dP - o_dot_do);

            // Accumulate dQ
            dq_sum += dS * K[k_offset + d];
        }

        atomicAdd(&grad_Q[q_offset + d], dq_sum * scale);
    }

    // Compute dK for all key positions that interact with this query
    for (var k_pos = 0u; k_pos <= q_pos; k_pos++) {
        let k_offset = batch_idx * seq_len * embedding_dim + k_pos * embedding_dim + head_idx * head_dim;
        let v_offset = k_offset;

        // Compute P[q_pos, k_pos]
        var score = 0.0;
        for (var dd = 0u; dd < head_dim; dd++) {
            score += Q[q_offset + dd] * K[k_offset + dd];
        }
        let P = exp(score * scale - max_score) / sum_exp;

        // Compute dP
        var dP = 0.0;
        for (var dd = 0u; dd < head_dim; dd++) {
            dP += grad_output[do_offset + dd] * V[v_offset + dd];
        }

        // Softmax backward
        let dS = P * (dP - o_dot_do);

        // Accumulate dK
        for (var d = tid; d < head_dim; d += BLOCK_SIZE) {
            atomicAdd(&grad_K[k_offset + d], dS * Q[q_offset + d] * scale);
        }
    }
}
"""

# ============================================================================
# FLASH ATTENTION BACKWARD KERNEL - FULL IMPLEMENTATION
# ============================================================================

FLASH_ATTENTION_BACKWARD_KERNEL = """
// FlashAttention backward pass
// Implements the algorithm from Dao et al. 2022, Section 3.2
//
// Key insight: Recompute attention weights on-the-fly using saved L and M statistics
// This avoids materializing the full attention matrix in memory

struct FlashAttentionParams {
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    Bc: u32,  // Block size for KV (columns)
    Br: u32,  // Block size for Q (rows)
}

@group(0) @binding(0) var<uniform> params: FlashAttentionParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read> O: array<f32>;   // Forward output
@group(0) @binding(5) var<storage, read> dO: array<f32>;  // Gradient of output
@group(0) @binding(6) var<storage, read> L: array<f32>;   // Saved softmax denominators
@group(0) @binding(7) var<storage, read> M: array<f32>;   // Saved max values
@group(0) @binding(8) var<storage, read_write> dQ: array<f32>;
@group(0) @binding(9) var<storage, read_write> dK: array<f32>;
@group(0) @binding(10) var<storage, read_write> dV: array<f32>;

const Bc: u32 = 32u;
const Br: u32 = 32u;
const HEAD_DIM: u32 = 64u;

// Shared memory tiles
var<workgroup> Qi: array<f32, 2048>;       // Br x head_dim (32 x 64)
var<workgroup> Kj: array<f32, 2048>;       // Bc x head_dim (32 x 64)
var<workgroup> Vj: array<f32, 2048>;       // Bc x head_dim (32 x 64)
var<workgroup> dOi: array<f32, 2048>;      // Br x head_dim
var<workgroup> Oi: array<f32, 2048>;       // Br x head_dim
var<workgroup> Pij: array<f32, 1024>;      // Br x Bc (32 x 32)
var<workgroup> dPij: array<f32, 1024>;     // Br x Bc
var<workgroup> dSij: array<f32, 1024>;     // Br x Bc
var<workgroup> Di: array<f32, 32>;         // Br - stores D = rowsum(dO * O)

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

    // ========================================================================
    // Load Q, dO, and O blocks into shared memory
    // ========================================================================

    for (var i = tid; i < actual_Br * d; i += 32u) {
        let local_row = i / d;
        let local_col = i % d;
        let global_row = q_start + local_row;
        let offset = batch_idx * N * embedding_dim + global_row * embedding_dim + head_idx * d + local_col;

        Qi[local_row * HEAD_DIM + local_col] = Q[offset];
        dOi[local_row * HEAD_DIM + local_col] = dO[offset];
        Oi[local_row * HEAD_DIM + local_col] = O[offset];
    }

    workgroupBarrier();

    // ========================================================================
    // Compute D = rowsum(dO * O) for each row in this block
    // This is needed for softmax backward
    // ========================================================================

    if (tid == 0u) {
        for (var row = 0u; row < actual_Br; row++) {
            var sum = 0.0;
            for (var col = 0u; col < d; col++) {
                sum += dOi[row * HEAD_DIM + col] * Oi[row * HEAD_DIM + col];
            }
            Di[row] = sum;
        }
    }

    workgroupBarrier();

    // ========================================================================
    // Iterate over KV blocks (same as forward pass)
    // ========================================================================

    let num_kv_blocks = (N + Bc - 1u) / Bc;

    for (var block_col = 0u; block_col < num_kv_blocks; block_col++) {
        let kv_start = block_col * Bc;
        let kv_end = min(kv_start + Bc, N);
        let actual_Bc = kv_end - kv_start;

        // Causal masking: only process blocks where KV positions <= Q positions
        if (kv_start >= q_end) {
            break;
        }

        // Load K and V blocks
        for (var i = tid; i < actual_Bc * d; i += 32u) {
            let local_row = i / d;
            let local_col = i % d;
            let global_row = kv_start + local_row;
            let offset = batch_idx * N * embedding_dim + global_row * embedding_dim + head_idx * d + local_col;

            Kj[local_row * HEAD_DIM + local_col] = K[offset];
            Vj[local_row * HEAD_DIM + local_col] = V[offset];
        }

        workgroupBarrier();

        // ====================================================================
        // Recompute attention weights Pij = softmax(Qi @ Kj^T / sqrt(d))
        // using saved L and M statistics
        // ====================================================================

        for (var i = tid; i < actual_Br * actual_Bc; i += 32u) {
            let row = i / actual_Bc;
            let col = i % actual_Bc;
            let q_pos = q_start + row;
            let kv_pos = kv_start + col;

            // Causal mask
            if (kv_pos > q_pos) {
                Pij[row * Bc + col] = 0.0;
                continue;
            }

            // Compute score = Qi[row] @ Kj[col]
            var score = 0.0;
            for (var k = 0u; k < d; k++) {
                score += Qi[row * HEAD_DIM + k] * Kj[col * HEAD_DIM + k];
            }

            // Retrieve saved statistics
            let stats_offset = batch_idx * N * params.n_heads + q_pos * params.n_heads + head_idx;
            let m = M[stats_offset];
            let l = L[stats_offset];

            // Recompute attention weight
            Pij[row * Bc + col] = exp(score * scale - m) / l;
        }

        workgroupBarrier();

        // ====================================================================
        // Compute dPij = dOi @ Vj^T
        // ====================================================================

        for (var i = tid; i < actual_Br * actual_Bc; i += 32u) {
            let row = i / actual_Bc;
            let col = i % actual_Bc;

            var sum = 0.0;
            for (var k = 0u; k < d; k++) {
                sum += dOi[row * HEAD_DIM + k] * Vj[col * HEAD_DIM + k];
            }

            dPij[row * Bc + col] = sum;
        }

        workgroupBarrier();

        // ====================================================================
        // Softmax backward: dSij = Pij * (dPij - Di)
        // ====================================================================

        for (var i = tid; i < actual_Br * actual_Bc; i += 32u) {
            let row = i / actual_Bc;
            let col = i % actual_Bc;

            dSij[row * Bc + col] = Pij[row * Bc + col] * (dPij[row * Bc + col] - Di[row]);
        }

        workgroupBarrier();

        // ====================================================================
        // Accumulate gradients
        // dQi += dSij @ Kj * scale
        // dKj += dSij^T @ Qi * scale
        // dVj += Pij^T @ dOi
        // ====================================================================

        // Accumulate dQ (local accumulation, then write out)
        for (var row = tid; row < actual_Br; row += 32u) {
            let global_row = q_start + row;
            let dq_offset = batch_idx * N * embedding_dim + global_row * embedding_dim + head_idx * d;

            for (var d_idx = 0u; d_idx < d; d_idx++) {
                var sum = 0.0;
                for (var col = 0u; col < actual_Bc; col++) {
                    sum += dSij[row * Bc + col] * Kj[col * HEAD_DIM + d_idx];
                }
                atomicAdd(&dQ[dq_offset + d_idx], sum * scale);
            }
        }

        // Accumulate dK
        for (var col = tid; col < actual_Bc; col += 32u) {
            let global_col = kv_start + col;
            let dk_offset = batch_idx * N * embedding_dim + global_col * embedding_dim + head_idx * d;

            for (var d_idx = 0u; d_idx < d; d_idx++) {
                var sum = 0.0;
                for (var row = 0u; row < actual_Br; row++) {
                    sum += dSij[row * Bc + col] * Qi[row * HEAD_DIM + d_idx];
                }
                atomicAdd(&dK[dk_offset + d_idx], sum * scale);
            }
        }

        // Accumulate dV
        for (var col = tid; col < actual_Bc; col += 32u) {
            let global_col = kv_start + col;
            let dv_offset = batch_idx * N * embedding_dim + global_col * embedding_dim + head_idx * d;

            for (var d_idx = 0u; d_idx < d; d_idx++) {
                var sum = 0.0;
                for (var row = 0u; row < actual_Br; row++) {
                    sum += Pij[row * Bc + col] * dOi[row * HEAD_DIM + d_idx];
                }
                atomicAdd(&dV[dv_offset + d_idx], sum);
            }
        }

        workgroupBarrier();
    }
}
"""

# ============================================================================
# CROSS ENTROPY LOSS KERNEL
# ============================================================================

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

    // Loss = -log(softmax[target])
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
