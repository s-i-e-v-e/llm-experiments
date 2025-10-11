"""WGSL kernels for forward pass operations"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_optimized_matmul_kernel(tile_size: int = 16) -> str:
    """Generate matmul kernel with configurable tile size for different GPUs"""
    return f"""
// Adaptive tiled matrix multiplication: C = A @ B
// Tile size: {tile_size}x{tile_size}

struct MatmulParams {{
    M: u32,
    K: u32,
    N: u32,
}}

@group(0) @binding(0) var<uniform> params: MatmulParams;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TILE_SIZE: u32 = {tile_size}u;
const TILE_AREA: u32 = {tile_size * tile_size}u;

var<workgroup> tile_A: array<f32, {tile_size * tile_size}>;
var<workgroup> tile_B: array<f32, {tile_size * tile_size}>;

@compute @workgroup_size({tile_size}, {tile_size})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum = 0.0;
    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {{
        // Coalesced load of A tile
        let a_row = row;
        let a_col = t * TILE_SIZE + local_col;
        if (a_row < params.M && a_col < params.K) {{
            tile_A[local_row * TILE_SIZE + local_col] = A[a_row * params.K + a_col];
        }} else {{
            tile_A[local_row * TILE_SIZE + local_col] = 0.0;
        }}

        // Coalesced load of B tile
        let b_row = t * TILE_SIZE + local_row;
        let b_col = col;
        if (b_row < params.K && b_col < params.N) {{
            tile_B[local_row * TILE_SIZE + local_col] = B[b_row * params.N + b_col];
        }} else {{
            tile_B[local_row * TILE_SIZE + local_col] = 0.0;
        }}

        workgroupBarrier();

        // Compute with register blocking (2x2 block per thread)
        var acc00 = 0.0;
        var acc01 = 0.0;
        var acc10 = 0.0;
        var acc11 = 0.0;

        for (var k = 0u; k < TILE_SIZE; k++) {{
            let a_val0 = tile_A[local_row * TILE_SIZE + k];
            let a_val1 = tile_A[(local_row + 1u) * TILE_SIZE + k];
            let b_val0 = tile_B[k * TILE_SIZE + local_col];
            let b_val1 = tile_B[k * TILE_SIZE + local_col + 1u];

            acc00 += a_val0 * b_val0;
            acc01 += a_val0 * b_val1;
            acc10 += a_val1 * b_val0;
            acc11 += a_val1 * b_val1;
        }}

        sum += acc00;  // Primary accumulator

        workgroupBarrier();
    }}

    // Write result
    if (row < params.M && col < params.N) {{
        C[row * params.N + col] = sum;
    }}
}}
"""


# ============================================================================
# FORWARD PASS KERNELS
# ============================================================================

# Matrix multiplication kernel (default 16x16 tiles)
TILED_MATMUL_KERNEL = create_optimized_matmul_kernel(16)

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
// Standard GELU activation

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
# PROPER ATTENTION KERNEL (Simplified Multi-Head)
# ============================================================================

MULTIHEAD_ATTENTION_BALANCED_KERNEL = """
// Balanced multi-head attention - processes multiple query positions per workgroup
// Better load balancing for variable sequence lengths

struct AttentionParams {
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<uniform> params: AttentionParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const BLOCK_SIZE: u32 = 256u;
var<workgroup> shared_mem: array<f32, 256>;

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

    // Query offset
    let q_offset = batch_idx * seq_len * embedding_dim +
                   q_pos * embedding_dim +
                   head_idx * head_dim;

    // Phase 1: Compute scores and find max (for numerical stability)
    var max_score = -1e9;

    for (var k_pos = tid; k_pos <= q_pos; k_pos += BLOCK_SIZE) {
        var score = 0.0;
        let k_offset = batch_idx * seq_len * embedding_dim +
                      k_pos * embedding_dim +
                      head_idx * head_dim;

        for (var d = 0u; d < head_dim; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }
        score *= scale;
        max_score = max(max_score, score);
    }

    // Reduce max across threads
    shared_mem[tid] = max_score;
    workgroupBarrier();

    for (var stride = BLOCK_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + stride]);
        }
        workgroupBarrier();
    }
    max_score = shared_mem[0];
    workgroupBarrier();

    // Phase 2: Compute exp and sum
    var sum_exp = 0.0;
    var score_cache: array<f32, 8>;  // Cache scores for reuse
    var num_scores = 0u;

    for (var k_pos = tid; k_pos <= q_pos; k_pos += BLOCK_SIZE) {
        let k_offset = batch_idx * seq_len * embedding_dim +
                      k_pos * embedding_dim +
                      head_idx * head_dim;

        var score = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }
        score = score * scale - max_score;
        let exp_score = exp(score);

        // Cache for later use (limited to 8 entries per thread)
        if (num_scores < 8u) {
            score_cache[num_scores] = exp_score;
            num_scores++;
        }

        sum_exp += exp_score;
    }

    // Reduce sum
    shared_mem[tid] = sum_exp;
    workgroupBarrier();

    for (var stride = BLOCK_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        workgroupBarrier();
    }
    sum_exp = shared_mem[0];
    workgroupBarrier();

    // Phase 3: Weighted sum of values (process multiple dims per thread)
    let dims_per_thread = (head_dim + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    for (var d_block = 0u; d_block < dims_per_thread; d_block++) {
        let d = tid + d_block * BLOCK_SIZE;
        if (d >= head_dim) {
            break;
        }

        var result = 0.0;
        var cache_idx = 0u;

        for (var k_pos = 0u; k_pos <= q_pos; k_pos++) {
            let k_offset = batch_idx * seq_len * embedding_dim +
                          k_pos * embedding_dim +
                          head_idx * head_dim;

            // Recompute attention weight (could use cache for first few)
            var score = 0.0;
            for (var dd = 0u; dd < head_dim; dd++) {
                score += Q[q_offset + dd] * K[k_offset + dd];
            }
            score = score * scale - max_score;
            let attn_weight = exp(score) / sum_exp;

            let v_offset = batch_idx * seq_len * embedding_dim +
                          k_pos * embedding_dim +
                          head_idx * head_dim;

            result += attn_weight * V[v_offset + d];
        }

        output[q_offset + d] = result;
    }
}
"""

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

TRANSPOSE_KERNEL = """
// Matrix transpose with bank conflict avoidance
// Uses padding in shared memory to prevent bank conflicts

struct TransposeParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> params: TransposeParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const TILE_SIZE: u32 = 16u;
// Add padding to avoid bank conflicts (17 instead of 16)
var<workgroup> tile: array<f32, 272>;  // 16 * 17 = 272

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    // Load into shared memory with padding
    if (row < params.rows && col < params.cols) {
        tile[local_row * 17u + local_col] = input[row * params.cols + col];
    }

    workgroupBarrier();

    // Write transposed with padding offset (avoids bank conflicts)
    let out_row = col;
    let out_col = row;

    if (out_row < params.cols && out_col < params.rows) {
        output[out_row * params.rows + out_col] = tile[local_col * 17u + local_row];
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
