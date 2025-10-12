"""WGSL kernels for forward pass operations"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_optimized_matmul_kernel(tile_size: int = 16) -> str:
    """
    Generate matmul kernel with configurable tile size for different GPUs

    Args:
        tile_size: Tile dimension (must be power of 2, typically 8, 16, or 32)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If tile_size is invalid
    """
    if tile_size <= 0 or (tile_size & (tile_size - 1)) != 0:
        raise ValueError(f"tile_size must be power of 2, got {tile_size}")

    if tile_size > 32:
        raise ValueError(f"tile_size too large: {tile_size}. Maximum is 32.")

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

        for (var k = 0u; k < TILE_SIZE; k++) {{
            sum += tile_A[local_row * TILE_SIZE + k] * tile_B[k * TILE_SIZE + local_col];
        }}

        workgroupBarrier();
    }}

    if (row < params.M && col < params.N) {{
        C[row * params.N + col] = sum;
    }}
}}
"""


def create_layernorm_kernel(workgroup_size: int = 256, epsilon: float = 1e-5) -> str:
    """
    Generate layer normalization kernel with configurable workgroup size

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)
        epsilon: Small constant for numerical stability

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
struct NormParams {{
    size: u32,
    n_elements: u32,
}}

@group(0) @binding(0) var<uniform> params: NormParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const EPS: f32 = {epsilon};
const BLOCK_SIZE: u32 = {workgroup_size}u;

var<workgroup> shared_data: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let idx = global_id.x;
    let tid = local_id.x;
    let elem_idx = idx / params.size;

    if (elem_idx >= params.n_elements) {{
        return;
    }}

    let offset = elem_idx * params.size;

    // Compute mean using parallel reduction
    var sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {{
        sum += input[offset + i];
    }}
    shared_data[tid] = sum;
    workgroupBarrier();

    // Reduction
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {{
        if (tid < s) {{
            shared_data[tid] += shared_data[tid + s];
        }}
        workgroupBarrier();
    }}

    let mean = shared_data[0] / f32(params.size);

    // Compute variance
    var var_sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {{
        let diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }}
    shared_data[tid] = var_sum;
    workgroupBarrier();

    // Reduction
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {{
        if (tid < s) {{
            shared_data[tid] += shared_data[tid + s];
        }}
        workgroupBarrier();
    }}

    let variance = shared_data[0] / f32(params.size);
    let inv_std = 1.0 / sqrt(variance + EPS);

    // Normalize and apply affine transformation
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {{
        let normalized = (input[offset + i] - mean) * inv_std;
        output[offset + i] = normalized * gamma[i] + beta[i];
    }}
}}
"""


def create_gelu_kernel(workgroup_size: int = 256) -> str:
    """
    Generate GELU activation kernel with configurable workgroup size

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
// Standard GELU activation

struct GeluParams {{
    size: u32,
}}

@group(0) @binding(0) var<uniform> params: GeluParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEFF: f32 = 0.044715;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;

    if (idx >= params.size) {{
        return;
    }}

    let x = input[idx];
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}}
"""


def create_residual_add_kernel(workgroup_size: int = 256) -> str:
    """
    Generate residual addition kernel with configurable workgroup size

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
struct AddParams {{
    size: u32,
}}

@group(0) @binding(0) var<uniform> params: AddParams;
@group(0) @binding(1) var<storage, read> input_a: array<f32>;
@group(0) @binding(2) var<storage, read> input_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;

    if (idx >= params.size) {{
        return;
    }}

    output[idx] = input_a[idx] + input_b[idx];
}}
"""


def create_embedding_kernel(workgroup_size: int = 256) -> str:
    """
    Generate embedding lookup kernel with configurable workgroup size

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
struct EmbedParams {{
    batch_size: u32,
    seq_len: u32,
    embedding_dim: u32,
}}

@group(0) @binding(0) var<uniform> params: EmbedParams;
@group(0) @binding(1) var<storage, read> embedding_table: array<f32>;
@group(0) @binding(2) var<storage, read> pos_encoding: array<f32>;
@group(0) @binding(3) var<storage, read> input_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    let total = params.batch_size * params.seq_len;

    if (idx >= total) {{
        return;
    }}

    let seq_idx = idx % params.seq_len;
    let token_id = input_ids[idx];
    let D = params.embedding_dim;

    let emb_offset = token_id * D;
    let pos_offset = seq_idx * D;
    let out_offset = idx * D;

    for (var d = 0u; d < D; d++) {{
        output[out_offset + d] = embedding_table[emb_offset + d] + pos_encoding[pos_offset + d];
    }}
}}
"""


def create_bias_add_kernel(workgroup_size: int = 256) -> str:
    """
    Generate bias addition kernel with configurable workgroup size

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
struct BiasParams {{
    size: u32,
    dim: u32,
}}

@group(0) @binding(0) var<uniform> params: BiasParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= params.size) {{
        return;
    }}

    let row = idx / params.dim;
    let col = idx % params.dim;

    output[idx] = input[idx] + bias[col];
}}
"""


def create_attention_kernel(workgroup_size: int = 256) -> str:
    """
    Generate multi-head attention kernel with configurable workgroup size

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
// Multi-head self-attention with causal masking
// Dynamic computation - no hardcoded arrays

struct AttentionParams {{
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
}}

@group(0) @binding(0) var<uniform> params: AttentionParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const BLOCK_SIZE: u32 = {workgroup_size}u;
var<workgroup> shared_scores: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let batch_idx = group_id.z;
    let head_idx = group_id.y;
    let q_pos = group_id.x;
    let tid = local_id.x;

    if (batch_idx >= params.batch_size || q_pos >= params.seq_len) {{
        return;
    }}

    let head_dim = params.head_dim;
    let seq_len = params.seq_len;
    let embedding_dim = params.n_heads * head_dim;
    let scale = 1.0 / sqrt(f32(head_dim));

    let q_offset = batch_idx * seq_len * embedding_dim +
                   q_pos * embedding_dim +
                   head_idx * head_dim;

    // Phase 1: Compute attention scores and find max
    var max_score = -1e9;

    for (var k_pos = tid; k_pos <= q_pos; k_pos += BLOCK_SIZE) {{
        let k_offset = batch_idx * seq_len * embedding_dim +
                      k_pos * embedding_dim +
                      head_idx * head_dim;

        var score = 0.0;
        for (var d = 0u; d < head_dim; d++) {{
            score += Q[q_offset + d] * K[k_offset + d];
        }}
        score *= scale;
        max_score = max(max_score, score);
    }}

    shared_scores[tid] = max_score;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {{
        if (tid < s) {{
            shared_scores[tid] = max(shared_scores[tid], shared_scores[tid + s]);
        }}
        workgroupBarrier();
    }}
    max_score = shared_scores[0];
    workgroupBarrier();

    // Phase 2: Compute exp and sum
    var sum_exp = 0.0;

    for (var k_pos = tid; k_pos <= q_pos; k_pos += BLOCK_SIZE) {{
        let k_offset = batch_idx * seq_len * embedding_dim +
                      k_pos * embedding_dim +
                      head_idx * head_dim;

        var score = 0.0;
        for (var d = 0u; d < head_dim; d++) {{
            score += Q[q_offset + d] * K[k_offset + d];
        }}
        score = score * scale - max_score;
        sum_exp += exp(score);
    }}

    shared_scores[tid] = sum_exp;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {{
        if (tid < s) {{
            shared_scores[tid] += shared_scores[tid + s];
        }}
        workgroupBarrier();
    }}
    sum_exp = shared_scores[0];
    workgroupBarrier();

    // Phase 3: Weighted sum of values
    let dims_per_thread = (head_dim + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    for (var d_block = 0u; d_block < dims_per_thread; d_block++) {{
        let d = tid + d_block * BLOCK_SIZE;
        if (d >= head_dim) {{
            continue;
        }}

        var output_val = 0.0;

        for (var k_pos = 0u; k_pos <= q_pos; k_pos++) {{
            let k_offset = batch_idx * seq_len * embedding_dim +
                          k_pos * embedding_dim +
                          head_idx * head_dim;

            var score = 0.0;
            for (var dd = 0u; dd < head_dim; dd++) {{
                score += Q[q_offset + dd] * K[k_offset + dd];
            }}
            let attn_weight = exp(score * scale - max_score) / sum_exp;

            let v_offset = batch_idx * seq_len * embedding_dim +
                          k_pos * embedding_dim +
                          head_idx * head_dim;
            output_val += attn_weight * V[v_offset + d];
        }}

        let out_offset = batch_idx * seq_len * embedding_dim +
                        q_pos * embedding_dim +
                        head_idx * head_dim;
        output[out_offset + d] = output_val;
    }}
}}
"""


def create_flash_attention_kernel(
    head_dim: int = 64, Bc: int = 32, Br: int = 32
) -> str:
    """
    Generate FlashAttention kernel with configurable parameters

    Args:
        head_dim: Dimension per attention head (64, 128, or 256)
        Bc: Block size for K/V (columns) - typically 16, 32, or 64
        Br: Block size for Q (rows) - typically 16, 32, or 64

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If parameters are invalid or exceed workgroup memory limits
    """
    if head_dim not in [64, 128, 256]:
        raise ValueError(f"head_dim must be 64, 128, or 256, got {head_dim}")

    if Bc <= 0 or (Bc & (Bc - 1)) != 0:
        raise ValueError(f"Bc must be power of 2, got {Bc}")

    if Br <= 0 or (Br & (Br - 1)) != 0:
        raise ValueError(f"Br must be power of 2, got {Br}")

    # Estimate workgroup memory usage
    qi_size = Br * head_dim
    kj_size = Bc * head_dim
    vj_size = Bc * head_dim
    sij_size = Br * Bc
    pij_size = Br * Bc
    dOi_size = Br * head_dim
    Oi_size = Br * head_dim
    dPij_size = Br * Bc
    dSij_size = Br * Bc
    Di_size = Br
    mi_old_size = Br

    total_workgroup_f32 = (
        qi_size
        + kj_size
        + vj_size
        + sij_size
        + pij_size
        + dOi_size
        + Oi_size
        + dPij_size
        + dSij_size
        + Di_size
        + mi_old_size
    )
    total_workgroup_bytes = total_workgroup_f32 * 4

    # Typical limits: 16KB-32KB for most GPUs
    if total_workgroup_bytes > 65536:  # 64KB (very conservative)
        raise ValueError(
            f"Workgroup memory {total_workgroup_bytes} bytes exceeds 64KB limit. "
            f"Try smaller head_dim, Bc, or Br values."
        )

    return f"""
// FlashAttention: Memory-efficient attention using tiling and online softmax
// Based on: Dao et al. 2022 - "FlashAttention: Fast and Memory-Efficient Exact Attention"
//
// Parameters: head_dim={head_dim}, Bc={Bc}, Br={Br}
// Workgroup memory: {total_workgroup_bytes} bytes

struct FlashAttentionParams {{
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    Bc: u32,
    Br: u32,
}}

@group(0) @binding(0) var<uniform> params: FlashAttentionParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@group(0) @binding(5) var<storage, read_write> L: array<f32>;
@group(0) @binding(6) var<storage, read_write> M: array<f32>;

const Bc: u32 = {Bc}u;
const Br: u32 = {Br}u;
const HEAD_DIM: u32 = {head_dim}u;

// Shared memory tiles
var<workgroup> Qi: array<f32, {qi_size}>;
var<workgroup> Kj: array<f32, {kj_size}>;
var<workgroup> Vj: array<f32, {vj_size}>;
var<workgroup> Sij: array<f32, {sij_size}>;
var<workgroup> Pij: array<f32, {pij_size}>;
var<workgroup> mi_old_storage: array<f32, {Br}>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let batch_idx = workgroup_id.z;
    let head_idx = workgroup_id.y;
    let block_row = workgroup_id.x;
    let tid = local_id.x;

    if (batch_idx >= params.batch_size) {{
        return;
    }}

    let d = params.head_dim;
    let N = params.seq_len;
    let embedding_dim = params.n_heads * d;
    let scale = 1.0 / sqrt(f32(d));

    let q_start = block_row * Br;
    let q_end = min(q_start + Br, N);
    let actual_Br = q_end - q_start;

    if (actual_Br == 0u) {{
        return;
    }}

    // Load Q block
    for (var i = tid; i < actual_Br * d; i += 32u) {{
        let local_row = i / d;
        let local_col = i % d;
        let global_row = q_start + local_row;

        let q_offset = batch_idx * N * embedding_dim +
                      global_row * embedding_dim +
                      head_idx * d + local_col;

        Qi[local_row * HEAD_DIM + local_col] = Q[q_offset];
    }}
    workgroupBarrier();

    // Initialize output accumulators
    var Oi: array<f32, {head_dim * Br}>;
    var mi: array<f32, {Br}>;
    var li: array<f32, {Br}>;

    if (tid == 0u) {{
        for (var i = 0u; i < actual_Br; i++) {{
            mi[i] = -1e9;
            li[i] = 0.0;
            for (var d_idx = 0u; d_idx < d; d_idx++) {{
                Oi[i * HEAD_DIM + d_idx] = 0.0;
            }}
        }}
    }}
    workgroupBarrier();

    let num_kv_blocks = (N + Bc - 1u) / Bc;

    for (var block_col = 0u; block_col < num_kv_blocks; block_col++) {{
        let kv_start = block_col * Bc;
        let kv_end = min(kv_start + Bc, N);
        let actual_Bc = kv_end - kv_start;

        if (kv_start > q_end) {{
            break;
        }}

        // Load K and V blocks
        for (var i = tid; i < actual_Bc * d; i += 32u) {{
            let local_row = i / d;
            let local_col = i % d;
            let global_row = kv_start + local_row;

            let kv_offset = batch_idx * N * embedding_dim +
                          global_row * embedding_dim +
                          head_idx * d + local_col;

            Kj[local_row * HEAD_DIM + local_col] = K[kv_offset];
            Vj[local_row * HEAD_DIM + local_col] = V[kv_offset];
        }}
        workgroupBarrier();

        // Compute scores
        for (var i = tid; i < actual_Br * actual_Bc; i += 32u) {{
            let row = i / actual_Bc;
            let col = i % actual_Bc;

            var score = 0.0;
            for (var k = 0u; k < d; k++) {{
                score += Qi[row * HEAD_DIM + k] * Kj[col * HEAD_DIM + k];
            }}

            let q_pos = q_start + row;
            let kv_pos = kv_start + col;
            if (kv_pos <= q_pos) {{
                Sij[row * Bc + col] = score * scale;
            }} else {{
                Sij[row * Bc + col] = -1e9;
            }}
        }}
        workgroupBarrier();

        // Online softmax update
        if (tid == 0u) {{
            for (var row = 0u; row < actual_Br; row++) {{
                let mi_old = mi[row];
                mi_old_storage[row] = mi_old;

                var mi_new = mi_old;
                for (var col = 0u; col < actual_Bc; col++) {{
                    mi_new = max(mi_new, Sij[row * Bc + col]);
                }}

                var li_new = 0.0;
                for (var col = 0u; col < actual_Bc; col++) {{
                    let p = exp(Sij[row * Bc + col] - mi_new);
                    Pij[row * Bc + col] = p;
                    li_new += p;
                }}

                let li_old = li[row];
                mi[row] = mi_new;
                li[row] = li_old * exp(mi_old - mi_new) + li_new;
            }}
        }}
        workgroupBarrier();

        // Update output
        for (var row = tid; row < actual_Br; row += 32u) {{
            let correction = exp(mi_old_storage[row] - mi[row]);

            for (var d_idx = 0u; d_idx < d; d_idx++) {{
                Oi[row * HEAD_DIM + d_idx] *= correction;
            }}

            for (var d_idx = 0u; d_idx < d; d_idx++) {{
                var sum = 0.0;
                for (var col = 0u; col < actual_Bc; col++) {{
                    sum += Pij[row * Bc + col] * Vj[col * HEAD_DIM + d_idx];
                }}
                Oi[row * HEAD_DIM + d_idx] += sum;
            }}
        }}
        workgroupBarrier();
    }}

    // Write output
    for (var row = tid; row < actual_Br; row += 32u) {{
        let global_row = q_start + row;

        for (var d_idx = 0u; d_idx < d; d_idx++) {{
            let o_offset = batch_idx * N * embedding_dim +
                          global_row * embedding_dim +
                          head_idx * d + d_idx;

            O[o_offset] = Oi[row * HEAD_DIM + d_idx] / li[row];
        }}

        let stats_offset = batch_idx * N * params.n_heads +
                          global_row * params.n_heads + head_idx;
        L[stats_offset] = li[row];
        M[stats_offset] = mi[row];
    }}
}}
"""


def create_transpose_kernel(tile_size: int = 16) -> str:
    """
    Generate matrix transpose kernel with bank conflict avoidance

    Args:
        tile_size: Tile dimension (must be power of 2, typically 8, 16, or 32)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If tile_size is invalid
    """
    if tile_size <= 0 or (tile_size & (tile_size - 1)) != 0:
        raise ValueError(f"tile_size must be power of 2, got {tile_size}")

    if tile_size > 32:
        raise ValueError(f"tile_size too large: {tile_size}. Maximum is 32.")

    # Add padding to avoid bank conflicts (tile_size + 1)
    padded_size = tile_size * (tile_size + 1)

    return f"""
// Matrix transpose with bank conflict avoidance
// Tile size: {tile_size}x{tile_size}

struct TransposeParams {{
    rows: u32,
    cols: u32,
}}

@group(0) @binding(0) var<uniform> params: TransposeParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const TILE_SIZE: u32 = {tile_size}u;
var<workgroup> tile: array<f32, {padded_size}>;

@compute @workgroup_size({tile_size}, {tile_size})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    if (row < params.rows && col < params.cols) {{
        tile[local_row * {tile_size + 1}u + local_col] = input[row * params.cols + col];
    }}

    workgroupBarrier();

    let out_row = col;
    let out_col = row;

    if (out_row < params.cols && out_col < params.rows) {{
        output[out_row * params.rows + out_col] = tile[local_col * {tile_size + 1}u + local_row];
    }}
}}
"""


def create_extract_last_tokens_kernel(workgroup_size: int = 256) -> str:
    """
    Generate kernel to extract last token from each sequence

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
// Extract last token from each sequence in batch

struct ExtractParams {{
    batch_size: u32,
    seq_len: u32,
    embedding_dim: u32,
}}

@group(0) @binding(0) var<uniform> params: ExtractParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let batch_idx = global_id.y;
    let dim_idx = global_id.x;

    if (batch_idx >= params.batch_size || dim_idx >= params.embedding_dim) {{
        return;
    }}

    let last_pos = params.seq_len - 1u;
    let input_offset = batch_idx * params.seq_len * params.embedding_dim +
                       last_pos * params.embedding_dim + dim_idx;
    let output_offset = batch_idx * params.embedding_dim + dim_idx;

    output[output_offset] = input[input_offset];
}}
"""


# ============================================================================
# FACTORY FUNCTIONS - Generate kernels from config
# ============================================================================


def get_matmul_kernel_from_config(config) -> str:
    """Get matmul kernel configured from GPUConfig"""
    return create_optimized_matmul_kernel(config.matmul_tile_size)


def get_layernorm_kernel_from_config(config) -> str:
    """Get layernorm kernel configured from GPUConfig"""
    return create_layernorm_kernel(
        config.layernorm_workgroup_size, config.layernorm_epsilon
    )


def get_gelu_kernel_from_config(config) -> str:
    """Get GELU kernel configured from GPUConfig"""
    return create_gelu_kernel(config.default_workgroup_size)


def get_residual_add_kernel_from_config(config) -> str:
    """Get residual add kernel configured from GPUConfig"""
    return create_residual_add_kernel(config.default_workgroup_size)


def get_embedding_kernel_from_config(config) -> str:
    """Get embedding kernel configured from GPUConfig"""
    return create_embedding_kernel(config.default_workgroup_size)


def get_bias_add_kernel_from_config(config) -> str:
    """Get bias add kernel configured from GPUConfig"""
    return create_bias_add_kernel(config.default_workgroup_size)


def get_attention_kernel_from_config(config) -> str:
    """Get attention kernel configured from GPUConfig"""
    return create_attention_kernel(config.attention_workgroup_size)


def get_flash_attention_kernel_from_config(config) -> str:
    """Get FlashAttention kernel configured from GPUConfig"""
    return create_flash_attention_kernel(
        config.flash_attn_max_head_dim, config.flash_attn_bc, config.flash_attn_br
    )


def get_transpose_kernel_from_config(config) -> str:
    """Get transpose kernel configured from GPUConfig"""
    return create_transpose_kernel(
        config.matmul_tile_size
    )  # Use same tile size as matmul


def get_extract_last_tokens_kernel_from_config(config) -> str:
    """Get extract last tokens kernel configured from GPUConfig"""
    return create_extract_last_tokens_kernel(config.default_workgroup_size)


# ============================================================================
# DEFAULT KERNELS (for backward compatibility)
# ============================================================================

# These are provided for backward compatibility only.
# New code should use get_*_kernel_from_config() functions instead.

TILED_MATMUL_KERNEL = create_optimized_matmul_kernel(16)
LAYERNORM_KERNEL = create_layernorm_kernel(256)
GELU_KERNEL = create_gelu_kernel(256)
RESIDUAL_ADD_KERNEL = create_residual_add_kernel(256)
EMBEDDING_KERNEL = create_embedding_kernel(256)
BIAS_ADD_KERNEL = create_bias_add_kernel(256)
MULTIHEAD_ATTENTION_KERNEL = create_attention_kernel(256)
FLASHATTENTION_FORWARD_KERNEL = create_flash_attention_kernel(64, 32, 32)
TRANSPOSE_KERNEL = create_transpose_kernel(16)
EXTRACT_LAST_TOKENS_KERNEL = create_extract_last_tokens_kernel(256)
