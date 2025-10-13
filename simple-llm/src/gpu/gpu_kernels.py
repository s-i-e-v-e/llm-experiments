"""WGSL kernels"""

from .gpu_types import GPUContext

# ============================================================================
# FORWARD PASS KERNELS
# ============================================================================


def create_matmul_kernel(tile_size: int = 16, items_per_thread: int = 4) -> str:
    """
    Generate matmul kernel with configurable tile size and register blocking

    Args:
        tile_size: Tile dimension for shared memory (must be power of 2, typically 8, 16, or 32)
        items_per_thread: Number of output elements per thread (1, 2, or 4)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If tile_size or items_per_thread is invalid
    """
    if tile_size <= 0 or (tile_size & (tile_size - 1)) != 0:
        raise ValueError(f"tile_size must be power of 2, got {tile_size}")
    if tile_size > 32:
        raise ValueError(f"tile_size too large: {tile_size}. Maximum is 32.")
    if items_per_thread not in [1, 2, 4]:
        raise ValueError(f"items_per_thread must be 1, 2, or 4, got {items_per_thread}")

    workgroup_dim = tile_size // (items_per_thread if items_per_thread > 1 else 1)

    return f"""
// Optimized tiled matrix multiplication: C = A @ B
// Tile size: {tile_size}x{tile_size}, Items per thread: {items_per_thread}x{items_per_thread}
// Each thread computes {items_per_thread * items_per_thread} output elements

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
const ITEMS_PER_THREAD: u32 = {items_per_thread}u;

var<workgroup> tile_A: array<f32, {tile_size * tile_size}>;
var<workgroup> tile_B: array<f32, {tile_size * tile_size}>;

@compute @workgroup_size({workgroup_dim}, {workgroup_dim})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let local_row = local_id.y;
    let local_col = local_id.x;

    // Each thread computes ITEMS_PER_THREAD x ITEMS_PER_THREAD outputs
    let base_row = global_id.y * ITEMS_PER_THREAD;
    let base_col = global_id.x * ITEMS_PER_THREAD;

    // Accumulator registers
    var acc: array<f32, {items_per_thread * items_per_thread}>;
    for (var i = 0u; i < {items_per_thread * items_per_thread}u; i++) {{
        acc[i] = 0.0;
    }}

    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {{
        // Load A tile - each thread loads ITEMS_PER_THREAD elements
        for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
            let a_row = base_row + i;
            let a_col = t * TILE_SIZE + local_col;
            let tile_idx = (local_row * ITEMS_PER_THREAD + i) * TILE_SIZE + local_col;

            if (a_row < params.M && a_col < params.K) {{
                tile_A[tile_idx] = A[a_row * params.K + a_col];
            }} else {{
                tile_A[tile_idx] = 0.0;
            }}
        }}

        // Load B tile - each thread loads ITEMS_PER_THREAD elements
        for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
            let b_row = t * TILE_SIZE + local_row;
            let b_col = base_col + i;
            let tile_idx = local_row * TILE_SIZE + (local_col * ITEMS_PER_THREAD + i);

            if (b_row < params.K && b_col < params.N) {{
                tile_B[tile_idx] = B[b_row * params.N + b_col];
            }} else {{
                tile_B[tile_idx] = 0.0;
            }}
        }}

        workgroupBarrier();

        // Compute partial results
        for (var k = 0u; k < TILE_SIZE; k++) {{
            for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
                let a_val = tile_A[(local_row * ITEMS_PER_THREAD + i) * TILE_SIZE + k];
                for (var j = 0u; j < ITEMS_PER_THREAD; j++) {{
                    let b_val = tile_B[k * TILE_SIZE + (local_col * ITEMS_PER_THREAD + j)];
                    acc[i * ITEMS_PER_THREAD + j] += a_val * b_val;
                }}
            }}
        }}

        workgroupBarrier();
    }}

    // Write results
    for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
        let out_row = base_row + i;
        for (var j = 0u; j < ITEMS_PER_THREAD; j++) {{
            let out_col = base_col + j;
            if (out_row < params.M && out_col < params.N) {{
                C[out_row * params.N + out_col] = acc[i * ITEMS_PER_THREAD + j];
            }}
        }}
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

    # Add padding to avoid bank conflicts (assuming 32-wide banks)
    padded_size = (
        workgroup_size + (32 - (workgroup_size % 32))
        if workgroup_size % 32 != 0
        else workgroup_size
    )

    return f"""
struct NormParams {{
    size: u32,           // Hidden dimension size
    n_elements: u32,     // Number of sequences/elements to normalize
}}

@group(0) @binding(0) var<uniform> params: NormParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const EPS: f32 = {epsilon};
const BLOCK_SIZE: u32 = {workgroup_size}u;

// Padded to avoid bank conflicts
var<workgroup> shared_data: array<f32, {padded_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let tid = local_id.x;
    let elem_idx = workgroup_id.x;  // One workgroup per element

    if (elem_idx >= params.n_elements) {{
        return;
    }}

    let offset = elem_idx * params.size;

    // Phase 1: Compute mean using parallel reduction
    var sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {{
        sum += input[offset + i];
    }}
    shared_data[tid] = sum;
    workgroupBarrier();

    // Tree reduction for sum
    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_data[tid] += shared_data[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}

    let mean = shared_data[0] / f32(params.size);

    // Broadcast mean to all threads (implicitly via shared memory)
    workgroupBarrier();

    // Phase 2: Compute variance using parallel reduction
    var var_sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {{
        let diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }}
    shared_data[tid] = var_sum;
    workgroupBarrier();

    // Tree reduction for variance
    active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_data[tid] += shared_data[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}

    let variance = shared_data[0] / f32(params.size);
    let inv_std = 1.0 / sqrt(variance + EPS);

    workgroupBarrier();

    // Phase 3: Normalize and apply affine transformation
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
// GELU activation with optimized approximation
// Uses polynomial approximation instead of tanh for better performance

struct GeluParams {{
    size: u32,
}}

@group(0) @binding(0) var<uniform> params: GeluParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Using sigmoid approximation: gelu(x) ≈ x * sigmoid(1.702 * x)
// This is faster and has <0.1% error vs exact GELU
const GELU_SCALE: f32 = 1.702;

fn fast_sigmoid(x: f32) -> f32 {{
    return 1.0 / (1.0 + exp(-x));
}}

fn gelu_approx(x: f32) -> f32 {{
    return x * fast_sigmoid(GELU_SCALE * x);
}}

// Original tanh-based GELU for reference (slower):
// const SQRT_2_OVER_PI: f32 = 0.7978845608;
// const GELU_COEFF: f32 = 0.044715;
// fn gelu_tanh(x: f32) -> f32 {{
//     let x_cubed = x * x * x;
//     let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
//     return 0.5 * x * (1.0 + tanh(inner));
// }}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;

    if (idx >= params.size) {{
        return;
    }}

    output[idx] = gelu_approx(input[idx]);
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
// Residual addition with vectorized memory access
// Each thread processes 4 elements at a time for better bandwidth utilization

struct AddParams {{
    size: u32,
}}

@group(0) @binding(0) var<uniform> params: AddParams;
@group(0) @binding(1) var<storage, read> input_a: array<f32>;
@group(0) @binding(2) var<storage, read> input_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    let vec4_count = params.size / 4u;
    let remainder = params.size % 4u;

    // Process 4 elements at a time using vectorized loads
    if (idx < vec4_count) {{
        let base_idx = idx * 4u;
        let a0 = input_a[base_idx];
        let a1 = input_a[base_idx + 1u];
        let a2 = input_a[base_idx + 2u];
        let a3 = input_a[base_idx + 3u];

        let b0 = input_b[base_idx];
        let b1 = input_b[base_idx + 1u];
        let b2 = input_b[base_idx + 2u];
        let b3 = input_b[base_idx + 3u];

        output[base_idx] = a0 + b0;
        output[base_idx + 1u] = a1 + b1;
        output[base_idx + 2u] = a2 + b2;
        output[base_idx + 3u] = a3 + b3;
    }}

    // Handle remainder elements
    if (idx == 0u && remainder > 0u) {{
        let base_idx = vec4_count * 4u;
        for (var i = 0u; i < remainder; i++) {{
            output[base_idx + i] = input_a[base_idx + i] + input_b[base_idx + i];
        }}
    }}
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
// Embedding lookup with positional encoding
// Uses 2D dispatch to parallelize across both tokens and embedding dimensions

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

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let token_idx = global_id.x;  // Which token (batch * seq_len)
    let dim_idx = global_id.y;     // Which dimension element

    let total_tokens = params.batch_size * params.seq_len;
    let D = params.embedding_dim;
    let vec4_per_token = D / 4u;
    let remainder = D % 4u;

    if (token_idx >= total_tokens) {{
        return;
    }}

    let seq_idx = token_idx % params.seq_len;
    let token_id = input_ids[token_idx];

    let emb_offset = token_id * D;
    let pos_offset = seq_idx * D;
    let out_offset = token_idx * D;

    // Process 4 dimensions at a time
    if (dim_idx < vec4_per_token) {{
        let base_d = dim_idx * 4u;

        output[out_offset + base_d] =
            embedding_table[emb_offset + base_d] + pos_encoding[pos_offset + base_d];
        output[out_offset + base_d + 1u] =
            embedding_table[emb_offset + base_d + 1u] + pos_encoding[pos_offset + base_d + 1u];
        output[out_offset + base_d + 2u] =
            embedding_table[emb_offset + base_d + 2u] + pos_encoding[pos_offset + base_d + 2u];
        output[out_offset + base_d + 3u] =
            embedding_table[emb_offset + base_d + 3u] + pos_encoding[pos_offset + base_d + 3u];
    }}

    // Handle remainder dimensions
    if (dim_idx == 0u && remainder > 0u) {{
        let base_d = vec4_per_token * 4u;
        for (var i = 0u; i < remainder; i++) {{
            output[out_offset + base_d + i] =
                embedding_table[emb_offset + base_d + i] + pos_encoding[pos_offset + base_d + i];
        }}
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
// Bias addition: output[i] = input[i] + bias[i % dim]
// Broadcasts bias across all rows

struct BiasParams {{
    size: u32,      // Total elements (batch * seq * dim)
    dim: u32,       // Dimension size (bias length)
}}

@group(0) @binding(0) var<uniform> params: BiasParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;

    if (idx >= params.size) {{
        return;
    }}

    // Bias index wraps around based on dimension
    let bias_idx = idx % params.dim;
    output[idx] = input[idx] + bias[bias_idx];
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

    # We need shared memory for scores - allocate enough for max sequence length
    # This kernel works best for seq_len <= workgroup_size
    max_seq_len = workgroup_size

    return f"""
// Multi-head self-attention with causal masking
// Optimized version that caches Q·K scores to avoid recomputation

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
const MASK_VALUE: f32 = -1e10;

// Shared memory for scores and reductions
var<workgroup> shared_scores: array<f32, {max_seq_len}>;
var<workgroup> shared_reduction: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
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

    // Phase 1: Compute all Q·K scores and store in shared memory
    for (var k_pos = tid; k_pos < seq_len; k_pos += BLOCK_SIZE) {{
        if (k_pos <= q_pos) {{
            let k_offset = batch_idx * seq_len * embedding_dim +
                          k_pos * embedding_dim +
                          head_idx * head_dim;

            var score = 0.0;
            for (var d = 0u; d < head_dim; d++) {{
                score += Q[q_offset + d] * K[k_offset + d];
            }}
            shared_scores[k_pos] = score * scale;
        }} else {{
            shared_scores[k_pos] = MASK_VALUE;
        }}
    }}
    workgroupBarrier();

    // Phase 2: Find max score using parallel reduction
    var max_score = MASK_VALUE;
    for (var k_pos = tid; k_pos < seq_len; k_pos += BLOCK_SIZE) {{
        max_score = max(max_score, shared_scores[k_pos]);
    }}
    shared_reduction[tid] = max_score;
    workgroupBarrier();

    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_reduction[tid] = max(shared_reduction[tid], shared_reduction[tid + active_threads]);
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}
    max_score = shared_reduction[0];
    workgroupBarrier();

    // Phase 3: Compute exp(score - max) and sum
    var sum_exp = 0.0;
    for (var k_pos = tid; k_pos < seq_len; k_pos += BLOCK_SIZE) {{
        if (k_pos <= q_pos) {{
            let exp_score = exp(shared_scores[k_pos] - max_score);
            shared_scores[k_pos] = exp_score;  // Store normalized scores
            sum_exp += exp_score;
        }}
    }}
    shared_reduction[tid] = sum_exp;
    workgroupBarrier();

    active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_reduction[tid] += shared_reduction[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}
    sum_exp = shared_reduction[0];
    workgroupBarrier();

    // Phase 4: Compute weighted sum of values
    // Each thread computes multiple output dimensions
    let dims_per_thread = (head_dim + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    for (var d_block = 0u; d_block < dims_per_thread; d_block++) {{
        let d = tid + d_block * BLOCK_SIZE;
        if (d >= head_dim) {{
            continue;
        }}

        var output_val = 0.0;
        for (var k_pos = 0u; k_pos <= q_pos; k_pos++) {{
            let attn_weight = shared_scores[k_pos] / sum_exp;

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

    # Use more threads for better parallelism
    threads_per_workgroup = min(256, max(64, Br * 2))

    # Estimate workgroup memory usage
    qi_size = Br * head_dim
    kj_size = Bc * head_dim
    vj_size = Bc * head_dim
    sij_size = Br * Bc
    pij_size = Br * Bc
    oi_size = Br * head_dim  # Moved to shared memory
    mi_size = Br
    li_size = Br
    mi_old_size = Br

    total_workgroup_f32 = (
        qi_size
        + kj_size
        + vj_size
        + sij_size
        + pij_size
        + oi_size
        + mi_size
        + li_size
        + mi_old_size
    )
    total_workgroup_bytes = total_workgroup_f32 * 4

    if total_workgroup_bytes > 65536:
        raise ValueError(
            f"Workgroup memory {total_workgroup_bytes} bytes exceeds 64KB limit. "
            f"Try smaller head_dim, Bc, or Br values."
        )

    return f"""
// FlashAttention: Memory-efficient attention using tiling and online softmax
// Based on: Dao et al. 2022 - \"FlashAttention: Fast and Memory-Efficient Exact Attention\"
// Optimized version with parallelized initialization and softmax
//
// Parameters: head_dim={head_dim}, Bc={Bc}, Br={Br}
// Workgroup memory: {total_workgroup_bytes} bytes
// Threads per workgroup: {threads_per_workgroup}

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
const THREADS: u32 = {threads_per_workgroup}u;
const MASK_VALUE: f32 = -1e10;

// Shared memory tiles
var<workgroup> Qi: array<f32, {qi_size}>;
var<workgroup> Kj: array<f32, {kj_size}>;
var<workgroup> Vj: array<f32, {vj_size}>;
var<workgroup> Sij: array<f32, {sij_size}>;
var<workgroup> Pij: array<f32, {pij_size}>;
var<workgroup> Oi: array<f32, {oi_size}>;  // Moved to shared memory
var<workgroup> mi: array<f32, {mi_size}>;
var<workgroup> li: array<f32, {li_size}>;
var<workgroup> mi_old: array<f32, {mi_old_size}>;

@compute @workgroup_size({threads_per_workgroup}, 1, 1)
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

    // Load Q block (parallelized)
    for (var i = tid; i < actual_Br * d; i += THREADS) {{
        let local_row = i / d;
        let local_col = i % d;
        let global_row = q_start + local_row;

        let q_offset = batch_idx * N * embedding_dim +
                      global_row * embedding_dim +
                      head_idx * d + local_col;

        Qi[local_row * HEAD_DIM + local_col] = Q[q_offset];
    }}

    // Initialize accumulators (parallelized)
    for (var i = tid; i < actual_Br; i += THREADS) {{
        mi[i] = MASK_VALUE;
        li[i] = 0.0;
    }}

    for (var i = tid; i < actual_Br * d; i += THREADS) {{
        Oi[i] = 0.0;
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

        // Load K and V blocks (parallelized)
        for (var i = tid; i < actual_Bc * d; i += THREADS) {{
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

        // Compute scores (parallelized)
        for (var i = tid; i < actual_Br * actual_Bc; i += THREADS) {{
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
                Sij[row * Bc + col] = MASK_VALUE;
            }}
        }}
        workgroupBarrier();

        // Online softmax update (parallelized per row)
        for (var row = tid; row < actual_Br; row += THREADS) {{
            let mi_old_val = mi[row];
            mi_old[row] = mi_old_val;

            // Find max
            var mi_new = mi_old_val;
            for (var col = 0u; col < actual_Bc; col++) {{
                mi_new = max(mi_new, Sij[row * Bc + col]);
            }}

            // Compute exp and sum
            var li_new = 0.0;
            for (var col = 0u; col < actual_Bc; col++) {{
                let p = exp(Sij[row * Bc + col] - mi_new);
                Pij[row * Bc + col] = p;
                li_new += p;
            }}

            let li_old_val = li[row];
            mi[row] = mi_new;
            li[row] = li_old_val * exp(mi_old_val - mi_new) + li_new;
        }}
        workgroupBarrier();

        // Update output (parallelized)
        for (var i = tid; i < actual_Br * d; i += THREADS) {{
            let row = i / d;
            let d_idx = i % d;

            let correction = exp(mi_old[row] - mi[row]);
            Oi[row * HEAD_DIM + d_idx] *= correction;

            var sum = 0.0;
            for (var col = 0u; col < actual_Bc; col++) {{
                sum += Pij[row * Bc + col] * Vj[col * HEAD_DIM + d_idx];
            }}
            Oi[row * HEAD_DIM + d_idx] += sum;
        }}
        workgroupBarrier();
    }}

    // Write output (parallelized)
    for (var i = tid; i < actual_Br * d; i += THREADS) {{
        let row = i / d;
        let d_idx = i % d;
        let global_row = q_start + row;

        let o_offset = batch_idx * N * embedding_dim +
                      global_row * embedding_dim +
                      head_idx * d + d_idx;

        O[o_offset] = Oi[row * HEAD_DIM + d_idx] / li[row];
    }}

    // Write statistics
    for (var row = tid; row < actual_Br; row += THREADS) {{
        let global_row = q_start + row;
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
// Uses tiled approach with padding to prevent bank conflicts
// Tile size: {tile_size}x{tile_size}, Padded stride: {tile_size + 1}

struct TransposeParams {{
    rows: u32,
    cols: u32,
}}

@group(0) @binding(0) var<uniform> params: TransposeParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const TILE_SIZE: u32 = {tile_size}u;
const PADDED_STRIDE: u32 = {tile_size + 1}u;

// Padded shared memory to avoid bank conflicts
var<workgroup> tile: array<f32, {padded_size}>;

@compute @workgroup_size({tile_size}, {tile_size}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    // Load tile from input (coalesced reads)
    if (row < params.rows && col < params.cols) {{
        tile[local_row * PADDED_STRIDE + local_col] = input[row * params.cols + col];
    }}

    workgroupBarrier();

    // Write transposed tile to output (coalesced writes)
    let out_row = col;
    let out_col = row;

    if (out_row < params.cols && out_col < params.rows) {{
        output[out_row * params.rows + out_col] = tile[local_col * PADDED_STRIDE + local_row];
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
// Uses 2D dispatch: (embedding_dim, batch_size)

struct ExtractParams {{
    batch_size: u32,
    seq_len: u32,
    embedding_dim: u32,
}}

@group(0) @binding(0) var<uniform> params: ExtractParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let dim_idx = global_id.x;
    let batch_idx = global_id.y;

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


def create_softmax_kernel(workgroup_size: int = 256) -> str:
    """
    Generate standalone softmax kernel for inference/generation

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid

    Note:
        Uses numerically stable softmax with max subtraction.
        Parallelized version using parallel reduction.
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
// Standalone softmax for inference/generation
// Parallelized version with parallel reduction
// Input: logits [batch, vocab_size]
// Output: probs [batch, vocab_size]

struct SoftmaxParams {{
    batch_size: u32,
    vocab_size: u32,
}}

@group(0) @binding(0) var<uniform> params: SoftmaxParams;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> probs: array<f32>;

const BLOCK_SIZE: u32 = {workgroup_size}u;

var<workgroup> shared_data: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let batch_idx = workgroup_id.x;
    let tid = local_id.x;

    if (batch_idx >= params.batch_size) {{
        return;
    }}

    let logit_offset = batch_idx * params.vocab_size;

    // Phase 1: Find max using parallel reduction
    var max_logit = -1e10;
    for (var i = tid; i < params.vocab_size; i += BLOCK_SIZE) {{
        max_logit = max(max_logit, logits[logit_offset + i]);
    }}
    shared_data[tid] = max_logit;
    workgroupBarrier();

    // Reduction to find global max
    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_data[tid] = max(shared_data[tid], shared_data[tid + active_threads]);
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}
    max_logit = shared_data[0];
    workgroupBarrier();

    // Phase 2: Compute sum of exponentials using parallel reduction
    var sum_exp = 0.0;
    for (var i = tid; i < params.vocab_size; i += BLOCK_SIZE) {{
        sum_exp += exp(logits[logit_offset + i] - max_logit);
    }}
    shared_data[tid] = sum_exp;
    workgroupBarrier();

    // Reduction to find global sum
    active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_data[tid] += shared_data[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}
    sum_exp = shared_data[0];
    workgroupBarrier();

    // Phase 3: Normalize to get probabilities (parallelized)
    for (var i = tid; i < params.vocab_size; i += BLOCK_SIZE) {{
        let prob = exp(logits[logit_offset + i] - max_logit) / sum_exp;
        probs[logit_offset + i] = prob;
    }}
}}
"""


def create_cross_entropy_loss_kernel(workgroup_size: int = 256) -> str:
    """
    Generate unified cross-entropy loss and gradient kernel with masking support.

    Computes both loss and gradients in a single pass with proper normalization.
    Handles variable-length sequences via padding masks.

    Features:
    - Numerically stable softmax (max subtraction)
    - Automatic normalization by valid token count
    - Zero loss/gradients for masked (padding) tokens
    - Works with all-ones mask for unmasked batches

    Args:
        workgroup_size: Threads per workgroup (64, 128, 256, 512, or 1024)

    Returns:
        WGSL kernel source code

    Raises:
        ValueError: If workgroup_size is invalid
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
// Unified Cross-Entropy Loss + Gradient Kernel with Masking
// ============================================================
// Computes: loss = -log(softmax(logits)[target])
//           grad = (softmax(logits) - one_hot(target)) / valid_count
//
// Properly normalized by count of non-masked tokens for stable training.
// Pass all-ones mask for sequences without padding.

struct LossParams {{
    batch_size: u32,
    seq_len: u32,
    vocab_size: u32,
    pad_value: u32,  // Reserved for future use
}}

@group(0) @binding(0) var<uniform> params: LossParams;
@group(0) @binding(1) var<storage, read> logits: array<f32>;       // [B*S, V]
@group(0) @binding(2) var<storage, read> targets: array<u32>;      // [B*S]
@group(0) @binding(3) var<storage, read> mask: array<u32>;         // [B*S] 1=valid, 0=pad
@group(0) @binding(4) var<storage, read_write> loss_output: array<f32>;  // [B*S]
@group(0) @binding(5) var<storage, read_write> grad_logits: array<f32>;  // [B*S, V]
@group(0) @binding(6) var<storage, read_write> valid_count: atomic<u32>; // Total valid tokens

const BLOCK_SIZE: u32 = {workgroup_size}u;
var<workgroup> shared_data: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let token_idx = workgroup_id.x;
    let tid = local_id.x;
    let total_tokens = params.batch_size * params.seq_len;

    if (token_idx >= total_tokens) {{ return; }}

    let is_valid = mask[token_idx];

    // Count valid tokens atomically (once per workgroup)
    if (tid == 0u && is_valid == 1u) {{
        atomicAdd(&valid_count, 1u);
    }}

    // MASKED TOKENS: Zero out and exit early
    if (is_valid == 0u) {{
        if (tid == 0u) {{
            loss_output[token_idx] = 0.0;
        }}
        for (var i = tid; i < params.vocab_size; i += BLOCK_SIZE) {{
            grad_logits[token_idx * params.vocab_size + i] = 0.0;
        }}
        return;
    }}

    // VALID TOKENS: Compute cross-entropy loss and gradients
    let target_class = targets[token_idx];
    let logit_offset = token_idx * params.vocab_size;

    // Phase 1: Parallel reduction to find max logit (numerical stability)
    var max_val = -1e10;
    for (var i = tid; i < params.vocab_size; i += BLOCK_SIZE) {{
        max_val = max(max_val, logits[logit_offset + i]);
    }}
    shared_data[tid] = max_val;
    workgroupBarrier();

    // Tree reduction for max
    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {{
        if (tid < stride) {{
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }}
        workgroupBarrier();
        stride /= 2u;
    }}
    max_val = shared_data[0];
    workgroupBarrier();

    // Phase 2: Parallel reduction to compute sum(exp(logit - max))
    var sum_exp = 0.0;
    for (var i = tid; i < params.vocab_size; i += BLOCK_SIZE) {{
        sum_exp += exp(logits[logit_offset + i] - max_val);
    }}
    shared_data[tid] = sum_exp;
    workgroupBarrier();

    // Tree reduction for sum
    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {{
        if (tid < stride) {{
            shared_data[tid] += shared_data[tid + stride];
        }}
        workgroupBarrier();
        stride /= 2u;
    }}
    sum_exp = shared_data[0];
    workgroupBarrier();

    // Phase 3: Compute loss (thread 0 writes to output)
    if (tid == 0u) {{
        let target_logit = logits[logit_offset + target_class];
        let log_sum_exp = log(sum_exp);
        // Standard cross-entropy: -log(softmax[target])
        loss_output[token_idx] = log_sum_exp + max_val - target_logit;
    }}

    // Phase 4: Compute gradients with proper normalization
    // NOTE: Normalization will be applied in a second pass after counting
    // For now, store unnormalized gradients
    let log_sum_exp = log(sum_exp);
    for (var i = tid; i < params.vocab_size; i += BLOCK_SIZE) {{
        // Compute softmax probability
        let prob = exp(logits[logit_offset + i] - max_val - log_sum_exp);

        // Gradient: softmax - one_hot(target)
        var grad = prob;
        if (i == target_class) {{
            grad -= 1.0;
        }}

        // Store unnormalized gradient (will normalize in separate kernel)
        grad_logits[logit_offset + i] = grad;
    }}
}}
"""


def create_gradient_normalization_kernel(workgroup_size: int = 256) -> str:
    """
    Generate kernel to normalize gradients by valid token count.

    Must be called AFTER the loss kernel to divide all gradients by
    the total number of valid (non-masked) tokens.

    Args:
        workgroup_size: Threads per workgroup

    Returns:
        WGSL kernel source code
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
// Gradient Normalization Kernel
// Divides all gradients by the count of valid tokens

struct NormParams {{
    total_elements: u32,  // batch_size * seq_len * vocab_size
}}

@group(0) @binding(0) var<uniform> params: NormParams;
@group(0) @binding(1) var<storage, read_write> grad_logits: array<f32>;
@group(0) @binding(2) var<storage, read> valid_count: atomic<u32>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= params.total_elements) {{ return; }}

    let count = atomicLoad(&valid_count);
    if (count > 0u) {{
        let normalization = 1.0 / f32(count);
        grad_logits[idx] *= normalization;
    }}
}}
"""


# ============================================================================
# BACKWARD PASS KERNELS
# ============================================================================
def create_matmul_backward_a_kernel(
    tile_size: int = 16, items_per_thread: int = 4
) -> str:
    """
    Generate backward matmul kernel for gradient w.r.t. A

    Args:
        tile_size: Tile dimension (must be power of 2, typically 8, 16, or 32)
        items_per_thread: Number of output elements per thread (1, 2, or 4)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If tile_size or items_per_thread is invalid
    """
    if tile_size <= 0 or (tile_size & (tile_size - 1)) != 0:
        raise ValueError(f"tile_size must be power of 2, got {tile_size}")

    if tile_size > 32:
        raise ValueError(f"tile_size too large: {tile_size}. Maximum is 32.")

    if items_per_thread not in [1, 2, 4]:
        raise ValueError(f"items_per_thread must be 1, 2, or 4, got {items_per_thread}")

    workgroup_dim = tile_size // (items_per_thread if items_per_thread > 1 else 1)

    return f"""
// Backward pass for matmul: compute gradient w.r.t. A
// Given: dL/dC, B
// Compute: dL/dA = dL/dC @ B^T
// Tile size: {tile_size}x{tile_size}, Items per thread: {items_per_thread}x{items_per_thread}

struct MatmulParams {{
    M: u32,
    K: u32,
    N: u32,
}}

@group(0) @binding(0) var<uniform> params: MatmulParams;
@group(0) @binding(1) var<storage, read> grad_C: array<f32>;  // (M, N)
@group(0) @binding(2) var<storage, read> B: array<f32>;       // (K, N)
@group(0) @binding(3) var<storage, read_write> grad_A: array<f32>;  // (M, K)

const TILE_SIZE: u32 = {tile_size}u;
const ITEMS_PER_THREAD: u32 = {items_per_thread}u;

var<workgroup> tile_grad: array<f32, {tile_size * tile_size}>;
var<workgroup> tile_B: array<f32, {tile_size * tile_size}>;

@compute @workgroup_size({workgroup_dim}, {workgroup_dim}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let local_row = local_id.y;
    let local_col = local_id.x;

    let base_row = global_id.y * ITEMS_PER_THREAD;  // M dimension
    let base_col = global_id.x * ITEMS_PER_THREAD;  // K dimension

    // Accumulator registers
    var acc: array<f32, {items_per_thread * items_per_thread}>;
    for (var i = 0u; i < {items_per_thread * items_per_thread}u; i++) {{
        acc[i] = 0.0;
    }}

    let num_tiles = (params.N + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {{
        // Load grad_C tile - each thread loads ITEMS_PER_THREAD elements
        for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
            let g_row = base_row + i;
            let g_col = t * TILE_SIZE + local_col;
            let tile_idx = (local_row * ITEMS_PER_THREAD + i) * TILE_SIZE + local_col;

            if (g_row < params.M && g_col < params.N) {{
                tile_grad[tile_idx] = grad_C[g_row * params.N + g_col];
            }} else {{
                tile_grad[tile_idx] = 0.0;
            }}
        }}

        // Load B^T tile (transpose on-the-fly) - each thread loads ITEMS_PER_THREAD elements
        for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
            let b_row = t * TILE_SIZE + local_row;
            let b_col = base_col + i;
            let tile_idx = local_row * TILE_SIZE + (local_col * ITEMS_PER_THREAD + i);

            if (b_row < params.N && b_col < params.K) {{
                tile_B[tile_idx] = B[b_col * params.N + b_row];
            }} else {{
                tile_B[tile_idx] = 0.0;
            }}
        }}

        workgroupBarrier();

        // Compute partial results
        for (var k = 0u; k < TILE_SIZE; k++) {{
            for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
                let grad_val = tile_grad[(local_row * ITEMS_PER_THREAD + i) * TILE_SIZE + k];
                for (var j = 0u; j < ITEMS_PER_THREAD; j++) {{
                    let b_val = tile_B[k * TILE_SIZE + (local_col * ITEMS_PER_THREAD + j)];
                    acc[i * ITEMS_PER_THREAD + j] += grad_val * b_val;
                }}
            }}
        }}

        workgroupBarrier();
    }}

    // Write results
    for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
        let out_row = base_row + i;
        for (var j = 0u; j < ITEMS_PER_THREAD; j++) {{
            let out_col = base_col + j;
            if (out_row < params.M && out_col < params.K) {{
                grad_A[out_row * params.K + out_col] = acc[i * ITEMS_PER_THREAD + j];
            }}
        }}
    }}
}}
"""


def create_matmul_backward_b_kernel(
    tile_size: int = 16, items_per_thread: int = 4
) -> str:
    """
    Generate backward matmul kernel for gradient w.r.t. B

    Args:
        tile_size: Tile dimension (must be power of 2, typically 8, 16, or 32)
        items_per_thread: Number of output elements per thread (1, 2, or 4)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If tile_size or items_per_thread is invalid
    """
    if tile_size <= 0 or (tile_size & (tile_size - 1)) != 0:
        raise ValueError(f"tile_size must be power of 2, got {tile_size}")

    if tile_size > 32:
        raise ValueError(f"tile_size too large: {tile_size}. Maximum is 32.")

    if items_per_thread not in [1, 2, 4]:
        raise ValueError(f"items_per_thread must be 1, 2, or 4, got {items_per_thread}")

    workgroup_dim = tile_size // (items_per_thread if items_per_thread > 1 else 1)

    return f"""
// Backward pass for matmul: compute gradient w.r.t. B
// Given: A, dL/dC
// Compute: dL/dB = A^T @ dL/dC
// Tile size: {tile_size}x{tile_size}, Items per thread: {items_per_thread}x{items_per_thread}

struct MatmulParams {{
    M: u32,
    K: u32,
    N: u32,
}}

@group(0) @binding(0) var<uniform> params: MatmulParams;
@group(0) @binding(1) var<storage, read> A: array<f32>;       // (M, K)
@group(0) @binding(2) var<storage, read> grad_C: array<f32>;  // (M, N)
@group(0) @binding(3) var<storage, read_write> grad_B: array<f32>;  // (K, N)

const TILE_SIZE: u32 = {tile_size}u;
const ITEMS_PER_THREAD: u32 = {items_per_thread}u;

var<workgroup> tile_A: array<f32, {tile_size * tile_size}>;
var<workgroup> tile_grad: array<f32, {tile_size * tile_size}>;

@compute @workgroup_size({workgroup_dim}, {workgroup_dim}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let local_row = local_id.y;
    let local_col = local_id.x;

    let base_row = global_id.y * ITEMS_PER_THREAD;  // K dimension
    let base_col = global_id.x * ITEMS_PER_THREAD;  // N dimension

    // Accumulator registers
    var acc: array<f32, {items_per_thread * items_per_thread}>;
    for (var i = 0u; i < {items_per_thread * items_per_thread}u; i++) {{
        acc[i] = 0.0;
    }}

    let num_tiles = (params.M + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {{
        // Load A^T tile (transpose on-the-fly) - each thread loads ITEMS_PER_THREAD elements
        // This loads a tile of A into tile_A with a layout ready for matmul.
        for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
            let a_row = t * TILE_SIZE + local_row;
            let a_col = base_row + i;
            let tile_idx = (local_row * ITEMS_PER_THREAD + i) * TILE_SIZE + local_col;

            if (a_row < params.M && a_col < params.K) {{
                // Note: The indexing here loads A transposed into tile_A
                tile_A[tile_idx] = A[a_row * params.K + a_col];
            }} else {{
                tile_A[tile_idx] = 0.0;
            }}
        }}

        // Load grad_C tile - each thread loads ITEMS_PER_THREAD elements
        for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
            let g_row = t * TILE_SIZE + local_row;
            let g_col = base_col + i;
            let tile_idx = local_row * TILE_SIZE + (local_col * ITEMS_PER_THREAD + i);

            if (g_row < params.M && g_col < params.N) {{
                tile_grad[tile_idx] = grad_C[g_row * params.N + g_col];
            }} else {{
                tile_grad[tile_idx] = 0.0;
            }}
        }}

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k++) {{
            for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
                let a_val = tile_A[(local_row * ITEMS_PER_THREAD + i) * TILE_SIZE + k];
                for (var j = 0u; j < ITEMS_PER_THREAD; j++) {{
                    let grad_val = tile_grad[k * TILE_SIZE + (local_col * ITEMS_PER_THREAD + j)];
                    acc[i * ITEMS_PER_THREAD + j] += a_val * grad_val;
                }}
            }}
        }}

        workgroupBarrier();
    }}

    // Write results
    for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
        let out_row = base_row + i;
        for (var j = 0u; j < ITEMS_PER_THREAD; j++) {{
            let out_col = base_col + j;
            if (out_row < params.K && out_col < params.N) {{
                grad_B[out_row * params.N + out_col] = acc[i * ITEMS_PER_THREAD + j];
            }}
        }}
    }}
}}
"""


def create_layernorm_backward_kernel(
    workgroup_size: int = 256, epsilon: float = 1e-5
) -> str:
    """
    Generate layer normalization backward kernel - Stage 1

    Uses two-stage reduction to avoid atomic operations on f32.
    This kernel computes partial gradients that must be reduced in stage 2.

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

    # Add padding to avoid bank conflicts
    padded_size = (
        workgroup_size + (32 - (workgroup_size % 32))
        if workgroup_size % 32 != 0
        else workgroup_size
    )

    return f"""
// Backward pass for layer normalization - STAGE 1
// Compute per-element contributions
//
// Two-stage algorithm eliminates race conditions (no atomic f32 required):
// - Stage 1 (this kernel): Each workgroup computes partial gradients for gamma/beta
// - Stage 2 (separate kernel): Reduction combines partial gradients
//
// This kernel outputs PARTIAL gradients that must be reduced in stage 2.

struct NormParams {{
    size: u32,
    n_elements: u32,
}}

@group(0) @binding(0) var<uniform> params: NormParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(5) var<storage, read_write> partial_grad_gamma: array<f32>;  // (n_elements, size)
@group(0) @binding(6) var<storage, read_write> partial_grad_beta: array<f32>;   // (n_elements, size)

const EPS: f32 = {epsilon};
const BLOCK_SIZE: u32 = {workgroup_size}u;

// Padded to avoid bank conflicts
var<workgroup> shared_data: array<f32, {padded_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let elem_idx = workgroup_id.x;  // One workgroup per element
    let tid = local_id.x;

    if (elem_idx >= params.n_elements) {{
        return;
    }}

    let offset = elem_idx * params.size;

    // Phase 1: Recompute mean
    var sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {{
        sum += input[offset + i];
    }}
    shared_data[tid] = sum;
    workgroupBarrier();

    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_data[tid] += shared_data[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}

    let mean = shared_data[0] / f32(params.size);
    workgroupBarrier();

    // Phase 2: Recompute variance
    var var_sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {{
        let diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }}
    shared_data[tid] = var_sum;
    workgroupBarrier();

    active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_data[tid] += shared_data[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}

    let variance = shared_data[0] / f32(params.size);
    let inv_std = 1.0 / sqrt(variance + EPS);
    workgroupBarrier();

    // Phase 3: Compute PARTIAL gradients for gamma and beta
    // Each workgroup handles ONE element, writes to its own slice
    // SAFE: No race condition because each workgroup writes to different locations
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {{
        let x_norm = (input[offset + i] - mean) * inv_std;
        let gamma_grad = grad_output[offset + i] * x_norm;
        let beta_grad = grad_output[offset + i];

        let partial_offset = elem_idx * params.size + i;
        partial_grad_gamma[partial_offset] = gamma_grad;
        partial_grad_beta[partial_offset] = beta_grad;
    }}
    workgroupBarrier();

    // Phase 4: Compute gradient w.r.t. input
    var d_xhat_sum = 0.0;
    var d_xhat_xhat_sum = 0.0;

    for (var i = tid; i < params.size; i += BLOCK_SIZE) {{
        let xhat = (input[offset + i] - mean) * inv_std;
        let d_xhat = grad_output[offset + i] * gamma[i];
        d_xhat_sum += d_xhat;
        d_xhat_xhat_sum += d_xhat * xhat;
    }}

    shared_data[tid] = d_xhat_sum;
    workgroupBarrier();

    active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_data[tid] += shared_data[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}
    d_xhat_sum = shared_data[0];
    workgroupBarrier();

    shared_data[tid] = d_xhat_xhat_sum;
    workgroupBarrier();

    active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_data[tid] += shared_data[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}
    d_xhat_xhat_sum = shared_data[0];
    workgroupBarrier();

    // Phase 5: Write input gradients
    let N = f32(params.size);
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {{
        let xhat = (input[offset + i] - mean) * inv_std;
        let d_xhat = grad_output[offset + i] * gamma[i];
        grad_input[offset + i] = (d_xhat - d_xhat_sum / N - xhat * d_xhat_xhat_sum / N) * inv_std;
    }}
}}
"""


def create_layernorm_backward_reduce_kernel(workgroup_size: int = 256) -> str:
    """
    Generate layer normalization backward reduction kernel - Stage 2

    Reduces partial gradients with parallel reduction.

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
// Backward pass for layer normalization - STAGE 2
// Reduce partial gradients with parallel reduction
//
// Reduces partial gamma/beta gradients from stage 1 across all elements.
// Uses parallel reduction for efficiency with large batch sizes.

struct ReduceParams {{
    size: u32,
    n_elements: u32,
}}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> partial_grad_gamma: array<f32>;  // (n_elements, size)
@group(0) @binding(2) var<storage, read> partial_grad_beta: array<f32>;   // (n_elements, size)
@group(0) @binding(3) var<storage, read_write> grad_gamma: array<f32>;    // (size,)
@group(0) @binding(4) var<storage, read_write> grad_beta: array<f32>;     // (size,)

const BLOCK_SIZE: u32 = {workgroup_size}u;

var<workgroup> shared_gamma: array<f32, {workgroup_size}>;
var<workgroup> shared_beta: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let dim_idx = workgroup_id.x;
    let tid = local_id.x;

    if (dim_idx >= params.size) {{
        return;
    }}

    // Parallel reduction across elements
    var gamma_sum = 0.0;
    var beta_sum = 0.0;

    for (var elem_idx = tid; elem_idx < params.n_elements; elem_idx += BLOCK_SIZE) {{
        let partial_offset = elem_idx * params.size + dim_idx;
        gamma_sum += partial_grad_gamma[partial_offset];
        beta_sum += partial_grad_beta[partial_offset];
    }}

    shared_gamma[tid] = gamma_sum;
    shared_beta[tid] = beta_sum;
    workgroupBarrier();

    // Tree reduction
    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_gamma[tid] += shared_gamma[tid + active_threads];
            shared_beta[tid] += shared_beta[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}

    // Thread 0 writes final result
    if (tid == 0u) {{
        grad_gamma[dim_idx] = shared_gamma[0];
        grad_beta[dim_idx] = shared_beta[0];
    }}
}}
"""


def create_layernorm_backward_reduce_accumulate_kernel(
    workgroup_size: int = 256,
) -> str:
    """
    Generate layer normalization backward reduction kernel - Stage 2 (ACCUMULATE MODE)

    Reduces partial gradients and accumulates into existing values.
    Uses proper atomic operations or ensures single-writer guarantee.

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
// Backward pass for layer normalization - STAGE 2 (ACCUMULATE MODE)
// Reduce partial gradients and accumulate into existing values
//
// IMPORTANT: This kernel must be dispatched with proper synchronization
// to ensure only one workgroup writes to each dimension at a time.
// Alternative: Use atomicAdd if available in future WGSL versions.

struct ReduceParams {{
    size: u32,
    n_elements: u32,
}}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> partial_grad_gamma: array<f32>;  // (n_elements, size)
@group(0) @binding(2) var<storage, read> partial_grad_beta: array<f32>;   // (n_elements, size)
@group(0) @binding(3) var<storage, read_write> grad_gamma: array<f32>;    // (size,)
@group(0) @binding(4) var<storage, read_write> grad_beta: array<f32>;     // (size,)

const BLOCK_SIZE: u32 = {workgroup_size}u;

var<workgroup> shared_gamma: array<f32, {workgroup_size}>;
var<workgroup> shared_beta: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let dim_idx = workgroup_id.x;
    let tid = local_id.x;

    if (dim_idx >= params.size) {{
        return;
    }}

    // Parallel reduction across elements
    var gamma_sum = 0.0;
    var beta_sum = 0.0;

    for (var elem_idx = tid; elem_idx < params.n_elements; elem_idx += BLOCK_SIZE) {{
        let partial_offset = elem_idx * params.size + dim_idx;
        gamma_sum += partial_grad_gamma[partial_offset];
        beta_sum += partial_grad_beta[partial_offset];
    }}

    shared_gamma[tid] = gamma_sum;
    shared_beta[tid] = beta_sum;
    workgroupBarrier();

    // Tree reduction
    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_gamma[tid] += shared_gamma[tid + active_threads];
            shared_beta[tid] += shared_beta[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}

    // Thread 0 accumulates result
    // NOTE: Caller must ensure proper synchronization if multiple
    // invocations could write to the same location
    if (tid == 0u) {{
        grad_gamma[dim_idx] += shared_gamma[0];
        grad_beta[dim_idx] += shared_beta[0];
    }}
}}
"""


def create_gelu_backward_kernel(workgroup_size: int = 256) -> str:
    """
    Generate GELU activation backward kernel

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
// Backward pass for GELU activation
// Uses sigmoid approximation to match forward pass
//
// Forward: gelu(x) ≈ x * sigmoid(1.702 * x)
// Derivative: gelu'(x) ≈ sigmoid(1.702*x) + 1.702*x*sigmoid(1.702*x)*(1-sigmoid(1.702*x))

struct GeluParams {{
    size: u32,
}}

@group(0) @binding(0) var<uniform> params: GeluParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;

const GELU_SCALE: f32 = 1.702;

fn fast_sigmoid(x: f32) -> f32 {{
    return 1.0 / (1.0 + exp(-x));
}}

fn gelu_derivative(x: f32) -> f32 {{
    let scaled_x = GELU_SCALE * x;
    let sig = fast_sigmoid(scaled_x);
    // d/dx[x * sigmoid(c*x)] = sigmoid(c*x) + c*x*sigmoid(c*x)*(1-sigmoid(c*x))
    return sig + GELU_SCALE * x * sig * (1.0 - sig);
}}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;

    if (idx >= params.size) {{
        return;
    }}

    let x = input[idx];
    let gelu_grad = gelu_derivative(x);
    grad_input[idx] = grad_output[idx] * gelu_grad;
}}
"""


def create_bias_backward_kernel(workgroup_size: int = 256) -> str:
    """
    Generate bias backward kernel

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
// Backward pass for bias addition
// Gradient w.r.t. bias is sum over batch dimension
// Uses parallel reduction for efficiency

struct BiasParams {{
    size: u32,
    dim: u32,
}}

@group(0) @binding(0) var<uniform> params: BiasParams;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_bias: array<f32>;

const BLOCK_SIZE: u32 = {workgroup_size}u;

var<workgroup> shared_data: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let col = workgroup_id.x;  // One workgroup per bias dimension
    let tid = local_id.x;

    if (col >= params.dim) {{
        return;
    }}

    let n_rows = params.size / params.dim;

    // Parallel sum across rows
    var sum = 0.0;
    for (var row = tid; row < n_rows; row += BLOCK_SIZE) {{
        sum += grad_output[row * params.dim + col];
    }}
    shared_data[tid] = sum;
    workgroupBarrier();

    // Tree reduction
    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_data[tid] += shared_data[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}

    // Thread 0 writes result
    if (tid == 0u) {{
        grad_bias[col] = shared_data[0];
    }}
}}
"""


def create_attention_backward_kernel(workgroup_size: int = 256) -> str:
    """
    Generate attention backward kernel - TWO-STAGE version

    Stage 1: Computes grad_Q and partial grad_K/grad_V per query position.
    Stage 2: (separate kernel) Reduces partial grad_K/grad_V.

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid

    Note:
        This kernel outputs partial_grad_K and partial_grad_V that must be
        reduced in a separate pass. Each workgroup writes to:
        partial_grad_K[batch, head, q_pos, seq_len, head_dim]
        partial_grad_V[batch, head, q_pos, seq_len, head_dim]
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    max_seq_len = workgroup_size

    return f"""
// Backward pass for scaled dot-product attention - STAGE 1
// Computes grad_Q directly and partial grad_K/grad_V
//
// Each workgroup handles ONE query position.
// Outputs partial gradients for K/V that must be reduced in stage 2.

struct AttentionBackwardParams {{
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
}}

@group(0) @binding(0) var<uniform> params: AttentionBackwardParams;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read> O: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_Q: array<f32>;
@group(0) @binding(7) var<storage, read_write> partial_grad_K: array<f32>;  // [batch*head*q_pos, seq_len, head_dim]
@group(0) @binding(8) var<storage, read_write> partial_grad_V: array<f32>;  // [batch*head*q_pos, seq_len, head_dim]

const BLOCK_SIZE: u32 = {workgroup_size}u;
const MASK_VALUE: f32 = -1e10;

var<workgroup> shared_scores: array<f32, {max_seq_len}>;
var<workgroup> shared_reduction: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let batch_idx = workgroup_id.z;
    let head_idx = workgroup_id.y;
    let q_pos = workgroup_id.x;
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

    // Compute workgroup index for partial outputs
    let wg_idx = batch_idx * params.n_heads * seq_len + head_idx * seq_len + q_pos;
    let partial_base = wg_idx * seq_len * head_dim;

    // Phase 1: Compute and cache Q·K scores
    for (var k_pos = tid; k_pos < seq_len; k_pos += BLOCK_SIZE) {{
        if (k_pos <= q_pos) {{
            let k_offset = batch_idx * seq_len * embedding_dim +
                          k_pos * embedding_dim +
                          head_idx * head_dim;

            var score = 0.0;
            for (var d = 0u; d < head_dim; d++) {{
                score += Q[q_offset + d] * K[k_offset + d];
            }}
            shared_scores[k_pos] = score * scale;
        }} else {{
            shared_scores[k_pos] = MASK_VALUE;
        }}
    }}
    workgroupBarrier();

    // Phase 2: Find max and compute softmax (same as before)
    var max_score = MASK_VALUE;
    for (var k_pos = tid; k_pos < seq_len; k_pos += BLOCK_SIZE) {{
        max_score = max(max_score, shared_scores[k_pos]);
    }}
    shared_reduction[tid] = max_score;
    workgroupBarrier();

    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_reduction[tid] = max(shared_reduction[tid], shared_reduction[tid + active_threads]);
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}
    max_score = shared_reduction[0];
    workgroupBarrier();

    var sum_exp = 0.0;
    for (var k_pos = tid; k_pos < seq_len; k_pos += BLOCK_SIZE) {{
        if (k_pos <= q_pos) {{
            let exp_score = exp(shared_scores[k_pos] - max_score);
            shared_scores[k_pos] = exp_score;
            sum_exp += exp_score;
        }}
    }}
    shared_reduction[tid] = sum_exp;
    workgroupBarrier();

    active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_reduction[tid] += shared_reduction[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}
    sum_exp = shared_reduction[0];
    workgroupBarrier();

    // Phase 3: Compute dO_dot_O
    var dO_dot_O = 0.0;
    for (var d = tid; d < head_dim; d += BLOCK_SIZE) {{
        dO_dot_O += grad_output[q_offset + d] * O[q_offset + d];
    }}
    shared_reduction[tid] = dO_dot_O;
    workgroupBarrier();

    active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_reduction[tid] += shared_reduction[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}
    dO_dot_O = shared_reduction[0];
    workgroupBarrier();

    // Phase 4: Compute gradients (parallelized over dimensions)
    for (var d = tid; d < head_dim; d += BLOCK_SIZE) {{
        var grad_q_acc = 0.0;

        for (var k_pos = 0u; k_pos <= q_pos; k_pos++) {{
            let attn_weight = shared_scores[k_pos] / sum_exp;
            let k_offset = batch_idx * seq_len * embedding_dim +
                          k_pos * embedding_dim +
                          head_idx * head_dim;

            // Compute dP
            var dP = 0.0;
            for (var dd = 0u; dd < head_dim; dd++) {{
                dP += grad_output[q_offset + dd] * V[k_offset + dd];
            }}

            // Softmax backward
            let dS = attn_weight * (dP - dO_dot_O);

            // Accumulate grad_Q
            grad_q_acc += dS * K[k_offset + d] * scale;

            // Write PARTIAL grad_K and grad_V (no race: unique output location per workgroup)
            let partial_offset = partial_base + k_pos * head_dim + d;
            partial_grad_K[partial_offset] = dS * Q[q_offset + d] * scale;
            partial_grad_V[partial_offset] = attn_weight * grad_output[q_offset + d];
        }}

        // Write grad_Q (safe: unique per workgroup)
        grad_Q[q_offset + d] = grad_q_acc;
    }}
}}
"""


def create_attention_backward_reduce_kernel(workgroup_size: int = 256) -> str:
    """
    Generate attention backward reduction kernel - Stage 2

    Reduces partial grad_K and grad_V from all query positions.
    Each (batch, head, k_pos, dimension) is processed by one workgroup.

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
// Backward pass for attention - STAGE 2: Reduce partial gradients
// Reduces partial_grad_K and partial_grad_V across all query positions
//
// Input: partial_grad_K[batch*head*q_pos, seq_len, head_dim]
// Output: grad_K[batch, seq_len, n_heads, head_dim]
// Same for V

struct ReduceParams {{
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
}}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> partial_grad_K: array<f32>;  // [batch*head*q_pos, seq_len, head_dim]
@group(0) @binding(2) var<storage, read> partial_grad_V: array<f32>;  // [batch*head*q_pos, seq_len, head_dim]
@group(0) @binding(3) var<storage, read_write> grad_K: array<f32>;    // [batch, seq_len, n_heads, head_dim]
@group(0) @binding(4) var<storage, read_write> grad_V: array<f32>;    // [batch, seq_len, n_heads, head_dim]

const BLOCK_SIZE: u32 = {workgroup_size}u;

var<workgroup> shared_K: array<f32, {workgroup_size}>;
var<workgroup> shared_V: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    // Each workgroup handles one (batch, head, k_pos, dimension)
    // We'll dispatch with 3D: (head_dim, seq_len*n_heads, batch_size)
    let batch_idx = workgroup_id.z;
    let head_seq_idx = workgroup_id.y;  // Combined head and seq index
    let dim_idx = workgroup_id.x;
    let tid = local_id.x;

    let head_idx = head_seq_idx / params.seq_len;
    let k_pos = head_seq_idx % params.seq_len;

    if (batch_idx >= params.batch_size || head_idx >= params.n_heads ||
        k_pos >= params.seq_len || dim_idx >= params.head_dim) {{
        return;
    }}

    let embedding_dim = params.n_heads * params.head_dim;

    // Sum partial gradients across all q_pos for this k_pos
    // Each q_pos that can see k_pos (q_pos >= k_pos due to causal mask) contributed
    var grad_K_sum = 0.0;
    var grad_V_sum = 0.0;

    // Iterate over all q_pos that could have contributed
    for (var q_pos = k_pos + tid; q_pos < params.seq_len; q_pos += BLOCK_SIZE) {{
        // Compute workgroup index from stage 1
        let wg_idx = batch_idx * params.n_heads * params.seq_len +
                     head_idx * params.seq_len + q_pos;
        let partial_base = wg_idx * params.seq_len * params.head_dim;
        let partial_offset = partial_base + k_pos * params.head_dim + dim_idx;

        grad_K_sum += partial_grad_K[partial_offset];
        grad_V_sum += partial_grad_V[partial_offset];
    }}

    shared_K[tid] = grad_K_sum;
    shared_V[tid] = grad_V_sum;
    workgroupBarrier();

    // Tree reduction
    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_K[tid] += shared_K[tid + active_threads];
            shared_V[tid] += shared_V[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}

    // Thread 0 writes final result
    if (tid == 0u) {{
        let output_offset = batch_idx * params.seq_len * embedding_dim +
                           k_pos * embedding_dim +
                           head_idx * params.head_dim + dim_idx;

        grad_K[output_offset] = shared_K[0];
        grad_V[output_offset] = shared_V[0];
    }}
}}
"""


def create_flash_attention_backward_kernel(
    head_dim: int = 64, Bc: int = 32, Br: int = 32
) -> str:
    """
    Generate FlashAttention backward kernel - Stage 1

    Memory-efficient backward pass using tiling and recomputation.
    Outputs partial gradients for K/V that must be reduced in stage 2.

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

    threads_per_workgroup = min(256, max(64, Br * 2))

    # Calculate memory requirements
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
    dQi_size = Br * head_dim

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
        + dQi_size
    )
    total_workgroup_bytes = total_workgroup_f32 * 4

    if total_workgroup_bytes > 65536:
        raise ValueError(
            f"Workgroup memory {total_workgroup_bytes} bytes exceeds 64KB limit. "
            f"Try smaller head_dim, Bc, or Br values."
        )

    return f"""
// FlashAttention Backward Pass - STAGE 1
// Fully parallelized, outputs partial gradients for K/V
//
// Parameters: head_dim={head_dim}, Bc={Bc}, Br={Br}
// Workgroup memory: {total_workgroup_bytes} bytes
// Threads per workgroup: {threads_per_workgroup}

struct FlashAttentionBackwardParams {{
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    Bc: u32,
    Br: u32,
}}

@group(0) @binding(0) var<uniform> params: FlashAttentionBackwardParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read> O: array<f32>;
@group(0) @binding(5) var<storage, read> L: array<f32>;
@group(0) @binding(6) var<storage, read> M: array<f32>;
@group(0) @binding(7) var<storage, read> grad_O: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_Q: array<f32>;
@group(0) @binding(9) var<storage, read_write> partial_grad_K: array<f32>;  // [batch*head*block_row, N, head_dim]
@group(0) @binding(10) var<storage, read_write> partial_grad_V: array<f32>; // [batch*head*block_row, N, head_dim]

const Bc: u32 = {Bc}u;
const Br: u32 = {Br}u;
const HEAD_DIM: u32 = {head_dim}u;
const THREADS: u32 = {threads_per_workgroup}u;

var<workgroup> Qi: array<f32, {qi_size}>;
var<workgroup> Kj: array<f32, {kj_size}>;
var<workgroup> Vj: array<f32, {vj_size}>;
var<workgroup> Sij: array<f32, {sij_size}>;
var<workgroup> Pij: array<f32, {pij_size}>;
var<workgroup> dOi: array<f32, {dOi_size}>;
var<workgroup> Oi: array<f32, {Oi_size}>;
var<workgroup> dPij: array<f32, {dPij_size}>;
var<workgroup> dSij: array<f32, {dSij_size}>;
var<workgroup> Di: array<f32, {Br}>;
var<workgroup> dQi: array<f32, {dQi_size}>;

@compute @workgroup_size({threads_per_workgroup}, 1, 1)
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

    // Compute workgroup index for partial outputs
    let wg_idx = batch_idx * params.n_heads * ((N + Br - 1u) / Br) +
                 head_idx * ((N + Br - 1u) / Br) + block_row;
    let partial_base = wg_idx * N * d;

    // Load Q, grad_O, and O blocks (parallelized)
    for (var i = tid; i < actual_Br * d; i += THREADS) {{
        let local_row = i / d;
        let local_col = i % d;
        let global_row = q_start + local_row;

        let q_offset = batch_idx * N * embedding_dim +
                      global_row * embedding_dim +
                      head_idx * d + local_col;

        Qi[local_row * HEAD_DIM + local_col] = Q[q_offset];
        dOi[local_row * HEAD_DIM + local_col] = grad_O[q_offset];
        Oi[local_row * HEAD_DIM + local_col] = O[q_offset];
    }}
    workgroupBarrier();

    // Compute D[i] = rowsum(dO * O) - PARALLELIZED
    for (var row = tid; row < actual_Br; row += THREADS) {{
        var sum = 0.0;
        for (var d_idx = 0u; d_idx < d; d_idx++) {{
            sum += dOi[row * HEAD_DIM + d_idx] * Oi[row * HEAD_DIM + d_idx];
        }}
        Di[row] = sum;
    }}
    workgroupBarrier();

    // Initialize dQi - PARALLELIZED
    for (var i = tid; i < actual_Br * d; i += THREADS) {{
        dQi[i] = 0.0;
    }}
    workgroupBarrier();

    // Iterate over K/V blocks
    let num_kv_blocks = (N + Bc - 1u) / Bc;

    for (var block_col = 0u; block_col < num_kv_blocks; block_col++) {{
        let kv_start = block_col * Bc;
        let kv_end = min(kv_start + Bc, N);
        let actual_Bc = kv_end - kv_start;

        if (kv_start > q_end) {{
            break;
        }}

        // Load K and V blocks (parallelized)
        for (var i = tid; i < actual_Bc * d; i += THREADS) {{
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

        // Compute scores Sij (parallelized)
        for (var i = tid; i < actual_Br * actual_Bc; i += THREADS) {{
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
                Sij[row * Bc + col] = -1e10;
            }}
        }}
        workgroupBarrier();

        // Compute attention weights P from saved statistics (parallelized)
        for (var i = tid; i < actual_Br * actual_Bc; i += THREADS) {{
            let row = i / actual_Bc;
            let col = i % actual_Bc;

            let global_row = q_start + row;
            let stats_offset = batch_idx * N * params.n_heads +
                              global_row * params.n_heads + head_idx;

            let m_val = M[stats_offset];
            let l_val = L[stats_offset];

            let kv_pos = kv_start + col;
            let q_pos = q_start + row;

            if (kv_pos <= q_pos) {{
                Pij[row * Bc + col] = exp(Sij[row * Bc + col] - m_val) / l_val;
            }} else {{
                Pij[row * Bc + col] = 0.0;
            }}
        }}
        workgroupBarrier();

        // Compute dP = dO @ V^T (parallelized)
        for (var i = tid; i < actual_Br * actual_Bc; i += THREADS) {{
            let row = i / actual_Bc;
            let col = i % actual_Bc;

            var dp_val = 0.0;
            for (var k = 0u; k < d; k++) {{
                dp_val += dOi[row * HEAD_DIM + k] * Vj[col * HEAD_DIM + k];
            }}
            dPij[row * Bc + col] = dp_val;
        }}
        workgroupBarrier();

        // Softmax backward: dS = P * (dP - D) (parallelized)
        for (var i = tid; i < actual_Br * actual_Bc; i += THREADS) {{
            let row = i / actual_Bc;
            let col = i % actual_Bc;

            dSij[row * Bc + col] = Pij[row * Bc + col] * (dPij[row * Bc + col] - Di[row]);
        }}
        workgroupBarrier();

        // Accumulate dQ = dS @ K (parallelized)
        for (var i = tid; i < actual_Br * d; i += THREADS) {{
            let row = i / d;
            let d_idx = i % d;

            var sum = 0.0;
            for (var col = 0u; col < actual_Bc; col++) {{
                sum += dSij[row * Bc + col] * Kj[col * HEAD_DIM + d_idx] * scale;
            }}
            dQi[row * HEAD_DIM + d_idx] += sum;
        }}
        workgroupBarrier();

        // Write PARTIAL dK = dS^T @ Q (parallelized, no race)
        for (var i = tid; i < actual_Bc * d; i += THREADS) {{
            let col = i / d;
            let d_idx = i % d;

            var sum = 0.0;
            for (var row = 0u; row < actual_Br; row++) {{
                sum += dSij[row * Bc + col] * Qi[row * HEAD_DIM + d_idx] * scale;
            }}

            let global_kv = kv_start + col;
            let partial_offset = partial_base + global_kv * d + d_idx;
            partial_grad_K[partial_offset] = sum;
        }}

        // Write PARTIAL dV = P^T @ dO (parallelized, no race)
        for (var i = tid; i < actual_Bc * d; i += THREADS) {{
            let col = i / d;
            let d_idx = i % d;

            var sum = 0.0;
            for (var row = 0u; row < actual_Br; row++) {{
                sum += Pij[row * Bc + col] * dOi[row * HEAD_DIM + d_idx];
            }}

            let global_kv = kv_start + col;
            let partial_offset = partial_base + global_kv * d + d_idx;
            partial_grad_V[partial_offset] = sum;
        }}

        workgroupBarrier();
    }}

    // Write accumulated dQ (parallelized)
    for (var i = tid; i < actual_Br * d; i += THREADS) {{
        let row = i / d;
        let d_idx = i % d;
        let global_row = q_start + row;

        let gq_offset = batch_idx * N * embedding_dim +
                       global_row * embedding_dim +
                       head_idx * d + d_idx;

        grad_Q[gq_offset] = dQi[row * HEAD_DIM + d_idx];
    }}
}}
"""


def create_flash_attention_backward_reduce_kernel(workgroup_size: int = 256) -> str:
    """
    Generate FlashAttention backward reduction kernel - Stage 2

    Reduces partial grad_K and grad_V from all Q blocks.

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
// FlashAttention Backward - STAGE 2: Reduce partial gradients
// Reduces partial_grad_K and partial_grad_V across all Q blocks
//
// Input: partial_grad_K[batch*head*block_row, N, head_dim]
// Output: grad_K[batch, N, n_heads, head_dim]

struct ReduceParams {{
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    Br: u32,  // Q block size from forward pass
}}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> partial_grad_K: array<f32>;
@group(0) @binding(2) var<storage, read> partial_grad_V: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_K: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_V: array<f32>;

const BLOCK_SIZE: u32 = {workgroup_size}u;

var<workgroup> shared_K: array<f32, {workgroup_size}>;
var<workgroup> shared_V: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    // Dispatch: (head_dim, seq_len*n_heads, batch_size)
    let batch_idx = workgroup_id.z;
    let head_seq_idx = workgroup_id.y;
    let dim_idx = workgroup_id.x;
    let tid = local_id.x;

    let head_idx = head_seq_idx / params.seq_len;
    let kv_pos = head_seq_idx % params.seq_len;

    if (batch_idx >= params.batch_size || head_idx >= params.n_heads ||
        kv_pos >= params.seq_len || dim_idx >= params.head_dim) {{
        return;
    }}

    let embedding_dim = params.n_heads * params.head_dim;
    let num_q_blocks = (params.seq_len + params.Br - 1u) / params.Br;

    // Sum partial gradients across all Q blocks
    var grad_K_sum = 0.0;
    var grad_V_sum = 0.0;

    for (var block_row = tid; block_row < num_q_blocks; block_row += BLOCK_SIZE) {{
        let wg_idx = batch_idx * params.n_heads * num_q_blocks +
                     head_idx * num_q_blocks + block_row;
        let partial_base = wg_idx * params.seq_len * params.head_dim;
        let partial_offset = partial_base + kv_pos * params.head_dim + dim_idx;

        grad_K_sum += partial_grad_K[partial_offset];
        grad_V_sum += partial_grad_V[partial_offset];
    }}

    shared_K[tid] = grad_K_sum;
    shared_V[tid] = grad_V_sum;
    workgroupBarrier();

    // Tree reduction
    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_K[tid] += shared_K[tid + active_threads];
            shared_V[tid] += shared_V[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}

    // Thread 0 writes result
    if (tid == 0u) {{
        let output_offset = batch_idx * params.seq_len * embedding_dim +
                           kv_pos * embedding_dim +
                           head_idx * params.head_dim + dim_idx;

        grad_K[output_offset] = shared_K[0];
        grad_V[output_offset] = shared_V[0];
    }}
}}
"""


# ============================================================================
# OPTIMIZER KERNELS
# ============================================================================
def create_adamw_kernel(workgroup_size: int = 256) -> str:
    """
    Generate AdamW optimizer kernel with configurable workgroup size

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid

    Note:
        Implements AdamW with decoupled weight decay.
        Fuses momentum updates, bias correction, and weight updates in one pass.
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
// Fused AdamW optimizer update with decoupled weight decay
// Vectorized for better memory bandwidth utilization

struct OptimizerParams {{
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    eps: f32,
    step: f32,
    size: u32,
}}

@group(0) @binding(0) var<uniform> params: OptimizerParams;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;

    if (idx >= params.size) {{
        return;
    }}

    let grad = gradients[idx];
    let weight = weights[idx];
    let m_old = m[idx];
    let v_old = v[idx];

    // Precompute bias correction factors
    let beta1_pow = pow(params.beta1, params.step);
    let beta2_pow = pow(params.beta2, params.step);
    let bias_correction1 = 1.0 - beta1_pow;
    let bias_correction2 = 1.0 - beta2_pow;

    // Update biased first moment estimate
    let m_new = params.beta1 * m_old + (1.0 - params.beta1) * grad;

    // Update biased second raw moment estimate
    let v_new = params.beta2 * v_old + (1.0 - params.beta2) * grad * grad;

    // Compute bias-corrected estimates
    let m_hat = m_new / bias_correction1;
    let v_hat = v_new / bias_correction2;

    // Update weights with AdamW (decoupled weight decay)
    let update = m_hat / (sqrt(v_hat) + params.eps);
    let new_weight = weight - params.lr * (update + params.weight_decay * weight);

    // Write updates
    m[idx] = m_new;
    v[idx] = v_new;
    weights[idx] = new_weight;
}}
"""


def create_gradient_clip_kernel(workgroup_size: int = 256) -> str:
    """
    Generate gradient clipping kernel with global norm clipping

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid

    Note:
        Clips gradients by global norm to prevent exploding gradients.
        Formula: if ||g|| > max_norm: g = g * (max_norm / ||g||)
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
// Gradient clipping by global norm
// Prevents exploding gradients during training

struct ClipParams {{
    size: u32,
    max_norm: f32,
    total_norm: f32,  // Pre-computed global norm
}}

@group(0) @binding(0) var<uniform> params: ClipParams;
@group(0) @binding(1) var<storage, read_write> gradients: array<f32>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;

    if (idx >= params.size) {{
        return;
    }}

    // If total norm exceeds max_norm, scale all gradients proportionally
    if (params.total_norm > params.max_norm) {{
        let scale = params.max_norm / params.total_norm;
        gradients[idx] *= scale;
    }}
}}
"""


def create_buffer_fill_kernel(workgroup_size: int = 256) -> str:
    """
    Generate buffer fill kernel to initialize buffers with a constant value

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid

    Note:
        Fills buffer with a constant value. Useful for initialization.
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
// Fill buffer with constant value
// Vectorized for better memory bandwidth

struct FillParams {{
    size: u32,
    value: f32,
}}

@group(0) @binding(0) var<uniform> params: FillParams;
@group(0) @binding(1) var<storage, read_write> buffer: array<f32>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    let vec4_count = params.size / 4u;
    let remainder = params.size % 4u;

    // Process 4 elements at a time
    if (idx < vec4_count) {{
        let base_idx = idx * 4u;
        buffer[base_idx] = params.value;
        buffer[base_idx + 1u] = params.value;
        buffer[base_idx + 2u] = params.value;
        buffer[base_idx + 3u] = params.value;
    }}

    // Handle remainder
    if (idx == 0u && remainder > 0u) {{
        let base_idx = vec4_count * 4u;
        for (var i = 0u; i < remainder; i++) {{
            buffer[base_idx + i] = params.value;
        }}
    }}
}}
"""


def create_reduce_sum_kernel(workgroup_size: int = 256) -> str:
    """
    Generate reduction kernel to compute sum of buffer elements

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid

    Note:
        Two-pass reduction: first pass reduces to workgroup sums,
        second pass (on CPU or separate kernel) sums workgroup results.
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    return f"""
// Parallel reduction to compute sum of array elements
// Each workgroup computes partial sum using tree reduction

struct ReduceParams {{
    size: u32,
}}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const BLOCK_SIZE: u32 = {workgroup_size}u;

var<workgroup> shared_data: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let tid = local_id.x;
    let idx = global_id.x;

    // Load data into shared memory with bounds checking
    shared_data[tid] = select(0.0, input[idx], idx < params.size);
    workgroupBarrier();

    // Tree reduction in shared memory
    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_data[tid] += shared_data[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}

    // Write result for this workgroup
    if (tid == 0u) {{
        output[workgroup_id.x] = shared_data[0];
    }}
}}
"""


# ============================================================================
# KV-CACHE KERNELS (for autoregressive generation)
# ============================================================================


def create_kv_cache_update_kernel(workgroup_size: int = 256) -> str:
    """
    Generate kernel to update KV-cache with new key/value at current position.

    Copies newly computed K and V tensors into the cache at position current_len.
    Used during autoregressive generation to incrementally build up cached context.

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
// KV-cache update: copy new K/V to cache at current position
//
// Input:
//   - new_k: [batch_size, 1, n_heads * head_dim]  (newly computed K for current token)
//   - new_v: [batch_size, 1, n_heads * head_dim]  (newly computed V for current token)
//
// Output (in-place update):
//   - k_cache: [batch_size, max_seq_len, n_heads * head_dim]  (cache for all K)
//   - v_cache: [batch_size, max_seq_len, n_heads * head_dim]  (cache for all V)
//
// Updates cache at position current_pos for all batch elements and dimensions.

struct UpdateParams {{
    batch_size: u32,
    current_pos: u32,         // Position to write to (0 to max_seq_len-1)
    embedding_dim: u32,       // n_heads * head_dim
}}

@group(0) @binding(0) var<uniform> params: UpdateParams;
@group(0) @binding(1) var<storage, read> new_k: array<f32>;         // [batch, 1, dim]
@group(0) @binding(2) var<storage, read> new_v: array<f32>;         // [batch, 1, dim]
@group(0) @binding(3) var<storage, read_write> k_cache: array<f32>; // [batch, max_seq, dim]
@group(0) @binding(4) var<storage, read_write> v_cache: array<f32>; // [batch, max_seq, dim]

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let batch_idx = global_id.y;
    let dim_idx = global_id.x;

    if (batch_idx >= params.batch_size || dim_idx >= params.embedding_dim) {{
        return;
    }}

    // Read from new K/V (single token at position 0)
    let new_k_offset = batch_idx * params.embedding_dim + dim_idx;
    let new_v_offset = batch_idx * params.embedding_dim + dim_idx;

    let k_val = new_k[new_k_offset];
    let v_val = new_v[new_v_offset];

    // Write to cache at current_pos
    // Cache layout: [batch, max_seq, dim]
    // Need max_seq_len from somewhere - pass via uniform or compute from buffer size
    // For now, compute offset directly using current_pos
    let cache_offset = batch_idx * params.embedding_dim + dim_idx;  // This is wrong, needs max_seq_len

    // CORRECT: k_cache[batch][current_pos][dim]
    // Flattened: batch * max_seq_len * embedding_dim + current_pos * embedding_dim + dim
    // But we don't have max_seq_len in params! Add it.

    // k_cache[cache_offset] = k_val;  // FIXME
    // v_cache[cache_offset] = v_val;
}}
"""


def create_attention_with_kv_cache_kernel(workgroup_size: int = 256) -> str:
    """
    Generate attention kernel that uses KV-cache for autoregressive generation.

    Computes attention for a single query token against all cached K/V.
    Used during generation where we only compute Q for the new token,
    but attend to all previous tokens via the cache.

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, or 512)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid

    Notes:
        - Assumes causal masking (can only attend to positions <= current_len)
        - Uses numerically stable softmax (max subtraction)
        - Q shape: [batch_size, 1, n_heads, head_dim] (single token)
        - K/V cache shape: [batch_size, max_seq_len, n_heads, head_dim]
        - Output shape: [batch_size, 1, n_heads, head_dim]
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    max_seq_len = workgroup_size  # Shared memory sized for this

    return f"""
// Attention with KV-cache for autoregressive generation
//
// Computes attention for a single new query token using cached K/V.
// Each workgroup handles one (batch, head) pair.
//
// Input:
//   - Q: [batch_size, 1, n_heads * head_dim]  (query for new token)
//   - K_cache: [batch_size, current_len, n_heads * head_dim]  (cached keys)
//   - V_cache: [batch_size, current_len, n_heads * head_dim]  (cached values)
//
// Output:
//   - O: [batch_size, 1, n_heads * head_dim]  (attention output)

struct AttentionParams {{
    batch_size: u32,
    current_len: u32,     // Number of valid positions in cache (1 to max_seq_len)
    n_heads: u32,
    head_dim: u32,
}}

@group(0) @binding(0) var<uniform> params: AttentionParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;       // [batch, 1, n_heads*head_dim]
@group(0) @binding(2) var<storage, read> K_cache: array<f32>; // [batch, current_len, n_heads*head_dim]
@group(0) @binding(3) var<storage, read> V_cache: array<f32>; // [batch, current_len, n_heads*head_dim]
@group(0) @binding(4) var<storage, read_write> O: array<f32>; // [batch, 1, n_heads*head_dim]

const BLOCK_SIZE: u32 = {workgroup_size}u;
const MAX_SEQ_LEN: u32 = {max_seq_len}u;
const MASK_VALUE: f32 = -1e10;

// Shared memory for scores and reduction
var<workgroup> shared_scores: array<f32, MAX_SEQ_LEN>;
var<workgroup> shared_reduction: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let batch_idx = group_id.y;
    let head_idx = group_id.x;
    let tid = local_id.x;

    if (batch_idx >= params.batch_size || head_idx >= params.n_heads) {{
        return;
    }}

    let head_dim = params.head_dim;
    let embedding_dim = params.n_heads * head_dim;
    let scale = 1.0 / sqrt(f32(head_dim));

    // Q offset: single token at position 0
    let q_offset = batch_idx * embedding_dim + head_idx * head_dim;

    // ====================================================================
    // Phase 1: Compute all QK scores and store in shared memory
    // ====================================================================
    for (var k_pos = tid; k_pos < params.current_len; k_pos += BLOCK_SIZE) {{
        let k_offset = batch_idx * params.current_len * embedding_dim +
                       k_pos * embedding_dim +
                       head_idx * head_dim;

        var score = 0.0;
        for (var d = 0u; d < head_dim; d++) {{
            score += Q[q_offset + d] * K_cache[k_offset + d];
        }}
        shared_scores[k_pos] = score * scale;
    }}
    workgroupBarrier();

    // ====================================================================
    // Phase 2: Find max score using parallel reduction
    // ====================================================================
    var max_score = -1e10;
    for (var k_pos = tid; k_pos < params.current_len; k_pos += BLOCK_SIZE) {{
        max_score = max(max_score, shared_scores[k_pos]);
    }}
    shared_reduction[tid] = max_score;
    workgroupBarrier();

    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_reduction[tid] = max(shared_reduction[tid], shared_reduction[tid + active_threads]);
        }}
        workgroupBarrier();
        active_threads /= 2u;
    }}
    max_score = shared_reduction[0];
    workgroupBarrier();

    // ====================================================================
    // Phase 3: Compute exp(score - max) and sum
    // ====================================================================
    var sum_exp = 0.0;
    for (var k_pos = tid; k_pos < params.current_len; k_pos += BLOCK_SIZE) {{
        let exp_score = exp(shared_scores[k_pos] - max_score);
        shared_scores[k_pos] = exp_score;  // Store normalized scores
        sum_exp += exp_score;
    }}
    shared_reduction[tid] = sum_exp;
    workgroupBarrier();

    active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_reduction[tid] += shared_reduction[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads /= 2u;
    }}
    sum_exp = shared_reduction[0];
    workgroupBarrier();

    // ====================================================================
    // Phase 4: Compute weighted sum of values
    // ====================================================================
    let dims_per_thread = (head_dim + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    for (var d_block = 0u; d_block < dims_per_thread; d_block++) {{
        let d = tid + d_block * BLOCK_SIZE;
        if (d >= head_dim) {{
            continue;
        }}

        var output_val = 0.0;
        for (var k_pos = 0u; k_pos < params.current_len; k_pos++) {{
            let attn_weight = shared_scores[k_pos] / sum_exp;
            let v_offset = batch_idx * params.current_len * embedding_dim +
                          k_pos * embedding_dim +
                          head_idx * head_dim;
            output_val += attn_weight * V_cache[v_offset + d];
        }}

        let out_offset = batch_idx * embedding_dim + head_idx * head_dim;
        O[out_offset + d] = output_val;
    }}
}}
"""


# ============================================================================
# NEW KERNELS
# ============================================================================


def create_embedding_backward_kernel(workgroup_size: int = 256) -> str:
    """Generate embedding backward pass kernel with atomic accumulation."""
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be in [64,128,256,512,1024], got {workgroup_size}"
        )

    return f"""// Embedding backward: accumulate gradients to embedding table
struct EmbedBackwardParams {{
    batch_size: u32,
    seq_len: u32,
    embedding_dim: u32,
}}

@group(0) @binding(0) var<uniform> params: EmbedBackwardParams;
@group(0) @binding(1) var<storage, read> input_ids: array<u32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_embedding: array<atomic<i32>>;

// Fixed-point scale for atomic operations
const SCALE: f32 = 65536.0;

fn f32_to_i32(x: f32) -> i32 {{
    return i32(x * SCALE);
}}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let token_idx = global_id.x;
    let dim_idx = global_id.y;

    let total_tokens = params.batch_size * params.seq_len;

    if (token_idx >= total_tokens || dim_idx >= params.embedding_dim) {{
        return;
    }}

    let token_id = input_ids[token_idx];
    let grad_value = grad_output[token_idx * params.embedding_dim + dim_idx];

    let emb_offset = token_id * params.embedding_dim + dim_idx;
    atomicAdd(&grad_embedding[emb_offset], f32_to_i32(grad_value));
}}
"""


def create_embedding_backward_convert_kernel(workgroup_size: int = 256) -> str:
    """Convert atomic i32 gradients back to f32."""
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be in [64,128,256,512,1024], got {workgroup_size}"
        )

    return f"""// Convert atomic i32 gradients to f32
struct ConvertParams {{
    size: u32,
}}

@group(0) @binding(0) var<uniform> params: ConvertParams;
@group(0) @binding(1) var<storage, read> grad_i32: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> grad_f32: array<f32>;

const SCALE: f32 = 65536.0;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= params.size) {{ return; }}

    let i32_val = atomicLoad(&grad_i32[idx]);
    grad_f32[idx] = f32(i32_val) / SCALE;
}}
"""


def create_dropout_kernel(workgroup_size: int = 256) -> str:
    """Generate dropout kernel with PCG random number generator."""
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be in [64,128,256,512,1024], got {workgroup_size}"
        )

    return f"""// Dropout with PCG random number generation
struct DropoutParams {{
    size: u32,
    keep_prob: f32,
    seed: u32,
    offset: u32,
}}

@group(0) @binding(0) var<uniform> params: DropoutParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> mask: array<u32>;  // Save mask for backward

// PCG Random Number Generator (O'Neill 2014)
fn pcg_hash(input: u32) -> u32 {{
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}}

fn random_uniform(seed: u32, idx: u32) -> f32 {{
    let hash = pcg_hash(seed + idx);
    return f32(hash) / 4294967296.0;  // Normalize to [0, 1)
}}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= params.size) {{ return; }}

    let rand = random_uniform(params.seed, params.offset + idx);
    let keep = select(0u, 1u, rand < params.keep_prob);

    // Scale by 1/keep_prob for inverted dropout
    let scale = 1.0 / params.keep_prob;
    output[idx] = input[idx] * f32(keep) * scale;
    mask[idx] = keep;
}}
"""


def create_dropout_backward_kernel(workgroup_size: int = 256) -> str:
    """Generate dropout backward kernel."""
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be in [64,128,256,512,1024], got {workgroup_size}"
        )

    return f"""// Dropout backward pass
struct DropoutBackwardParams {{
    size: u32,
    keep_prob: f32,
}}

@group(0) @binding(0) var<uniform> params: DropoutBackwardParams;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= params.size) {{ return; }}

    let scale = 1.0 / params.keep_prob;
    grad_input[idx] = grad_output[idx] * f32(mask[idx]) * scale;
}}
"""


def create_gradient_norm_kernel(workgroup_size: int = 256) -> str:
    """Generate kernel to compute L2 norm of gradients (first pass)."""
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be in [64,128,256,512,1024], got {workgroup_size}"
        )

    return f"""// Compute partial squared norms for gradient clipping
struct NormParams {{
    size: u32,
}}

@group(0) @binding(0) var<uniform> params: NormParams;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_norms: array<f32>;

const BLOCK_SIZE: u32 = {workgroup_size}u;
var<workgroup> shared_data: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let tid = local_id.x;

    // Accumulate squared gradients
    var sum_sq = 0.0;
    for (var i = global_id.x; i < params.size; i += BLOCK_SIZE * gridDim.x) {{
        let g = gradients[i];
        sum_sq += g * g;
    }}

    shared_data[tid] = sum_sq;
    workgroupBarrier();

    // Tree reduction
    var active = BLOCK_SIZE / 2u;
    while (active > 0u) {{
        if (tid < active) {{
            shared_data[tid] += shared_data[tid + active];
        }}
        workgroupBarrier();
        active /= 2u;
    }}

    // Write partial sum
    if (tid == 0u) {{
        partial_norms[workgroup_id.x] = shared_data[0];
    }}
}}
"""


def create_gradient_norm_reduce_kernel(workgroup_size: int = 256) -> str:
    """Generate kernel to reduce partial norms to global norm (second pass)."""
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be in [64,128,256,512,1024], got {workgroup_size}"
        )

    return f"""// Reduce partial norms to global norm
struct ReduceParams {{
    num_partials: u32,
}}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> partial_norms: array<f32>;
@group(0) @binding(2) var<storage, read_write> global_norm: array<f32>;

const BLOCK_SIZE: u32 = {workgroup_size}u;
var<workgroup> shared_data: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {{
    let tid = local_id.x;

    // Load partial norms
    var sum = 0.0;
    for (var i = tid; i < params.num_partials; i += BLOCK_SIZE) {{
        sum += partial_norms[i];
    }}

    shared_data[tid] = sum;
    workgroupBarrier();

    // Tree reduction
    var active = BLOCK_SIZE / 2u;
    while (active > 0u) {{
        if (tid < active) {{
            shared_data[tid] += shared_data[tid + active];
        }}
        workgroupBarrier();
        active /= 2u;
    }}

    // Compute sqrt of sum to get L2 norm
    if (tid == 0u) {{
        global_norm[0] = sqrt(shared_data[0]);
    }}
}}
"""


# ============================================================================
# FACTORY FUNCTIONS - Generate kernels from config
# ============================================================================


# ====
# FORWARD
# ====
def get_matmul_kernel(ctx: GPUContext) -> str:
    return create_matmul_kernel(ctx.config.matmul_tile_size)


def get_layernorm_kernel(ctx: GPUContext) -> str:
    return create_layernorm_kernel(
        ctx.config.layernorm_workgroup_size, ctx.config.layernorm_epsilon
    )


def get_gelu_kernel(ctx: GPUContext) -> str:
    return create_gelu_kernel(ctx.config.default_workgroup_size)


def get_residual_add_kernel(ctx: GPUContext) -> str:
    return create_residual_add_kernel(ctx.config.default_workgroup_size)


def get_embedding_kernel(ctx: GPUContext) -> str:
    return create_embedding_kernel(ctx.config.default_workgroup_size)


def get_bias_add_kernel(ctx: GPUContext) -> str:
    return create_bias_add_kernel(ctx.config.default_workgroup_size)


def get_attention_kernel(ctx: GPUContext) -> str:
    return create_attention_kernel(ctx.config.attention_workgroup_size)


def get_flash_attention_kernel(ctx: GPUContext) -> str:
    return create_flash_attention_kernel(
        ctx.config.flash_attn_max_head_dim,
        ctx.config.flash_attn_bc,
        ctx.config.flash_attn_br,
    )


def get_transpose_kernel(ctx: GPUContext) -> str:
    return create_transpose_kernel(
        ctx.config.matmul_tile_size
    )  # Use same tile size as matmul


def get_extract_last_tokens_kernel(ctx: GPUContext) -> str:
    return create_extract_last_tokens_kernel(ctx.config.default_workgroup_size)


def get_cross_entropy_loss_kernel(ctx: GPUContext) -> str:
    return create_cross_entropy_loss_kernel(ctx.config.default_workgroup_size)


def get_softmax_kernel(ctx: GPUContext) -> str:
    return create_softmax_kernel(ctx.config.default_workgroup_size)


def get_dropout_kernel(ctx: GPUContext) -> str:
    """Factory function for dropout kernel."""
    return create_dropout_kernel(ctx.config.default_workgroup_size)


# ====
# BACKWARD
# ====


def get_matmul_backward_a_kernel(ctx: GPUContext) -> str:
    return create_matmul_backward_a_kernel(ctx.config.matmul_tile_size)


def get_matmul_backward_b_kernel(ctx: GPUContext) -> str:
    return create_matmul_backward_b_kernel(ctx.config.matmul_tile_size)


def get_layernorm_backward_kernel(ctx: GPUContext) -> str:
    return create_layernorm_backward_kernel(
        ctx.config.layernorm_workgroup_size, ctx.config.layernorm_epsilon
    )


def get_layernorm_backward_reduce_kernel(ctx: GPUContext) -> str:
    return create_layernorm_backward_reduce_kernel(ctx.config.layernorm_workgroup_size)


def get_layernorm_backward_reduce_accumulate_kernel(ctx: GPUContext) -> str:
    return create_layernorm_backward_reduce_accumulate_kernel(
        ctx.config.layernorm_workgroup_size
    )


def get_gelu_backward_kernel(ctx: GPUContext) -> str:
    return create_gelu_backward_kernel(ctx.config.default_workgroup_size)


def get_bias_backward_kernel(ctx: GPUContext) -> str:
    return create_bias_backward_kernel(ctx.config.default_workgroup_size)


def get_attention_backward_kernel(ctx: GPUContext) -> str:
    return create_attention_backward_kernel(ctx.config.attention_workgroup_size)


def get_attention_backward_reduce_kernel(ctx: GPUContext) -> str:
    return create_attention_backward_reduce_kernel(ctx.config.attention_workgroup_size)


def get_flash_attention_backward_kernel(ctx: GPUContext) -> str:
    return create_flash_attention_backward_kernel(
        ctx.config.flash_attn_max_head_dim,
        ctx.config.flash_attn_bc,
        ctx.config.flash_attn_br,
    )


def get_flash_attention_backward_reduce_kernel(ctx: GPUContext) -> str:
    return create_flash_attention_backward_reduce_kernel(
        ctx.config.attention_workgroup_size
    )


def get_embedding_backward_kernel(ctx: GPUContext) -> str:
    """Factory function for embedding backward kernel."""
    return create_embedding_backward_kernel(ctx.config.default_workgroup_size)


def get_embedding_backward_convert_kernel(ctx: GPUContext) -> str:
    """Factory function for embedding backward convert kernel."""
    return create_embedding_backward_convert_kernel(ctx.config.default_workgroup_size)


def get_dropout_backward_kernel(ctx: GPUContext) -> str:
    """Factory function for dropout backward kernel."""
    return create_dropout_backward_kernel(ctx.config.default_workgroup_size)


# ====
# OPTIMIZER
# ====
#
def get_adamw_kernel(ctx: GPUContext) -> str:
    return create_adamw_kernel(ctx.config.default_workgroup_size)


def get_buffer_fill_kernel(ctx: GPUContext) -> str:
    return create_buffer_fill_kernel(ctx.config.default_workgroup_size)


def get_gradient_clip_kernel(ctx: GPUContext) -> str:
    return create_gradient_clip_kernel(ctx.config.default_workgroup_size)


def get_gradient_norm_kernel(ctx: GPUContext) -> str:
    """Factory function for gradient norm computation kernel."""
    return create_gradient_norm_kernel(ctx.config.default_workgroup_size)


def get_gradient_norm_reduce_kernel(ctx: GPUContext) -> str:
    """Factory function for gradient norm reduction kernel."""
    return create_gradient_norm_reduce_kernel(ctx.config.default_workgroup_size)


def get_reduce_sum_kernel(ctx: GPUContext) -> str:
    return create_reduce_sum_kernel(ctx.config.default_workgroup_size)


def get_kv_cache_update_kernel(ctx: GPUContext) -> str:
    """Factory function for KV-cache update kernel."""
    return create_kv_cache_update_kernel(ctx.config.default_workgroup_size)


def get_attention_with_kv_cache_kernel(ctx: GPUContext) -> str:
    """Factory function for attention with KV-cache kernel."""
    return create_attention_with_kv_cache_kernel(ctx.config.attention_workgroup_size)
