"""WGSL kernels"""

from .gpu_types import GPUContext

# ============================================================================
# FORWARD PASS KERNELS
# ============================================================================


def create_matmul_kernel(tile_size: int, items_per_thread: int) -> str:
    """Generate matmul kernel with configurable tile size and register blocking"""
    if tile_size <= 0 or (tile_size & (tile_size - 1)) != 0:
        raise ValueError(f"tile_size must be power of 2, got {tile_size}")
    if tile_size > 32:
        raise ValueError(f"tile_size too large: {tile_size}. Maximum is 32.")
    if items_per_thread not in [1, 2, 4]:
        raise ValueError(f"items_per_thread must be 1, 2, or 4, got {items_per_thread}")

    workgroup_dim = tile_size // (items_per_thread if items_per_thread > 1 else 1)
    num_threads = workgroup_dim * workgroup_dim
    tile_elements = tile_size * tile_size
    elements_per_thread_load = tile_elements // num_threads

    return f"""
// Optimized tiled matrix multiplication: C = A @ B
// Tile size: {tile_size}x{tile_size}, Items per thread: {items_per_thread}x{items_per_thread}

struct MatmulParams {{
    M: u32,  // Rows of A
    K: u32,  // Cols of A / Rows of B
    N: u32,  // Cols of B
}}

@group(0) @binding(0) var<uniform> params: MatmulParams;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;


const TILE_SIZE: u32 = {tile_size}u;
const ITEMS_PER_THREAD: u32 = {items_per_thread}u;
const WORKGROUP_DIM: u32 = {workgroup_dim}u;
const NUM_THREADS: u32 = {num_threads}u;
const ELEMENTS_PER_THREAD_LOAD: u32 = {elements_per_thread_load}u;

var<workgroup> tile_A: array<f32, {tile_elements}>;
var<workgroup> tile_B: array<f32, {tile_elements}>;

@compute @workgroup_size({workgroup_dim}, {workgroup_dim}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let local_row = local_id.y;
    let local_col = local_id.x;
    let thread_id = local_row * WORKGROUP_DIM + local_col;

    let base_row = workgroup_id.y * TILE_SIZE;
    let base_col = workgroup_id.x * TILE_SIZE;

    // Initialize accumulator
    var acc: array<f32, {items_per_thread * items_per_thread}>;
    for (var i = 0u; i < ITEMS_PER_THREAD * ITEMS_PER_THREAD; i++) {{
        acc[i] = 0.0;
    }}

    // Number of tiles in K dimension
    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    // Loop over K dimension tiles
    for (var t = 0u; t < num_tiles; t++) {{
        // Load A tile - ALL threads cooperatively load
        for (var idx = 0u; idx < ELEMENTS_PER_THREAD_LOAD; idx++) {{
            let tile_idx = thread_id * ELEMENTS_PER_THREAD_LOAD + idx;
            let tile_row = tile_idx / TILE_SIZE;
            let tile_col = tile_idx % TILE_SIZE;

            // FIXED: Global indices for A matrix
            let a_row = base_row + tile_row;
            let a_col = t * TILE_SIZE + tile_col;

            if (a_row < params.M && a_col < params.K) {{
                tile_A[tile_idx] = A[a_row * params.K + a_col];
            }} else {{
                tile_A[tile_idx] = 0.0;
            }}
        }}

        // Load B tile - ALL threads cooperatively load
        for (var idx = 0u; idx < ELEMENTS_PER_THREAD_LOAD; idx++) {{
            let tile_idx = thread_id * ELEMENTS_PER_THREAD_LOAD + idx;
            let tile_row = tile_idx / TILE_SIZE;
            let tile_col = tile_idx % TILE_SIZE;

            // FIXED: Global indices for B matrix
            let b_row = t * TILE_SIZE + tile_row;
            let b_col = base_col + tile_col;

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
        let out_row = base_row + local_row * ITEMS_PER_THREAD + i;
        for (var j = 0u; j < ITEMS_PER_THREAD; j++) {{
            let out_col = base_col + local_col * ITEMS_PER_THREAD + j;
            if (out_row < params.M && out_col < params.N) {{
                C[out_row * params.N + out_col] = acc[i * ITEMS_PER_THREAD + j];
            }}
        }}
    }}
}}
"""


def create_layernorm_kernel(workgroup_size: int, epsilon: float) -> str:
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


def create_gelu_kernel(workgroup_size: int) -> str:
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


def create_residual_add_kernel(workgroup_size: int) -> str:
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


def create_embedding_kernel(workgroup_size: int) -> str:
    """
    Generate embedding lookup kernel with configurable workgroup size

    FIXED: Now uses 1D dispatch and loops over all dimensions for each token.
    Each thread processes one token and all its embedding dimensions.

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
// FIXED: 1D dispatch with per-token processing of all dimensions
// Each thread handles one token and processes all embedding dimensions

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
    let total_tokens = params.batch_size * params.seq_len;

    if (token_idx >= total_tokens) {{
        return;
    }}

    // Compute sequence position for positional encoding
    let seq_idx = token_idx % params.seq_len;

    // Look up token ID
    let token_id = input_ids[token_idx];

    // Compute offsets
    let emb_offset = token_id * params.embedding_dim;
    let pos_offset = seq_idx * params.embedding_dim;
    let out_offset = token_idx * params.embedding_dim;

    // Process ALL dimensions for this token
    // Each thread handles one complete token embedding
    for (var d = 0u; d < params.embedding_dim; d++) {{
        output[out_offset + d] = embedding_table[emb_offset + d] +
                                  pos_encoding[pos_offset + d];
    }}
}}
"""


def create_bias_add_kernel(workgroup_size: int) -> str:
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


def create_flash_attention_kernel(
    head_dim: int, Bc: int, Br: int, max_workgroup_storage: int
) -> str:
    """
    Generate FlashAttention kernel with configurable parameters

    Args:
        head_dim: Dimension per attention head (16, 32, , 64, 128, or 256)
        Bc: Block size for K/V (columns) - typically 8, 16, 32, or 64
        Br: Block size for Q (rows) - typically 8, 16, 32, or 64
        max_workgroup_storage: Maximum workgroup storage size in bytes (default 16384)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If parameters are invalid or exceed workgroup memory limits
    """
    if head_dim not in [16, 32, 64, 128, 256]:
        raise ValueError(f"head_dim must be 16, 32, 64, 128, or 256, got {head_dim}")

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
    oi_size = Br * head_dim
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

    # CHANGED: Use dynamic limit instead of hardcoded 65536
    if total_workgroup_bytes > max_workgroup_storage:
        raise ValueError(
            f"Workgroup memory {total_workgroup_bytes} bytes exceeds "
            f"device limit of {max_workgroup_storage} bytes. "
            f"Try smaller head_dim, Bc, or Br values. "
            f"Current config: head_dim={head_dim}, Bc={Bc}, Br={Br}"
        )

    return f"""
// FlashAttention: Memory-efficient attention using tiling and online softmax
// Based on: Dao et al. 2022 - "FlashAttention: Fast and Memory-Efficient Exact Attention"
// Optimized version with parallelized initialization and softmax
//
// Parameters: head_dim={head_dim}, Bc={Bc}, Br={Br}
// Workgroup memory: {total_workgroup_bytes} bytes (limit: {max_workgroup_storage})
// Threads per workgroup: {threads_per_workgroup}

struct FlashAttentionParams {{
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    Bc: u32,
    Br: u32,
    num_q_blocks: u32,
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

        O[o_offset] = Oi[row * HEAD_DIM + d_idx] / max(li[row], 1e-8);

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


def create_transpose_kernel(tilesize: int) -> str:
    """Generate matrix transpose kernel with bank conflict avoidance"""
    if tilesize & (tilesize - 1) != 0:
        raise ValueError(f"tilesize must be power of 2, got {tilesize}")
    if tilesize > 32:
        raise ValueError(f"tilesize too large {tilesize}. Maximum is 32.")

    # Add padding to avoid bank conflicts: tilesize + 1
    paddedsize = tilesize * (tilesize + 1)

    return f"""
// Matrix transpose with bank conflict avoidance
// Uses tiled approach with padding to prevent bank conflicts
// Tile size: {tilesize}x{tilesize}, Padded stride: {tilesize + 1}

struct TransposeParams {{
    rows: u32,
    cols: u32,
}}

@group(0) @binding(0) var<uniform> params: TransposeParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const TILESIZE: u32 = {tilesize}u;
const PADDEDSTRIDE: u32 = {tilesize + 1}u;

// Padded shared memory to avoid bank conflicts
var<workgroup> tile: array<f32, {paddedsize}>;

@compute @workgroup_size({tilesize}, {tilesize}, 1)
fn main(
    @builtin(global_invocation_id) globalid: vec3<u32>,
    @builtin(local_invocation_id) localid: vec3<u32>,
    @builtin(workgroup_id) workgroupid: vec3<u32>
) {{
    // Compute base tile coordinates
    let tilerow = workgroupid.y * TILESIZE;
    let tilecol = workgroupid.x * TILESIZE;

    // Global position in input matrix
    let row = tilerow + localid.y;
    let col = tilecol + localid.x;

    let localrow = localid.y;
    let localcol = localid.x;

    // Load tile from input (coalesced reads)
    if (row < params.rows && col < params.cols) {{
        tile[localrow * PADDEDSTRIDE + localcol] = input[row * params.cols + col];
    }}

    workgroupBarrier();

    // Write transposed tile to output (coalesced writes)
    // Output position: swap row/col and swap tile coords
    let outrow = tilecol + localid.y;  // Note: tilecol (not tilerow)
    let outcol = tilerow + localid.x;  // Note: tilerow (not tilecol)

    if (outrow < params.cols && outcol < params.rows) {{
        output[outrow * params.rows + outcol] = tile[localcol * PADDEDSTRIDE + localrow];
    }}
}}
"""


def create_extract_last_tokens_kernel(workgroup_size: int) -> str:
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


def create_softmax_kernel(workgroup_size: int) -> str:
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


def create_cross_entropy_loss_kernel(workgroup_size: int) -> str:
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


def create_gradient_normalization_kernel(workgroup_size: int) -> str:
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
def create_matmul_backward_a_kernel(tile_size: int, items_per_thread: int) -> str:
    """Generate backward A kernel: grad_A = grad_C @ B^T"""
    if tile_size <= 0 or (tile_size & (tile_size - 1)) != 0:
        raise ValueError(f"tile_size must be power of 2, got {tile_size}")
    if tile_size > 32:
        raise ValueError(f"tile_size too large: {tile_size}. Maximum is 32.")
    if items_per_thread not in [1, 2, 4]:
        raise ValueError(f"items_per_thread must be 1, 2, or 4, got {items_per_thread}")

    workgroup_dim = tile_size // (items_per_thread if items_per_thread > 1 else 1)
    num_threads = workgroup_dim * workgroup_dim
    tile_elements = tile_size * tile_size
    elements_per_thread_load = tile_elements // num_threads

    return f"""
// Backward pass for A: grad_A = grad_C @ B^T

struct BackwardAParams {{
    M: u32,
    N: u32,
    K: u32,
}}

@group(0) @binding(0) var<uniform> params: BackwardAParams;
@group(0) @binding(1) var<storage, read> grad_C: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_A: array<f32>;

const TILE_SIZE: u32 = {tile_size}u;
const ITEMS_PER_THREAD: u32 = {items_per_thread}u;
const WORKGROUP_DIM: u32 = {workgroup_dim}u;
const NUM_THREADS: u32 = {num_threads}u;
const ELEMENTS_PER_THREAD_LOAD: u32 = {elements_per_thread_load}u;

var<workgroup> tile_grad_C: array<f32, {tile_elements}>;
var<workgroup> tile_B: array<f32, {tile_elements}>;

@compute @workgroup_size({workgroup_dim}, {workgroup_dim}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let local_row = local_id.y;
    let local_col = local_id.x;
    let thread_id = local_row * WORKGROUP_DIM + local_col;

    let base_row = workgroup_id.y * TILE_SIZE;
    let base_col = workgroup_id.x * TILE_SIZE;

    var acc: array<f32, {items_per_thread * items_per_thread}>;
    for (var i = 0u; i < ITEMS_PER_THREAD * ITEMS_PER_THREAD; i++) {{
        acc[i] = 0.0;
    }}

    let num_tiles = (params.N + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {{
        // Load grad_C tile
        for (var idx = 0u; idx < ELEMENTS_PER_THREAD_LOAD; idx++) {{
            let tile_idx = thread_id * ELEMENTS_PER_THREAD_LOAD + idx;
            let tile_row = tile_idx / TILE_SIZE;
            let tile_col = tile_idx % TILE_SIZE;

            let gc_row = base_row + tile_row;
            let gc_col = t * TILE_SIZE + tile_col;

            if (gc_row < params.M && gc_col < params.N) {{
                tile_grad_C[tile_idx] = grad_C[gc_row * params.N + gc_col];
            }} else {{
                tile_grad_C[tile_idx] = 0.0;
            }}
        }}

        // Load B tile and store it TRANSPOSED in shared memory
        // So that tile_B[row][col] contains B^T[row][col] = B[col][row]
        for (var idx = 0u; idx < ELEMENTS_PER_THREAD_LOAD; idx++) {{
            let tile_idx = thread_id * ELEMENTS_PER_THREAD_LOAD + idx;
            let tile_row = tile_idx / TILE_SIZE;
            let tile_col = tile_idx % TILE_SIZE;

            // We want tile_B to contain B^T[t*TILE+row, base_col+col]
            // Which equals B[base_col+col, t*TILE+row]
            let b_row = base_col + tile_col;  // Note the swap
            let b_col = t * TILE_SIZE + tile_row;  // Note the swap

            if (b_row < params.K && b_col < params.N) {{
                // Store transposed: tile_B[row][col] = B[col][row]
                tile_B[tile_row * TILE_SIZE + tile_col] = B[b_row * params.N + b_col];
            }} else {{
                tile_B[tile_row * TILE_SIZE + tile_col] = 0.0;
            }}
        }}

        workgroupBarrier();

        // Compute: grad_A += grad_C @ B^T
        for (var k = 0u; k < TILE_SIZE; k++) {{
            for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
                let gc_val = tile_grad_C[(local_row * ITEMS_PER_THREAD + i) * TILE_SIZE + k];
                for (var j = 0u; j < ITEMS_PER_THREAD; j++) {{
                    // Access B^T: tile_B[k][local_col * IPT + j]
                    let b_val = tile_B[k * TILE_SIZE + (local_col * ITEMS_PER_THREAD + j)];
                    acc[i * ITEMS_PER_THREAD + j] += gc_val * b_val;
                }}
            }}
        }}

        workgroupBarrier();
    }}

    // Write results
    for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
        let out_row = base_row + local_row * ITEMS_PER_THREAD + i;
        for (var j = 0u; j < ITEMS_PER_THREAD; j++) {{
            let out_col = base_col + local_col * ITEMS_PER_THREAD + j;
            if (out_row < params.M && out_col < params.K) {{
                grad_A[out_row * params.K + out_col] = acc[i * ITEMS_PER_THREAD + j];
            }}
        }}
    }}
}}
"""


def create_matmul_backward_b_kernel(tile_size: int, items_per_thread: int) -> str:
    """Generate backward B kernel: grad_B = A^T @ grad_C"""
    if tile_size <= 0 or (tile_size & (tile_size - 1)) != 0:
        raise ValueError(f"tile_size must be power of 2, got {tile_size}")
    if tile_size > 32:
        raise ValueError(f"tile_size too large: {tile_size}. Maximum is 32.")
    if items_per_thread not in [1, 2, 4]:
        raise ValueError(f"items_per_thread must be 1, 2, or 4, got {items_per_thread}")

    workgroup_dim = tile_size // (items_per_thread if items_per_thread > 1 else 1)
    num_threads = workgroup_dim * workgroup_dim
    tile_elements = tile_size * tile_size
    elements_per_thread_load = tile_elements // num_threads

    return f"""
// Backward pass for B: grad_B = A^T @ grad_C
// Tile size: {tile_size}x{tile_size}, Items per thread: {items_per_thread}x{items_per_thread}

struct BackwardBParams {{
    M: u32,  // Rows of A
    K: u32,  // Cols of A / Cols of grad_B
    N: u32,  // Cols of grad_C
}}

@group(0) @binding(0) var<uniform> params: BackwardBParams;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> grad_C: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_B: array<f32>;

const TILE_SIZE: u32 = {tile_size}u;
const ITEMS_PER_THREAD: u32 = {items_per_thread}u;
const WORKGROUP_DIM: u32 = {workgroup_dim}u;
const NUM_THREADS: u32 = {num_threads}u;
const ELEMENTS_PER_THREAD_LOAD: u32 = {elements_per_thread_load}u;

var<workgroup> tile_A: array<f32, {tile_elements}>;
var<workgroup> tile_grad_C: array<f32, {tile_elements}>;

@compute @workgroup_size({workgroup_dim}, {workgroup_dim}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let local_row = local_id.y;
    let local_col = local_id.x;
    let thread_id = local_row * WORKGROUP_DIM + local_col;

    let base_row = workgroup_id.y * TILE_SIZE;
    let base_col = workgroup_id.x * TILE_SIZE;

    // Initialize accumulator
    var acc: array<f32, {items_per_thread * items_per_thread}>;
    for (var i = 0u; i < ITEMS_PER_THREAD * ITEMS_PER_THREAD; i++) {{
        acc[i] = 0.0;
    }}

    // Number of tiles in M dimension
    let num_tiles = (params.M + TILE_SIZE - 1u) / TILE_SIZE;

    // Loop over M dimension tiles
    for (var t = 0u; t < num_tiles; t++) {{
        // Load A^T tile (transpose on-the-fly) - ALL threads cooperatively load
        for (var idx = 0u; idx < ELEMENTS_PER_THREAD_LOAD; idx++) {{
            let tile_idx = thread_id * ELEMENTS_PER_THREAD_LOAD + idx;
            let tile_row = tile_idx / TILE_SIZE;
            let tile_col = tile_idx % TILE_SIZE;

            // FIXED: Global indices for A matrix (transposed access)
            // We want A^T[tile_row, tile_col] = A[tile_col, tile_row] in global coords
            let a_row = t * TILE_SIZE + tile_col;  // Swap for transpose
            let a_col = base_row + tile_row;  // Swap for transpose

            if (a_row < params.M && a_col < params.K) {{
                tile_A[tile_idx] = A[a_row * params.K + a_col];
            }} else {{
                tile_A[tile_idx] = 0.0;
            }}
        }}

        // Load grad_C tile - ALL threads cooperatively load
        for (var idx = 0u; idx < ELEMENTS_PER_THREAD_LOAD; idx++) {{
            let tile_idx = thread_id * ELEMENTS_PER_THREAD_LOAD + idx;
            let tile_row = tile_idx / TILE_SIZE;
            let tile_col = tile_idx % TILE_SIZE;

            // FIXED: Global indices for grad_C matrix
            let gc_row = t * TILE_SIZE + tile_row;
            let gc_col = base_col + tile_col;

            if (gc_row < params.M && gc_col < params.N) {{
                tile_grad_C[tile_idx] = grad_C[gc_row * params.N + gc_col];
            }} else {{
                tile_grad_C[tile_idx] = 0.0;
            }}
        }}

        workgroupBarrier();

        // Compute partial results
        for (var k = 0u; k < TILE_SIZE; k++) {{
            for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
                let a_val = tile_A[(local_row * ITEMS_PER_THREAD + i) * TILE_SIZE + k];
                for (var j = 0u; j < ITEMS_PER_THREAD; j++) {{
                    let gc_val = tile_grad_C[k * TILE_SIZE + (local_col * ITEMS_PER_THREAD + j)];
                    acc[i * ITEMS_PER_THREAD + j] += a_val * gc_val;
                }}
            }}
        }}

        workgroupBarrier();
    }}

    // Write results
    for (var i = 0u; i < ITEMS_PER_THREAD; i++) {{
        let out_row = base_row + local_row * ITEMS_PER_THREAD + i;
        for (var j = 0u; j < ITEMS_PER_THREAD; j++) {{
            let out_col = base_col + local_col * ITEMS_PER_THREAD + j;
            if (out_row < params.K && out_col < params.N) {{
                grad_B[out_row * params.N + out_col] = acc[i * ITEMS_PER_THREAD + j];
            }}
        }}
    }}
}}
"""


def create_layernorm_backward_kernel(workgroup_size: int, epsilon: float) -> str:
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


def create_layernorm_backward_reduce_kernel(workgroup_size: int) -> str:
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
    workgroup_size: int,
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


def create_gelu_backward_kernel(workgroup_size: int) -> str:
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


def create_bias_backward_kernel(workgroup_size: int) -> str:
    """
    Each workgroup computes partial sum for a subset of rows for one bias dimension.
    Multiple workgroups may process the same bias dimension.

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
// Bias Backward - STAGE 1: Compute partial sums
// Each workgroup computes partial sum over a subset of batch elements
// Output: workspace[workgroup_id, dim]

struct BiasParams {{
    n_elements: u32,
    dim: u32,
}}

@group(0) @binding(0) var<uniform> params: BiasParams;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> workspace: array<f32>;

const BLOCK_SIZE: u32 = {workgroup_size}u;

var<workgroup> shared_data: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {{
    let workgroup_idx = workgroup_id.x;
    let tid = local_id.x;
    let total_workgroups = num_workgroups.x;

    // Compute how many elements this workgroup processes
    let elements_per_workgroup = (params.n_elements + total_workgroups - 1u) / total_workgroups;
    let start_element = workgroup_idx * elements_per_workgroup;
    let end_element = min(start_element + elements_per_workgroup, params.n_elements);

    // Each workgroup processes all dimensions
    for (var dim_idx = 0u; dim_idx < params.dim; dim_idx++) {{
        // Parallel sum across assigned elements for this dimension
        var sum = 0.0;
        for (var elem = start_element + tid; elem < end_element; elem += BLOCK_SIZE) {{
            sum += grad_output[elem * params.dim + dim_idx];
        }}
        shared_data[tid] = sum;
        workgroupBarrier();

        // Tree reduction within workgroup
        var active_threads = BLOCK_SIZE / 2u;
        while (active_threads > 0u) {{
            if (tid < active_threads) {{
                shared_data[tid] += shared_data[tid + active_threads];
            }}
            workgroupBarrier();
            active_threads >>= 1u;
        }}

        // Thread 0 writes partial result to workspace
        if (tid == 0u) {{
            let workspace_idx = workgroup_idx * params.dim + dim_idx;
            workspace[workspace_idx] = shared_data[0];
        }}
        workgroupBarrier();
    }}
}}
"""


def create_bias_backward_reduce_kernel(workgroup_size: int) -> str:
    """
    Reduces partial sums from all workgroups to final gradient.
    One workgroup per bias dimension.

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
// Bias Backward - STAGE 2: Reduce partial sums
// Each workgroup reduces all partial sums for one bias dimension

struct ReduceParams {{
    num_workgroups: u32,
    dim: u32,
}}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> workspace: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_bias: array<f32>;

const BLOCK_SIZE: u32 = {workgroup_size}u;

var<workgroup> shared_data: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let dim_idx = workgroup_id.x;
    let tid = local_id.x;

    if (dim_idx >= params.dim) {{
        return;
    }}

    // Sum all partial results for this dimension
    var sum = 0.0;
    for (var wg = tid; wg < params.num_workgroups; wg += BLOCK_SIZE) {{
        let workspace_idx = wg * params.dim + dim_idx;
        sum += workspace[workspace_idx];
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

    // Thread 0 writes final result
    if (tid == 0u) {{
        grad_bias[dim_idx] = shared_data[0];
    }}
}}
"""


def create_flash_attention_backward_kernel(
    head_dim: int, Bc: int, Br: int, max_workgroup_storage: int
) -> str:
    """
    Generate FlashAttention backward kernel - Stage 1

    Memory-efficient backward pass using tiling and recomputation.
    Outputs partial gradients for KV that must be reduced in stage 2.

    Args:
        head_dim: Dimension per attention head (16, 32, 64, 128, or 256)
        Bc: Block size for K/V (columns) - typically 16, 32, or 64
        Br: Block size for Q (rows) - typically 16, 32, or 64
        max_workgroup_storage: Maximum workgroup storage size in bytes (default 16384)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If parameters are invalid or exceed workgroup memory limits
    """
    if head_dim not in [16, 32, 64, 128, 256]:
        raise ValueError(f"head_dim must be 16, 32, 64, 128, or 256, got {head_dim}")

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

    # CHANGED: Use dynamic limit instead of hardcoded 65536
    if total_workgroup_bytes > max_workgroup_storage:
        raise ValueError(
            f"Workgroup memory {total_workgroup_bytes} bytes exceeds "
            f"device limit of {max_workgroup_storage} bytes. "
            f"Try smaller head_dim, Bc, or Br values. "
            f"Current config: head_dim={head_dim}, Bc={Bc}, Br={Br}"
        )

    return f"""
    // FlashAttention Backward Pass - STAGE 1
    // Fully parallelized, outputs partial gradients for KV
    // Parameters: head_dim={head_dim}, Bc={Bc}, Br={Br}
    // Workgroup memory: {total_workgroup_bytes} bytes (limit: {max_workgroup_storage})
    // Threads per workgroup: {threads_per_workgroup}

struct FlashAttentionBackwardParams {{
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    Bc: u32,
    Br: u32,
    q_block_idx: u32,
}}

@group(0) @binding(0) var<uniform> params: FlashAttentionBackwardParams;
@group(0) @binding(1) var<storage, read> grad_O: array<f32>;
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read> O: array<f32>;
@group(0) @binding(6) var<storage, read> L: array<f32>;
@group(0) @binding(7) var<storage, read> M: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_Q: array<f32>;
@group(0) @binding(9) var<storage, read_write> grad_K_workspace: array<f32>;
@group(0) @binding(10) var<storage, read_write> grad_V_workspace: array<f32>;

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
    let tid = local_id.x;

    if (batch_idx >= params.batch_size) {{
        return;
    }}

    let d = params.head_dim;
    let N = params.seq_len;
    let embedding_dim = params.n_heads * d;
    let scale = 1.0 / sqrt(f32(d));

    // Use q_block_idx from params instead of workgroup_id.x
    let block_row = params.q_block_idx;
    let q_start = block_row * Br;
    let q_end = min(q_start + Br, N);
    let actual_Br = q_end - q_start;

    if (actual_Br == 0u) {{
        return;
    }}

    // Compute workspace offset for this Q-block
    let total_tokens = params.batch_size * N;
    let workspace_offset = params.q_block_idx * total_tokens * embedding_dim;

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

        // Write PARTIAL dK to workspace = dS^T @ Q (parallelized, no race)
        for (var i = tid; i < actual_Bc * d; i += THREADS) {{
            let col = i / d;
            let d_idx = i % d;

            var sum = 0.0;
            for (var row = 0u; row < actual_Br; row++) {{
                sum += dSij[row * Bc + col] * Qi[row * HEAD_DIM + d_idx] * scale;
            }}

            let global_kv = kv_start + col;
            let workspace_idx = workspace_offset +
                               batch_idx * N * embedding_dim +
                               global_kv * embedding_dim +
                               head_idx * d + d_idx;
            grad_K_workspace[workspace_idx] = sum;
        }}

        // Write PARTIAL dV to workspace = P^T @ dO (parallelized, no race)
        for (var i = tid; i < actual_Bc * d; i += THREADS) {{
            let col = i / d;
            let d_idx = i % d;

            var sum = 0.0;
            for (var row = 0u; row < actual_Br; row++) {{
                sum += Pij[row * Bc + col] * dOi[row * HEAD_DIM + d_idx];
            }}

            let global_kv = kv_start + col;
            let workspace_idx = workspace_offset +
                               batch_idx * N * embedding_dim +
                               global_kv * embedding_dim +
                               head_idx * d + d_idx;
            grad_V_workspace[workspace_idx] = sum;
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


def create_flash_attention_backward_reduce_kernel(workgroup_size: int) -> str:
    """
    Generate FlashAttention backward reduction kernel - Stage 2

    Reduces partial grad_K and grad_V from all Q blocks into final gradients.
    Each workgroup processes one (token, dimension) pair.

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
// Reduces grad_K_workspace and grad_V_workspace across all Q blocks

struct ReduceParams {{
    batch_size: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    num_q_blocks: u32,
}}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> grad_K_workspace: array<f32>;
@group(0) @binding(2) var<storage, read> grad_V_workspace: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_K: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_V: array<f32>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {{
    let total_tokens = params.batch_size * params.seq_len;
    let embedding_dim = params.n_heads * params.head_dim;
    let global_idx = global_id.x;

    // Each thread processes one (token, dim) position
    if global_idx >= total_tokens * embedding_dim {{
        return;
    }}

    let token_idx = global_idx / embedding_dim;
    let dim_idx = global_idx % embedding_dim;

    // Reduce across all Q-blocks
    var sum_grad_K: f32 = 0.0;
    var sum_grad_V: f32 = 0.0;

    for (var q_block: u32 = 0u; q_block < params.num_q_blocks; q_block = q_block + 1u) {{
        let workspace_offset = q_block * total_tokens * embedding_dim;
        let workspace_idx = workspace_offset + token_idx * embedding_dim + dim_idx;

        sum_grad_K = sum_grad_K + grad_K_workspace[workspace_idx];
        sum_grad_V = sum_grad_V + grad_V_workspace[workspace_idx];
    }}

    // Write final reduced gradients
    let output_idx = token_idx * embedding_dim + dim_idx;
    grad_K[output_idx] = sum_grad_K;
    grad_V[output_idx] = sum_grad_V;
}}
"""


# ============================================================================
# OPTIMIZER KERNELS
# ============================================================================
def create_adamw_kernel(workgroup_size: int) -> str:
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


def create_gradient_clip_kernel(workgroup_size: int) -> str:
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


def create_buffer_fill_kernel(workgroup_size: int) -> str:
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


def create_reduce_sum_kernel(workgroup_size: int) -> str:
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


def create_kv_cache_update_kernel(workgroup_size: int) -> str:
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
//   - new_k: [batch_size, 1, embedding_dim]  (newly computed K for current token)
//   - new_v: [batch_size, 1, embedding_dim]  (newly computed V for current token)
//
// Output (in-place update):
//   - k_cache: [batch_size, max_seq_len, embedding_dim]  (cache for all K)
//   - v_cache: [batch_size, max_seq_len, embedding_dim]  (cache for all V)
//
// Updates cache at position current_pos for all batch elements and dimensions.

struct UpdateParams {{
    batch_size: u32,
    current_pos: u32,         // Position to write to (0 to max_seq_len-1)
    embedding_dim: u32,       // n_heads * head_dim
    max_seq_len: u32,         // Maximum sequence length cache can hold
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

    // Read from new K/V (single token)
    // Layout: [batch_size, 1, embedding_dim]
    // Flattened: batch * embedding_dim + dim
    let new_offset = batch_idx * params.embedding_dim + dim_idx;
    let k_val = new_k[new_offset];
    let v_val = new_v[new_offset];

    // Write to cache at current_pos
    // Layout: [batch_size, max_seq_len, embedding_dim]
    // Flattened: batch * max_seq_len * embedding_dim + current_pos * embedding_dim + dim
    let cache_offset = batch_idx * params.max_seq_len * params.embedding_dim +
                       params.current_pos * params.embedding_dim +
                       dim_idx;

    k_cache[cache_offset] = k_val;
    v_cache[cache_offset] = v_val;
}}
"""


def create_attention_with_kv_cache_kernel(workgroup_size: int) -> str:
    """Generate attention kernel that uses KV-cache for autoregressive generation."""
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be 64, 128, 256, 512, or 1024, got {workgroup_size}"
        )

    max_seq_len = workgroup_size

    return f"""
// Attention with KV-cache for autoregressive generation
struct AttentionParams {{
    batch_size: u32,
    current_len: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,  // ADDED: Need this for proper indexing
}}

@group(0) @binding(0) var<uniform> params: AttentionParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K_cache: array<f32>;  // [batch*max_seq, embedding]
@group(0) @binding(3) var<storage, read> V_cache: array<f32>;  // [batch*max_seq, embedding]
@group(0) @binding(4) var<storage, read_write> O: array<f32>;

const BLOCK_SIZE: u32 = {workgroup_size}u;
const MAX_SEQ_LEN: u32 = {max_seq_len}u;

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

    // Q offset: [batch, embedding]
    let q_offset = batch_idx * embedding_dim + head_idx * head_dim;

    // ====================================================================
    // Phase 1: Compute QK scores
    // ====================================================================
    for (var k_pos = tid; k_pos < params.current_len; k_pos += BLOCK_SIZE) {{
        // FIXED: Cache is [batch * max_seq_len, embedding]
        let cache_row = batch_idx * params.max_seq_len + k_pos;
        let k_offset = cache_row * embedding_dim + head_idx * head_dim;

        var score = 0.0;
        for (var d = 0u; d < head_dim; d++) {{
            score += Q[q_offset + d] * K_cache[k_offset + d];
        }}
        shared_scores[k_pos] = score * scale;
    }}
    workgroupBarrier();

    // ====================================================================
    // Phase 2: Find max score
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
    // Phase 3: Compute exp and sum
    // ====================================================================
    var sum_exp = 0.0;
    for (var k_pos = tid; k_pos < params.current_len; k_pos += BLOCK_SIZE) {{
        let exp_score = exp(shared_scores[k_pos] - max_score);
        shared_scores[k_pos] = exp_score;
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
    // Phase 4: Compute weighted sum of V
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

            // FIXED: Same cache indexing as K
            let cache_row = batch_idx * params.max_seq_len + k_pos;
            let v_offset = cache_row * embedding_dim + head_idx * head_dim;

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


def create_embedding_backward_kernel(workgroup_size: int) -> str:
    """
    Simply copies grad_output to workspace (no race conditions).

    Args:
        workgroup_size: Number of threads per workgroup

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be in [64,128,256,512,1024], got {workgroup_size}"
        )

    return f"""
// Embedding Backward - STAGE 1: Copy gradients to workspace

struct Stage1Params {{
    total_tokens: u32,
    embedding_dim: u32,
}}

@group(0) @binding(0) var<uniform> params: Stage1Params;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> workspace: array<f32>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    let total_size = params.total_tokens * params.embedding_dim;

    if (idx >= total_size) {{
        return;
    }}

    workspace[idx] = grad_output[idx];
}}
"""


def create_embedding_backward_reduce_kernel(workgroup_size: int) -> str:
    """
    For each vocab entry, sum gradients from all tokens that reference it.
    Each workgroup processes one (vocab_id, dimension) pair.

    Args:
        workgroup_size: Number of threads per workgroup

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be in [64,128,256,512,1024], got {workgroup_size}"
        )

    return f"""
// Embedding Backward - STAGE 2: Reduce workspace by vocab ID

struct ReduceParams {{
    total_tokens: u32,
    embedding_dim: u32,
    vocab_size: u32,
}}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> input_ids: array<u32>;
@group(0) @binding(2) var<storage, read> workspace: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_embedding: array<f32>;

const BLOCK_SIZE: u32 = {workgroup_size}u;

var<workgroup> shared_sum: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let vocab_id = workgroup_id.x;
    let dim_idx = workgroup_id.y;
    let tid = local_id.x;

    if (vocab_id >= params.vocab_size || dim_idx >= params.embedding_dim) {{
        return;
    }}

    // Sum gradients from all tokens that use this vocab_id
    var sum = 0.0;
    for (var token_idx = tid; token_idx < params.total_tokens; token_idx += BLOCK_SIZE) {{
        if (input_ids[token_idx] == vocab_id) {{
            sum += workspace[token_idx * params.embedding_dim + dim_idx];
        }}
    }}
    shared_sum[tid] = sum;
    workgroupBarrier();

    // Tree reduction
    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_sum[tid] += shared_sum[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads >>= 1u;
    }}

    // Thread 0 writes final result
    if (tid == 0u) {{
        grad_embedding[vocab_id * params.embedding_dim + dim_idx] = shared_sum[0];
    }}
}}
"""


def create_dropout_kernel(workgroup_size: int) -> str:
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


def create_dropout_backward_kernel(workgroup_size: int) -> str:
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


def create_gradient_norm_kernel(workgroup_size: int) -> str:
    """Generate kernel to compute L2 norm of gradients (first pass).

    Args:
        workgroup_size: Number of threads per workgroup (64, 128, 256, 512, or 1024)

    Returns:
        WGSL kernel source code as string

    Raises:
        ValueError: If workgroup_size is invalid

    Note:
        Each workgroup computes partial squared norm for its assigned region.
        Results are written to workspace at offset specified in params.
        This enables multiple gradient buffers to write to different sections
        of a shared workspace without overwriting each other.

        The kernel uses a grid-stride loop to handle cases where buffer size
        exceeds total number of threads dispatched.
    """
    if workgroup_size not in [64, 128, 256, 512, 1024]:
        raise ValueError(
            f"workgroup_size must be in [64,128,256,512,1024], got {workgroup_size}"
        )

    return f"""// Compute partial squared norms for gradient clipping
// Each workgroup processes a chunk and writes partial result to workspace

struct NormParams {{
    size: u32,
    workspace_offset: u32,
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
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {{
    let tid = local_id.x;
    let global_idx = global_id.x;

    // FIXED: Use builtin num_workgroups instead of non-existent gridDim
    let grid_size = BLOCK_SIZE * num_workgroups.x;

    // Grid-stride loop: each thread processes multiple elements
    var sum_sq = 0.0;
    for (var i = global_idx; i < params.size; i += grid_size) {{
        let g = gradients[i];
        sum_sq += g * g;
    }}

    shared_data[tid] = sum_sq;
    workgroupBarrier();

    // Tree reduction within workgroup
    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_data[tid] += shared_data[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads /= 2u;
    }}

    // Write partial sum to workspace at correct offset
    if (tid == 0u) {{
        partial_norms[params.workspace_offset + workgroup_id.x] = shared_data[0];
    }}
}}
"""


def create_gradient_norm_reduce_kernel(workgroup_size: int) -> str:
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
    var active_threads = BLOCK_SIZE / 2u;
    while (active_threads > 0u) {{
        if (tid < active_threads) {{
            shared_data[tid] += shared_data[tid + active_threads];
        }}
        workgroupBarrier();
        active_threads /= 2u;
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
    return create_matmul_kernel(
        ctx.config.matmul_tile_size, ctx.config.matmul_items_per_thread
    )


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


def get_flash_attention_kernel(ctx: GPUContext) -> str:
    return create_flash_attention_kernel(
        ctx.config.flash_attn_max_head_dim,
        ctx.config.flash_attn_bc,
        ctx.config.flash_attn_br,
        ctx.config.max_workgroup_storage_size,
    )


def get_transpose_kernel(ctx: GPUContext) -> str:
    return create_transpose_kernel(
        ctx.config.matmul_tile_size
    )  # Use same tile size as matmul


def get_extract_last_tokens_kernel(ctx: GPUContext) -> str:
    return create_extract_last_tokens_kernel(ctx.config.default_workgroup_size)


def get_cross_entropy_loss_kernel(ctx: GPUContext) -> str:
    return create_cross_entropy_loss_kernel(ctx.config.reduction_workgroup_size)


def get_softmax_kernel(ctx: GPUContext) -> str:
    return create_softmax_kernel(ctx.config.reduction_workgroup_size)


def get_dropout_kernel(ctx: GPUContext) -> str:
    """Factory function for dropout kernel."""
    return create_dropout_kernel(ctx.config.default_workgroup_size)


def get_gradient_normalization_kernel(ctx: GPUContext) -> str:
    """Factory function for gradient norm computation kernel."""
    return create_gradient_normalization_kernel(ctx.config.optimizer_workgroup_size)


# ====
# BACKWARD
# ====


def get_matmul_backward_a_kernel(ctx: GPUContext) -> str:
    return create_matmul_backward_a_kernel(
        ctx.config.matmul_tile_size, ctx.config.matmul_items_per_thread
    )


def get_matmul_backward_b_kernel(ctx: GPUContext) -> str:
    return create_matmul_backward_b_kernel(
        ctx.config.matmul_tile_size, ctx.config.matmul_items_per_thread
    )


def get_layernorm_backward_kernel(ctx: GPUContext) -> str:
    return create_layernorm_backward_kernel(
        ctx.config.layernorm_workgroup_size, ctx.config.layernorm_epsilon
    )


def get_layernorm_backward_reduce_kernel(ctx: GPUContext) -> str:
    return create_layernorm_backward_reduce_kernel(ctx.config.reduction_workgroup_size)


def get_layernorm_backward_reduce_accumulate_kernel(ctx: GPUContext) -> str:
    return create_layernorm_backward_reduce_accumulate_kernel(
        ctx.config.layernorm_workgroup_size
    )


def get_gelu_backward_kernel(ctx: GPUContext) -> str:
    return create_gelu_backward_kernel(ctx.config.default_workgroup_size)


def get_bias_backward_kernel(ctx: GPUContext) -> str:
    return create_bias_backward_kernel(ctx.config.reduction_workgroup_size)


def get_bias_backward_reduce_kernel(ctx: GPUContext) -> str:
    return create_bias_backward_reduce_kernel(ctx.config.reduction_workgroup_size)


def get_flash_attention_backward_kernel(ctx: GPUContext) -> str:
    return create_flash_attention_backward_kernel(
        ctx.config.flash_attn_max_head_dim,
        ctx.config.flash_attn_bc,
        ctx.config.flash_attn_br,
        ctx.config.max_workgroup_storage_size,
    )


def get_flash_attention_backward_reduce_kernel(ctx: GPUContext) -> str:
    return create_flash_attention_backward_reduce_kernel(
        ctx.config.reduction_workgroup_size
    )


def get_embedding_backward_kernel(ctx: GPUContext) -> str:
    """Factory function for embedding backward kernel."""
    return create_embedding_backward_kernel(ctx.config.default_workgroup_size)


def get_embedding_backward_reduce_kernel(ctx: GPUContext) -> str:
    """Factory function for embedding backward convert kernel."""
    return create_embedding_backward_reduce_kernel(ctx.config.default_workgroup_size)


def get_dropout_backward_kernel(ctx: GPUContext) -> str:
    """Factory function for dropout backward kernel."""
    return create_dropout_backward_kernel(ctx.config.default_workgroup_size)


# ====
# OPTIMIZER
# ====
#
def get_adamw_kernel(ctx: GPUContext) -> str:
    return create_adamw_kernel(ctx.config.optimizer_workgroup_size)


def get_gradient_clip_kernel(ctx: GPUContext) -> str:
    return create_gradient_clip_kernel(ctx.config.optimizer_workgroup_size)


def get_gradient_norm_kernel(ctx: GPUContext) -> str:
    """Factory function for gradient norm computation kernel."""
    return create_gradient_norm_kernel(ctx.config.optimizer_workgroup_size)


def get_gradient_norm_reduce_kernel(ctx: GPUContext) -> str:
    """Factory function for gradient norm reduction kernel."""
    return create_gradient_norm_reduce_kernel(ctx.config.default_workgroup_size)


def get_buffer_fill_kernel(ctx: GPUContext) -> str:
    return create_buffer_fill_kernel(ctx.config.default_workgroup_size)


def get_reduce_sum_kernel(ctx: GPUContext) -> str:
    return create_reduce_sum_kernel(ctx.config.default_workgroup_size)


def get_kv_cache_update_kernel(ctx: GPUContext) -> str:
    """Factory function for KV-cache update kernel."""
    return create_kv_cache_update_kernel(ctx.config.default_workgroup_size)


def get_attention_with_kv_cache_kernel(ctx: GPUContext) -> str:
    """Factory function for attention with KV-cache kernel."""
    return create_attention_with_kv_cache_kernel(ctx.config.attention_workgroup_size)
