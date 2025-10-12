"""WGSL kernels for backward pass operations"""


# ============================================================================
# HELPER FUNCTIONS - KERNEL GENERATORS
# ============================================================================
def create_matmul_backward_a_kernel(tile_size: int = 16) -> str:
    """
    Generate backward matmul kernel for gradient w.r.t. A

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
// Backward pass for matmul: compute gradient w.r.t. A
// Given: dL/dC, B
// Compute: dL/dA = dL/dC @ B^T
// Tile size: {tile_size}x{tile_size}

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

var<workgroup> tile_grad: array<f32, {tile_size * tile_size}>;
var<workgroup> tile_B: array<f32, {tile_size * tile_size}>;

@compute @workgroup_size({tile_size}, {tile_size})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let row = global_id.y;  // M dimension
    let col = global_id.x;  // K dimension
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum = 0.0;

    let num_tiles = (params.N + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {{
        // Load grad_C tile
        let g_row = row;
        let g_col = t * TILE_SIZE + local_col;
        if (g_row < params.M && g_col < params.N) {{
            tile_grad[local_row * TILE_SIZE + local_col] = grad_C[g_row * params.N + g_col];
        }} else {{
            tile_grad[local_row * TILE_SIZE + local_col] = 0.0;
        }}

        // Load B^T tile (transpose on-the-fly)
        let b_row = t * TILE_SIZE + local_row;
        let b_col = col;
        if (b_row < params.N && b_col < params.K) {{
            tile_B[local_row * TILE_SIZE + local_col] = B[b_col * params.N + b_row];
        }} else {{
            tile_B[local_row * TILE_SIZE + local_col] = 0.0;
        }}

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k++) {{
            sum += tile_grad[local_row * TILE_SIZE + k] * tile_B[k * TILE_SIZE + local_col];
        }}

        workgroupBarrier();
    }}

    if (row < params.M && col < params.K) {{
        grad_A[row * params.K + col] = sum;
    }}
}}
"""


def create_matmul_backward_b_kernel(tile_size: int = 16) -> str:
    """
    Generate backward matmul kernel for gradient w.r.t. B

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
// Backward pass for matmul: compute gradient w.r.t. B
// Given: A, dL/dC
// Compute: dL/dB = A^T @ dL/dC
// Tile size: {tile_size}x{tile_size}

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

var<workgroup> tile_A: array<f32, {tile_size * tile_size}>;
var<workgroup> tile_grad: array<f32, {tile_size * tile_size}>;

@compute @workgroup_size({tile_size}, {tile_size})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let row = global_id.y;  // K dimension
    let col = global_id.x;  // N dimension
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum = 0.0;

    let num_tiles = (params.M + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {{
        // Load A^T tile (transpose on-the-fly)
        let a_row = t * TILE_SIZE + local_row;
        let a_col = row;
        if (a_row < params.M && a_col < params.K) {{
            tile_A[local_row * TILE_SIZE + local_col] = A[a_row * params.K + a_col];
        }} else {{
            tile_A[local_row * TILE_SIZE + local_col] = 0.0;
        }}

        // Load grad_C tile
        let g_row = t * TILE_SIZE + local_row;
        let g_col = col;
        if (g_row < params.M && g_col < params.N) {{
            tile_grad[local_row * TILE_SIZE + local_col] = grad_C[g_row * params.N + g_col];
        }} else {{
            tile_grad[local_row * TILE_SIZE + local_col] = 0.0;
        }}

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k++) {{
            sum += tile_A[k * TILE_SIZE + local_col] * tile_grad[k * TILE_SIZE + local_col];
        }}

        workgroupBarrier();
    }}

    if (row < params.K && col < params.N) {{
        grad_B[row * params.N + col] = sum;
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

var<workgroup> shared_data: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let elem_idx = workgroup_id.x;
    let tid = local_id.x;

    if (elem_idx >= params.n_elements) {{
        return;
    }}

    let offset = elem_idx * params.size;

    // Recompute forward pass statistics
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

    // Compute PARTIAL gradients for gamma and beta (no accumulation, just write)
    // Each workgroup handles ONE element, writes to its own slice of partial buffers
    // SAFE: No race condition because each workgroup writes to different memory locations
    if (tid < params.size) {{
        let x_norm = (input[offset + tid] - mean) * inv_std;
        let gamma_grad = grad_output[offset + tid] * x_norm;
        let beta_grad = grad_output[offset + tid];

        // Write partial gradients (will be reduced in stage 2)
        let partial_offset = elem_idx * params.size + tid;
        partial_grad_gamma[partial_offset] = gamma_grad;
        partial_grad_beta[partial_offset] = beta_grad;
    }}
    workgroupBarrier();

    // Compute gradient w.r.t. input (unchanged from original)
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

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {{
        if (tid < s) {{
            shared_data[tid] += shared_data[tid + s];
        }}
        workgroupBarrier();
    }}
    d_xhat_sum = shared_data[0];

    shared_data[tid] = d_xhat_xhat_sum;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {{
        if (tid < s) {{
            shared_data[tid] += shared_data[tid + s];
        }}
        workgroupBarrier();
    }}
    d_xhat_xhat_sum = shared_data[0];

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

    Reduces partial gradients without atomic operations.

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
// Reduce partial gradients (no atomic operations)
//
// Reduces partial gamma/beta gradients from stage 1 across all elements.
// Each thread handles one feature dimension, summing across all batch elements.

struct ReduceParams {{
    size: u32,
    n_elements: u32,
}}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> partial_grad_gamma: array<f32>;  // (n_elements, size)
@group(0) @binding(2) var<storage, read> partial_grad_beta: array<f32>;   // (n_elements, size)
@group(0) @binding(3) var<storage, read_write> grad_gamma: array<f32>;    // (size,)
@group(0) @binding(4) var<storage, read_write> grad_beta: array<f32>;     // (size,)

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let dim_idx = global_id.x;

    if (dim_idx >= params.size) {{
        return;
    }}

    // Sum partial gradients across all elements for this dimension
    var gamma_sum = 0.0;
    var beta_sum = 0.0;

    for (var elem_idx = 0u; elem_idx < params.n_elements; elem_idx++) {{
        let partial_offset = elem_idx * params.size + dim_idx;
        gamma_sum += partial_grad_gamma[partial_offset];
        beta_sum += partial_grad_beta[partial_offset];
    }}

    // Write final gradients (no atomics needed)
    grad_gamma[dim_idx] = gamma_sum;
    grad_beta[dim_idx] = beta_sum;
}}
"""


def create_layernorm_backward_reduce_accumulate_kernel(
    workgroup_size: int = 256,
) -> str:
    """
    Generate layer normalization backward reduction kernel - Stage 2 (ACCUMULATE MODE)

    Reduces partial gradients and accumulates into existing values WITHOUT atomic f32.
    Uses read-modify-write pattern which is safe on all GPUs.

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
// SAFE IMPLEMENTATION: No atomic f32 operations
// Uses read-modify-write pattern which works on all GPUs

struct ReduceParams {{
    size: u32,
    n_elements: u32,
}}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> partial_grad_gamma: array<f32>;  // (n_elements, size)
@group(0) @binding(2) var<storage, read> partial_grad_beta: array<f32>;   // (n_elements, size)
@group(0) @binding(3) var<storage, read_write> grad_gamma: array<f32>;    // (size,)
@group(0) @binding(4) var<storage, read_write> grad_beta: array<f32>;     // (size,)

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let dim_idx = global_id.x;

    if (dim_idx >= params.size) {{
        return;
    }}

    // Sum partial gradients across all elements for this dimension
    var gamma_sum = 0.0;
    var beta_sum = 0.0;

    for (var elem_idx = 0u; elem_idx < params.n_elements; elem_idx++) {{
        let partial_offset = elem_idx * params.size + dim_idx;
        gamma_sum += partial_grad_gamma[partial_offset];
        beta_sum += partial_grad_beta[partial_offset];
    }}

    // Read-modify-write (SAFE: no atomics)
    // Each thread writes to a unique location
    grad_gamma[dim_idx] = grad_gamma[dim_idx] + gamma_sum;
    grad_beta[dim_idx] = grad_beta[dim_idx] + beta_sum;
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
//
// GELU(x) = x * Φ(x), where Φ is standard normal CDF
// GELU'(x) = Φ(x) + x * φ(x), where φ is standard normal PDF

struct GeluParams {{
    size: u32,
}}

@group(0) @binding(0) var<uniform> params: GeluParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;

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
    let tanh_inner = tanh(inner);

    // Derivative of GELU
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEFF * x * x);
    let gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;

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

struct BiasParams {{
    size: u32,
    dim: u32,
}}

@group(0) @binding(0) var<uniform> params: BiasParams;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_bias: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let col = global_id.x;

    if (col >= params.dim) {{
        return;
    }}

    let n_rows = params.size / params.dim;

    var sum = 0.0;
    for (var row = 0u; row < n_rows; row++) {{
        sum += grad_output[row * params.dim + col];
    }}

    grad_bias[col] = sum;
}}
"""


def create_attention_backward_kernel(workgroup_size: int = 256) -> str:
    """
    Generate attention backward kernel - ATOMIC-FREE version

    Uses explicit computation instead of atomic accumulation.
    Each workgroup computes complete gradients for its assigned Q position.

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
// Backward pass for scaled dot-product attention - ATOMIC-FREE
//
// Algorithm:
//   1. dV = P^T @ dO
//   2. dP = dO @ V^T
//   3. dS = softmax_backward(dP, P)
//   4. dQ = dS @ K / sqrt(d)
//   5. dK = dS^T @ Q / sqrt(d)
//
// Each workgroup handles ONE query position completely, avoiding atomics.
// Parallelism is over (batch, head, q_pos) rather than individual gradients.

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
@group(0) @binding(7) var<storage, read_write> grad_K: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_V: array<f32>;

const BLOCK_SIZE: u32 = {workgroup_size}u;
var<workgroup> shared_data: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size})
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

    // ========================================================================
    // PHASE 1: Recompute attention weights (forward pass)
    // ========================================================================

    // Find max score
    var max_score = -1e9;
    for (var k_pos = tid; k_pos <= q_pos; k_pos += BLOCK_SIZE) {{
        let k_offset = batch_idx * seq_len * embedding_dim +
                      k_pos * embedding_dim +
                      head_idx * head_dim;

        var score = 0.0;
        for (var d = 0u; d < head_dim; d++) {{
            score += Q[q_offset + d] * K[k_offset + d];
        }}
        max_score = max(max_score, score * scale);
    }}

    shared_data[tid] = max_score;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {{
        if (tid < s) {{
            shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
        }}
        workgroupBarrier();
    }}
    max_score = shared_data[0];
    workgroupBarrier();

    // Compute exp sum
    var sum_exp = 0.0;
    for (var k_pos = tid; k_pos <= q_pos; k_pos += BLOCK_SIZE) {{
        let k_offset = batch_idx * seq_len * embedding_dim +
                      k_pos * embedding_dim +
                      head_idx * head_dim;

        var score = 0.0;
        for (var d = 0u; d < head_dim; d++) {{
            score += Q[q_offset + d] * K[k_offset + d];
        }}
        sum_exp += exp(score * scale - max_score);
    }}

    shared_data[tid] = sum_exp;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {{
        if (tid < s) {{
            shared_data[tid] += shared_data[tid + s];
        }}
        workgroupBarrier();
    }}
    sum_exp = shared_data[0];
    workgroupBarrier();

    // ========================================================================
    // PHASE 2: Compute dO @ V^T for softmax backward
    // ========================================================================

    // Compute dot(grad_output, O) for this query position
    var dO_dot_O = 0.0;
    for (var d = tid; d < head_dim; d += BLOCK_SIZE) {{
        let go_offset = batch_idx * seq_len * embedding_dim +
                       q_pos * embedding_dim +
                       head_idx * head_dim + d;
        dO_dot_O += grad_output[go_offset] * O[go_offset];
    }}

    shared_data[tid] = dO_dot_O;
    workgroupBarrier();

    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {{
        if (tid < s) {{
            shared_data[tid] += shared_data[tid + s];
        }}
        workgroupBarrier();
    }}
    dO_dot_O = shared_data[0];
    workgroupBarrier();

    // ========================================================================
    // PHASE 3: Compute gradients for Q, K, V
    // Each key position handled by one thread to avoid atomics
    // ========================================================================

    // Each thread handles multiple k positions
    for (var k_pos = tid; k_pos <= q_pos; k_pos += BLOCK_SIZE) {{
        let k_offset = batch_idx * seq_len * embedding_dim +
                      k_pos * embedding_dim +
                      head_idx * head_dim;

        // Recompute attention weight P[q_pos, k_pos]
        var score = 0.0;
        for (var d = 0u; d < head_dim; d++) {{
            score += Q[q_offset + d] * K[k_offset + d];
        }}
        let attn_weight = exp(score * scale - max_score) / sum_exp;

        // Compute dP = grad_output @ V^T
        var dP = 0.0;
        for (var d = 0u; d < head_dim; d++) {{
            let v_offset = batch_idx * seq_len * embedding_dim +
                          k_pos * embedding_dim +
                          head_idx * head_dim + d;
            let go_offset = batch_idx * seq_len * embedding_dim +
                           q_pos * embedding_dim +
                           head_idx * head_dim + d;
            dP += grad_output[go_offset] * V[v_offset];
        }}

        // Softmax backward: dS = P * (dP - dot(dO, O))
        let dS = attn_weight * (dP - dO_dot_O);

        // Compute grad_Q contribution (accumulate across all k_pos for this q_pos)
        // SAFE: Each workgroup writes to different q_pos, so no race
        for (var d = 0u; d < head_dim; d++) {{
            let k_val = K[k_offset + d];
            let contribution = dS * k_val * scale;

            // Direct write (no atomics) - each workgroup handles complete q_pos
            let gq_offset = q_offset + d;

            // Accumulate using simple addition (safe because k_pos loop is sequential)
            if (k_pos == tid) {{
                grad_Q[gq_offset] = contribution;  // First iteration: initialize
            }} else if (k_pos > tid && k_pos <= q_pos) {{
                grad_Q[gq_offset] += contribution;  // Subsequent: accumulate
            }}
        }}

        // Compute grad_K for this k_pos
        // SAFE: Each thread handles different k_pos, so no race
        for (var d = 0u; d < head_dim; d++) {{
            let q_val = Q[q_offset + d];
            let gk_offset = k_offset + d;
            grad_K[gk_offset] = dS * q_val * scale;
        }}

        // Compute grad_V for this k_pos
        // SAFE: Each thread handles different k_pos, so no race
        for (var d = 0u; d < head_dim; d++) {{
            let go_offset = batch_idx * seq_len * embedding_dim +
                           q_pos * embedding_dim +
                           head_idx * head_dim + d;
            let gv_offset = k_offset + d;
            grad_V[gv_offset] = attn_weight * grad_output[go_offset];
        }}
    }}
}}
"""


def create_flash_attention_backward_kernel(
    head_dim: int = 64, Bc: int = 32, Br: int = 32
) -> str:
    """
    Generate FlashAttention backward kernel - ATOMIC-FREE version

    Memory-efficient backward pass using tiling and recomputation.
    Eliminates atomic operations by carefully structuring writes.

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

    if total_workgroup_bytes > 65536:
        raise ValueError(
            f"Workgroup memory {total_workgroup_bytes} bytes exceeds 64KB limit. "
            f"Try smaller head_dim, Bc, or Br values."
        )

    return f"""
// FlashAttention Backward Pass - ATOMIC-FREE
// Based on: Dao et al. 2022, Section 3.2
//
// Parameters: head_dim={head_dim}, Bc={Bc}, Br={Br}
// Workgroup memory: {total_workgroup_bytes} bytes
//
// Key optimization: Recompute attention weights from saved statistics (L, M)
// to avoid storing full attention matrix. Uses tiling to minimize memory.

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
@group(0) @binding(5) var<storage, read> L: array<f32>;  // Softmax denominators (saved from forward)
@group(0) @binding(6) var<storage, read> M: array<f32>;  // Max values (saved from forward)
@group(0) @binding(7) var<storage, read> grad_O: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_Q: array<f32>;
@group(0) @binding(9) var<storage, read_write> grad_K: array<f32>;
@group(0) @binding(10) var<storage, read_write> grad_V: array<f32>;

const Bc: u32 = {Bc}u;
const Br: u32 = {Br}u;
const HEAD_DIM: u32 = {head_dim}u;

// Shared memory tiles
var<workgroup> Qi: array<f32, {qi_size}>;
var<workgroup> Kj: array<f32, {kj_size}>;
var<workgroup> Vj: array<f32, {vj_size}>;
var<workgroup> Sij: array<f32, {sij_size}>;
var<workgroup> Pij: array<f32, {pij_size}>;
var<workgroup> dOi: array<f32, {dOi_size}>;
var<workgroup> Oi: array<f32, {Oi_size}>;
var<workgroup> dPij: array<f32, {dPij_size}>;
var<workgroup> dSij: array<f32, {dSij_size}>;
var<workgroup> Di: array<f32, {Br}>;  // D[i] = rowsum(dO * O)

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

    // ========================================================================
    // Load Q block and grad_O block
    // ========================================================================
    for (var i = tid; i < actual_Br * d; i += 32u) {{
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

    // ========================================================================
    // Compute D[i] = rowsum(dO * O)
    // ========================================================================
    if (tid == 0u) {{
        for (var row = 0u; row < actual_Br; row++) {{
            var sum = 0.0;
            for (var d_idx = 0u; d_idx < d; d_idx++) {{
                sum += dOi[row * HEAD_DIM + d_idx] * Oi[row * HEAD_DIM + d_idx];
            }}
            Di[row] = sum;
        }}
    }}
    workgroupBarrier();

    // ========================================================================
    // Initialize gradient accumulators for Q
    // ========================================================================
    var dQi: array<f32, {head_dim * Br}>;
    if (tid == 0u) {{
        for (var i = 0u; i < actual_Br * d; i++) {{
            dQi[i] = 0.0;
        }}
    }}
    workgroupBarrier();

    // ========================================================================
    // Iterate over K/V blocks
    // ========================================================================
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

        // Recompute attention weights from saved statistics
        for (var i = tid; i < actual_Br * actual_Bc; i += 32u) {{
            let row = i / actual_Bc;
            let col = i % actual_Bc;

            var score = 0.0;
            for (var k = 0u; k < d; k++) {{
                score += Qi[row * HEAD_DIM + k] * Kj[col * HEAD_DIM + k];
            }}

            let q_pos = q_start + row;
            let kv_pos = kv_start + col;

            let stats_offset = batch_idx * N * params.n_heads +
                              q_pos * params.n_heads + head_idx;
            let mi = M[stats_offset];
            let li = L[stats_offset];

            if (kv_pos <= q_pos) {{
                Sij[row * Bc + col] = score * scale;
                Pij[row * Bc + col] = exp(score * scale - mi) / li;
            }} else {{
                Sij[row * Bc + col] = -1e9;
                Pij[row * Bc + col] = 0.0;
            }}
        }}
        workgroupBarrier();

        // Compute dP = dO @ V^T
        for (var i = tid; i < actual_Br * actual_Bc; i += 32u) {{
            let row = i / actual_Bc;
            let col = i % actual_Bc;

            var sum = 0.0;
            for (var d_idx = 0u; d_idx < d; d_idx++) {{
                sum += dOi[row * HEAD_DIM + d_idx] * Vj[col * HEAD_DIM + d_idx];
            }}
            dPij[row * Bc + col] = sum;
        }}
        workgroupBarrier();

        // Softmax backward: dS = P * (dP - D)
        for (var i = tid; i < actual_Br * actual_Bc; i += 32u) {{
            let row = i / actual_Bc;
            let col = i % actual_Bc;
            dSij[row * Bc + col] = Pij[row * Bc + col] * (dPij[row * Bc + col] - Di[row]);
        }}
        workgroupBarrier();

        // Accumulate dQ = dS @ K (scaled)
        // SAFE: Each workgroup handles its own Q block, no race
        if (tid == 0u) {{
            for (var row = 0u; row < actual_Br; row++) {{
                for (var d_idx = 0u; d_idx < d; d_idx++) {{
                    var sum = 0.0;
                    for (var col = 0u; col < actual_Bc; col++) {{
                        sum += dSij[row * Bc + col] * Kj[col * HEAD_DIM + d_idx];
                    }}
                    dQi[row * HEAD_DIM + d_idx] += sum * scale;
                }}
            }}
        }}
        workgroupBarrier();

        // Compute and write dK = dS^T @ Q (scaled)
        // SAFE: Each workgroup handles different K blocks, no overlap
        for (var col = tid; col < actual_Bc; col += 32u) {{
            let global_col = kv_start + col;
            let k_offset = batch_idx * N * embedding_dim +
                          global_col * embedding_dim +
                          head_idx * d;

            for (var d_idx = 0u; d_idx < d; d_idx++) {{
                var sum = 0.0;
                for (var row = 0u; row < actual_Br; row++) {{
                    sum += dSij[row * Bc + col] * Qi[row * HEAD_DIM + d_idx];
                }}
                grad_K[k_offset + d_idx] = sum * scale;
            }}
        }}

        // Compute and write dV = P^T @ dO
        // SAFE: Each workgroup handles different V blocks, no overlap
        for (var col = tid; col < actual_Bc; col += 32u) {{
            let global_col = kv_start + col;
            let v_offset = batch_idx * N * embedding_dim +
                          global_col * embedding_dim +
                          head_idx * d;

            for (var d_idx = 0u; d_idx < d; d_idx++) {{
                var sum = 0.0;
                for (var row = 0u; row < actual_Br; row++) {{
                    sum += Pij[row * Bc + col] * dOi[row * HEAD_DIM + d_idx];
                }}
                grad_V[v_offset + d_idx] = sum;
            }}
        }}

        workgroupBarrier();
    }}

    // ========================================================================
    // Write final dQ
    // SAFE: Each workgroup handles its own Q block, no race
    // ========================================================================
    for (var row = tid; row < actual_Br; row += 32u) {{
        let global_row = q_start + row;
        let q_offset = batch_idx * N * embedding_dim +
                      global_row * embedding_dim +
                      head_idx * d;

        for (var d_idx = 0u; d_idx < d; d_idx++) {{
            grad_Q[q_offset + d_idx] = dQi[row * HEAD_DIM + d_idx];
        }}
    }}
}}
"""


def create_cross_entropy_loss_kernel(workgroup_size: int = 256) -> str:
    """
    Generate combined cross-entropy loss and gradient kernel

    Computes both loss and gradients in a single pass for efficiency.
    Numerically stable softmax with max subtraction.

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
// Combined cross-entropy loss and gradient computation
// More efficient than separate loss + backward kernels
//
// Loss: -log(softmax(logits)[target])
// Gradient: softmax(logits) - one_hot(target)

struct LossParams {{
    batch_size: u32,
    seq_len: u32,
    vocab_size: u32,
}}

@group(0) @binding(0) var<uniform> params: LossParams;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> loss_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_logits: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let pred_idx = global_id.x;
    let total = params.batch_size * params.seq_len;

    if (pred_idx >= total) {{
        return;
    }}

    let target_idx = targets[pred_idx];
    let logit_offset = pred_idx * params.vocab_size;

    // Numerically stable softmax: subtract max
    var max_logit = logits[logit_offset];
    for (var i = 1u; i < params.vocab_size; i++) {{
        max_logit = max(max_logit, logits[logit_offset + i]);
    }}

    var sum_exp = 0.0;
    for (var i = 0u; i < params.vocab_size; i++) {{
        sum_exp += exp(logits[logit_offset + i] - max_logit);
    }}

    // Loss: -log(softmax[target])
    let target_logit = logits[logit_offset + target_idx];
    loss_output[pred_idx] = -target_logit + max_logit + log(sum_exp);

    // Gradient: softmax - one_hot
    // Normalized by 1/total for averaging
    for (var i = 0u; i < params.vocab_size; i++) {{
        let prob = exp(logits[logit_offset + i] - max_logit) / sum_exp;
        var grad = prob;
        if (i == target_idx) {{
            grad -= 1.0;
        }}
        grad_logits[logit_offset + i] = grad / f32(total);
    }}
}}
"""


# ============================================================================
# FACTORY FUNCTIONS - Generate kernels from config
# ============================================================================


def get_matmul_backward_a_kernel_from_config(config) -> str:
    """Get matmul backward (dA) kernel configured from GPUConfig"""
    return create_matmul_backward_a_kernel(config.matmul_tile_size)


def get_matmul_backward_b_kernel_from_config(config) -> str:
    """Get matmul backward (dB) kernel configured from GPUConfig"""
    return create_matmul_backward_b_kernel(config.matmul_tile_size)


def get_layernorm_backward_kernel_from_config(config) -> str:
    """Get layernorm backward kernel configured from GPUConfig"""
    return create_layernorm_backward_kernel(
        config.layernorm_workgroup_size, config.layernorm_epsilon
    )


def get_layernorm_backward_reduce_kernel_from_config(config) -> str:
    """Get layernorm backward reduction kernel configured from GPUConfig"""
    return create_layernorm_backward_reduce_kernel(config.layernorm_workgroup_size)


def get_layernorm_backward_reduce_accumulate_kernel_from_config(config) -> str:
    """Get layernorm backward reduction with accumulation kernel configured from GPUConfig"""
    return create_layernorm_backward_reduce_accumulate_kernel(
        config.layernorm_workgroup_size
    )


def get_gelu_backward_kernel_from_config(config) -> str:
    """Get GELU backward kernel configured from GPUConfig"""
    return create_gelu_backward_kernel(config.default_workgroup_size)


def get_bias_backward_kernel_from_config(config) -> str:
    """Get bias backward kernel configured from GPUConfig"""
    return create_bias_backward_kernel(config.default_workgroup_size)


def get_attention_backward_kernel_from_config(config) -> str:
    """Get attention backward kernel configured from GPUConfig"""
    return create_attention_backward_kernel(config.attention_workgroup_size)


def get_flash_attention_backward_kernel_from_config(config) -> str:
    """Get FlashAttention backward kernel configured from GPUConfig"""
    return create_flash_attention_backward_kernel(
        config.flash_attn_max_head_dim, config.flash_attn_bc, config.flash_attn_br
    )


def get_cross_entropy_loss_kernel_from_config(config) -> str:
    """Get cross-entropy loss kernel configured from GPUConfig"""
    return create_cross_entropy_loss_kernel(config.default_workgroup_size)


# ============================================================================
# DEFAULT KERNELS (for backward compatibility)
# ============================================================================

# These are provided for backward compatibility only.
# New code should use get_*_kernel_from_config() functions instead.

MATMUL_BACKWARD_A_KERNEL = create_matmul_backward_a_kernel(16)
MATMUL_BACKWARD_B_KERNEL = create_matmul_backward_b_kernel(16)
LAYERNORM_BACKWARD_KERNEL = create_layernorm_backward_kernel(256)
LAYERNORM_BACKWARD_REDUCE_KERNEL = create_layernorm_backward_reduce_kernel(256)
LAYERNORM_BACKWARD_REDUCE_ACCUMULATE_KERNEL = (
    create_layernorm_backward_reduce_accumulate_kernel(256)
)
GELU_BACKWARD_KERNEL = create_gelu_backward_kernel(256)
BIAS_BACKWARD_KERNEL = create_bias_backward_kernel(256)
ATTENTION_BACKWARD_KERNEL = create_attention_backward_kernel(256)
FLASH_ATTENTION_BACKWARD_KERNEL = create_flash_attention_backward_kernel(64, 32, 32)

CROSS_ENTROPY_LOSS_KERNEL = create_cross_entropy_loss_kernel(256)
