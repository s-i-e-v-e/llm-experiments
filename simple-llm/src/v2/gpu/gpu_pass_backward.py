import numpy as np
from gpu_buffer import (
    clear_buffer,
    create_gpu_buffer_2d,
    pool_release_buffer,
    pool_take_buffer_2d,
)
from gpu_types import (
    BatchState,
    GPUBuffer1D,
    GPUBuffer2D,
    PipelineCache,
)

from .gpu_kernels import (
    get_attention_backward_kernel,
    get_bias_backward_kernel,
    get_flash_attention_backward_kernel,
    get_gelu_backward_kernel,
    get_layernorm_backward_kernel,
    get_layernorm_backward_reduce_accumulate_kernel,
    get_layernorm_backward_reduce_kernel,
    get_matmul_backward_a_kernel,
    get_matmul_backward_b_kernel,
)
from .gpu_ops import (
    _add_compute_to_batch_internal,
    validate_buffer_shape_1d,
    validate_buffer_shape_2d,
)

# ============================================================================
# BACKWARD PASS OPERATIONS
# ============================================================================

"""Backward pass operations - individual kernel dispatches

MUTATION SEMANTICS:
- All backward operations MUTATE their output gradient buffers
- Accumulation buffers (grad_gamma, grad_beta, grad_bias) are automatically
  zeroed before kernel dispatch - caller does NOT need to pre-zero them
- Workspace buffers (grad_input, etc.) are NOT auto-zeroed - caller must
  ensure these are properly initialized if reusing buffers
- Functions return None to signal mutation
"""


def matmul_backward_a(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    grad_C: GPUBuffer2D,
    B: GPUBuffer2D,
    grad_A: GPUBuffer2D,
) -> None:
    """Compute gradient w.r.t. A: grad_A = grad_C @ B^T (mutation).

    This function MUTATES grad_A by writing the computed gradients.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        grad_C: Gradient of loss w.r.t. C (M, N)
        B: Forward pass B matrix (K, N)
        grad_A: Output gradient w.r.t. A (M, K) (MUTATED)

    Raises:
        ValueError: If dimensions are incompatible or invalid
    """
    M, N = grad_C.shape
    K, N2 = B.shape

    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"Invalid dimensions: grad_C=({M}, {N}), B=({K}, {N2})")

    if N != N2:
        raise ValueError(f"Dimension mismatch: grad_C.shape[1]={N} != B.shape[1]={N2}")

    config = pipeline_cache.device.config
    TILE_SIZE = config.matmul_tile_size

    validate_buffer_shape_2d(grad_A, (M, K), "grad_A")

    params = np.array([M, K, N], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_matmul_backward_a_kernel(config),
        params,
        [grad_C, B, grad_A],
        (K + TILE_SIZE - 1) // TILE_SIZE,
        (M + TILE_SIZE - 1) // TILE_SIZE,
        1,
    )


def matmul_backward_b(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    A: GPUBuffer2D,
    grad_C: GPUBuffer2D,
    grad_B: GPUBuffer2D,
) -> None:
    """Compute gradient w.r.t. B: grad_B = A^T @ grad_C (mutation).

    This function MUTATES grad_B by writing the computed gradients.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        A: Forward pass A matrix (M, K)
        grad_C: Gradient of loss w.r.t. C (M, N)
        grad_B: Output gradient w.r.t. B (K, N) (MUTATED)

    Raises:
        ValueError: If dimensions are incompatible or invalid
    """
    M, K = A.shape
    M2, N = grad_C.shape

    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError(f"Invalid dimensions: A=({M}, {K}), grad_C=({M2}, {N})")

    if M != M2:
        raise ValueError(f"Dimension mismatch: A.shape[0]={M} != grad_C.shape[0]={M2}")

    validate_buffer_shape_2d(grad_B, (K, N), "grad_B")

    config = pipeline_cache.device.config
    TILE_SIZE = config.matmul_tile_size
    params = np.array([M, K, N], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_matmul_backward_b_kernel(config),
        params,
        [A, grad_C, grad_B],
        (N + TILE_SIZE - 1) // TILE_SIZE,
        (K + TILE_SIZE - 1) // TILE_SIZE,
        1,
    )


def layernorm_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
    grad_gamma: GPUBuffer1D,
    grad_beta: GPUBuffer1D,
    accumulate: bool = False,
) -> None:
    """
    Backward pass for layer normalization

    Uses two-stage reduction to avoid race conditions:
    - Stage 1: Each workgroup computes partial gamma/beta gradients
    - Stage 2: Reduction kernel sums partial gradients into final result

    Example:
        # Gradient accumulation (multiple mini-batches)
        run_layernorm_backward(cache, x1, gamma, grad_out1, grad_in1, grad_g, grad_b, accumulate=False)
        run_layernorm_backward(cache, x2, gamma, grad_out2, grad_in2, grad_g, grad_b, accumulate=True)

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_buf: Input from forward pass (n_elements, size)
        gamma: Scale parameters from forward pass (size,)
        grad_output: Gradient of loss w.r.t. output (n_elements, size)
        grad_input: Output gradient w.r.t. input (n_elements, size) (MUTATED)
        grad_gamma: Output gradient w.r.t. gamma (size,) (MUTATED)
        grad_beta: Output gradient w.r.t. beta (size,) (MUTATED)
        accumulate: If False (default), zeros grad_gamma/grad_beta before operation.
                   If True, accumulates into existing values.

    Raises:
        ValueError: If buffer shapes don't match
    """
    n_elements, size = input_buf.shape
    config = pipeline_cache.device.config

    if n_elements == 0 or size == 0:
        raise ValueError(f"Invalid input_buf shape: {(n_elements, size)}")

    validate_buffer_shape_1d(gamma, size, "gamma")
    validate_buffer_shape_2d(grad_output, (n_elements, size), "grad_output")
    validate_buffer_shape_2d(grad_input, (n_elements, size), "grad_input")
    validate_buffer_shape_1d(grad_gamma, size, "grad_gamma")
    validate_buffer_shape_1d(grad_beta, size, "grad_beta")

    # Zero accumulation buffers
    clear_buffer(grad_gamma)
    clear_buffer(grad_beta)

    # Allocate temporary buffers for partial gradients
    buffer_pool = (
        pipeline_cache.device.buffer_pool
        if hasattr(pipeline_cache.device, "buffer_pool")
        else None
    )

    if buffer_pool is not None:
        partial_grad_gamma = pool_take_buffer_2d(buffer_pool, n_elements, size)
        partial_grad_beta = pool_take_buffer_2d(buffer_pool, n_elements, size)
    else:
        # Fallback: create temporary buffers directly
        partial_grad_gamma = create_gpu_buffer_2d(
            pipeline_cache.device,
            n_elements,
            size,
            np.zeros((n_elements, size), dtype=np.float32),
        )
        partial_grad_beta = create_gpu_buffer_2d(
            pipeline_cache.device,
            n_elements,
            size,
            np.zeros((n_elements, size), dtype=np.float32),
        )

    # Stage 1: Compute partial gradients
    params = np.array([size, n_elements], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_layernorm_backward_kernel(config),
        params,
        [input_buf, gamma, grad_output, grad_input, grad_gamma, grad_beta],
        n_elements,
    )

    # Stage 2: Reduce partial gradients
    if accumulate:
        # Use atomic accumulation kernel (still needed for accumulation mode)
        reduction_kernel = (get_layernorm_backward_reduce_accumulate_kernel(config),)
    else:
        clear_buffer(grad_gamma)
        clear_buffer(grad_beta)
        reduction_kernel = (get_layernorm_backward_reduce_kernel(config),)

    params_reduce = np.array([size, n_elements], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        reduction_kernel,
        params_reduce,
        [partial_grad_gamma, partial_grad_beta, grad_gamma, grad_beta],
        (size + 255) // 256,
    )

    # Release temporary buffers
    if buffer_pool is not None:
        pool_release_buffer(buffer_pool, partial_grad_gamma)
        pool_release_buffer(buffer_pool, partial_grad_beta)


def gelu_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
) -> None:
    """Backward pass for GELU activation

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_buf: Input from forward pass
        grad_output: Gradient of loss w.r.t. output
        grad_input: Output gradient w.r.t. input (MUTATED)

    Raises:
        ValueError: If buffer shapes don't match
    """
    if input_buf.shape != grad_output.shape:
        raise ValueError(
            f"input_buf shape {input_buf.shape} doesn't match grad_output shape {grad_output.shape}"
        )

    if grad_input.shape != input_buf.shape:
        raise ValueError(
            f"grad_input shape {grad_input.shape} doesn't match input_buf shape {input_buf.shape}"
        )

    total_size = input_buf.size
    config = pipeline_cache.device.config
    params = np.array([total_size], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_gelu_backward_kernel(config),
        params,
        [input_buf, grad_output, grad_input],
        (total_size + 255) // 256,
    )


def bias_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    grad_output: GPUBuffer2D,
    grad_bias: GPUBuffer1D,
    accumulate: bool = False,
) -> None:
    """
    Backward pass for bias - sum gradients over batch

    Example:
        # Gradient accumulation (multiple mini-batches)
        run_bias_backward(cache, grad_out1, grad_bias, accumulate=False)  # First batch
        run_bias_backward(cache, grad_out2, grad_bias, accumulate=True)   # Accumulate
        run_bias_backward(cache, grad_out3, grad_bias, accumulate=True)   # Accumulate

    Args:
        pipelinecache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        grad_output: Gradient of loss w.r.t. output (n_elements, dim) (MUTATED)
        grad_bias: Output gradient w.r.t. bias (dim,)
        accumulate: If False (default), zeros grad_bias before operation.
                   If True, accumulates into existing grad_bias values.

    Raises:
        ValueError: If buffer shapes don't match
    """
    n_elements, dim = grad_output.shape

    if n_elements == 0 or dim == 0:
        raise ValueError(f"Invalid grad_output shape: {(n_elements, dim)}")

    validate_buffer_shape_1d(grad_bias, dim, "grad_bias")

    # Zero accumulation buffer only if not accumulating
    if not accumulate:
        clear_buffer(grad_bias)

    total_size = n_elements * dim
    config = pipeline_cache.device.config
    params = np.array([total_size, dim], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_bias_backward_kernel(config),
        params,
        [grad_output, grad_bias],
        (dim + 255) // 256,
    )


def attention_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    grad_output: GPUBuffer2D,
    Q: GPUBuffer2D,
    K: GPUBuffer2D,
    V: GPUBuffer2D,
    O: GPUBuffer2D,
    grad_Q: GPUBuffer2D,
    grad_K: GPUBuffer2D,
    grad_V: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
) -> None:
    """
    Attention backward pass - ATOMIC-FREE version (mutation)

    Each workgroup handles ONE query position completely, avoiding race conditions.
    """
    if batch_size <= 0 or seq_len <= 0 or n_heads <= 0 or head_dim <= 0:
        raise ValueError(
            f"Invalid dimensions: batch_size={batch_size}, seq_len={seq_len}, "
            f"n_heads={n_heads}, head_dim={head_dim}"
        )

    embedding_dim = n_heads * head_dim
    total_tokens = batch_size * seq_len

    validate_buffer_shape_2d(grad_output, (total_tokens, embedding_dim), "grad_output")
    validate_buffer_shape_2d(Q, (total_tokens, embedding_dim), "Q")
    validate_buffer_shape_2d(K, (total_tokens, embedding_dim), "K")
    validate_buffer_shape_2d(V, (total_tokens, embedding_dim), "V")
    validate_buffer_shape_2d(O, (total_tokens, embedding_dim), "O")
    validate_buffer_shape_2d(grad_Q, (total_tokens, embedding_dim), "grad_Q")
    validate_buffer_shape_2d(grad_K, (total_tokens, embedding_dim), "grad_K")
    validate_buffer_shape_2d(grad_V, (total_tokens, embedding_dim), "grad_V")

    config = pipeline_cache.device.config
    params = np.array([batch_size, seq_len, n_heads, head_dim], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_attention_backward_kernel(config),
        params,
        [grad_output, Q, K, V, O, grad_Q, grad_K, grad_V],
        seq_len,
        n_heads,
        batch_size,
    )


def run_flash_attention_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    Q: GPUBuffer2D,
    K: GPUBuffer2D,
    V: GPUBuffer2D,
    O: GPUBuffer2D,
    L: GPUBuffer1D,
    M: GPUBuffer1D,
    grad_O: GPUBuffer2D,
    grad_Q: GPUBuffer2D,
    grad_K: GPUBuffer2D,
    grad_V: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
) -> None:
    """
    Recomputes attention weights from saved statistics (L, M) to avoid
    materializing full attention matrix.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        grad_O: Gradient w.r.t. output [batch_size*seq_len, n_heads*head_dim]
        Q: Query from forward pass [batch_size*seq_len, n_heads*head_dim]
        K: Key from forward pass [batch_size*seq_len, n_heads*head_dim]
        V: Value from forward pass [batch_size*seq_len, n_heads*head_dim]
        O: Output from forward pass [batch_size*seq_len, n_heads*head_dim]
        L: Softmax normalization from forward [batch_size*seq_len*n_heads]
        M: Max values from forward [batch_size*seq_len*n_heads]
        grad_Q: Gradient w.r.t. Q [batch_size*seq_len, n_heads*head_dim] (MUTATED)
        grad_K: Gradient w.r.t. K [batch_size*seq_len, n_heads*head_dim] (MUTATED)
        grad_V: Gradient w.r.t. V [batch_size*seq_len, n_heads*head_dim] (MUTATED)
        batch_size: Batch size
        seq_len: Sequence length
        n_heads: Number of attention heads
        head_dim: Dimension per head

    Raises:
        ValueError: If buffer shapes don't match or dimensions invalid
    """
    config = pipeline_cache.device.config

    # Validate head_dim against config
    if head_dim > config.flash_attn_max_head_dim:
        raise ValueError(
            f"head_dim {head_dim} exceeds maximum supported by config: "
            f"{config.flash_attn_max_head_dim}"
        )

    if batch_size <= 0 or seq_len <= 0 or n_heads <= 0 or head_dim <= 0:
        raise ValueError(
            f"Invalid dimensions: batch_size={batch_size}, seq_len={seq_len}, "
            f"n_heads={n_heads}, head_dim={head_dim}"
        )

    embedding_dim = n_heads * head_dim
    total_tokens = batch_size * seq_len
    stats_size = batch_size * seq_len * n_heads

    validate_buffer_shape_2d(Q, (total_tokens, embedding_dim), "Q")
    validate_buffer_shape_2d(K, (total_tokens, embedding_dim), "K")
    validate_buffer_shape_2d(V, (total_tokens, embedding_dim), "V")
    validate_buffer_shape_2d(O, (total_tokens, embedding_dim), "O")
    validate_buffer_shape_1d(L, stats_size, "L")
    validate_buffer_shape_1d(M, stats_size, "M")
    validate_buffer_shape_2d(grad_O, (total_tokens, embedding_dim), "grad_O")
    validate_buffer_shape_2d(grad_Q, (total_tokens, embedding_dim), "grad_Q")
    validate_buffer_shape_2d(grad_K, (total_tokens, embedding_dim), "grad_K")
    validate_buffer_shape_2d(grad_V, (total_tokens, embedding_dim), "grad_V")

    params = np.array(
        [
            batch_size,
            seq_len,
            n_heads,
            head_dim,
            config.flash_attn_bc,
            config.flash_attn_br,
        ],
        dtype=np.uint32,
    )

    Br = config.flash_attn_br
    num_q_blocks = (seq_len + Br - 1) // Br

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_flash_attention_backward_kernel(config),
        params,
        [grad_O, Q, K, V, O, L, M, grad_Q, grad_K, grad_V],
        num_q_blocks,
        n_heads,
        batch_size,
    )
