"""Backward pass operations - individual kernel dispatches

MUTATION SEMANTICS:
- All backward operations MUTATE their output gradient buffers
- Accumulation buffers (grad_gamma, grad_beta, grad_bias) are automatically
  zeroed before kernel dispatch - caller does NOT need to pre-zero them
- Workspace buffers (grad_input, etc.) are NOT auto-zeroed - caller must
  ensure these are properly initialized if reusing buffers
- Functions return None to signal mutation
"""

import numpy as np
from gpu_buffer import clear_buffer
from gpu_kernels_backward import (
    BIAS_BACKWARD_KERNEL,
    GELU_BACKWARD_KERNEL,
    LAYERNORM_BACKWARD_KERNEL,
    MATMUL_BACKWARD_A_KERNEL,
    MATMUL_BACKWARD_B_KERNEL,
)
from gpu_ops import dispatch_simple_compute
from gpu_types import GPUBuffer1D, GPUBuffer2D, PipelineCache

# ============================================================================
# BACKWARD PASS OPERATIONS
# ============================================================================


def run_matmul_backward_a(
    pipeline_cache: PipelineCache,
    grad_C: GPUBuffer2D,
    B: GPUBuffer2D,
    grad_A: GPUBuffer2D,
) -> None:
    """Compute gradient w.r.t. A: grad_A = grad_C @ B^T (mutation).

    This function MUTATES grad_A by writing the computed gradients.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
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

    if grad_A.shape != (M, K):
        raise ValueError(
            f"grad_A shape {grad_A.shape} doesn't match expected ({M}, {K})"
        )

    params = np.array([M, K, N], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        MATMUL_BACKWARD_A_KERNEL,
        params,
        [grad_C, B, grad_A],
        (K + 15) // 16,
        (M + 15) // 16,
        1,
    )


def run_matmul_backward_b(
    pipeline_cache: PipelineCache,
    A: GPUBuffer2D,
    grad_C: GPUBuffer2D,
    grad_B: GPUBuffer2D,
) -> None:
    """Compute gradient w.r.t. B: grad_B = A^T @ grad_C (mutation).

    This function MUTATES grad_B by writing the computed gradients.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
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

    if grad_B.shape != (K, N):
        raise ValueError(
            f"grad_B shape {grad_B.shape} doesn't match expected ({K}, {N})"
        )

    params = np.array([M, K, N], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        MATMUL_BACKWARD_B_KERNEL,
        params,
        [A, grad_C, grad_B],
        (N + 15) // 16,
        (K + 15) // 16,
        1,
    )


def run_layernorm_backward(
    pipelinecache: PipelineCache,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
    grad_gamma: GPUBuffer1D,
    grad_beta: GPUBuffer1D,
    accumulate: bool = False,
) -> None:
    """
    Backward pass for layer normalization (mutation)

    This function MUTATES grad_input, grad_gamma, and grad_beta by writing gradients.
    Returns None to signal mutation.

    IMPLEMENTATION: Uses two-stage reduction to avoid race conditions:
    - Stage 1: Each workgroup computes partial gamma/beta gradients
    - Stage 2: Reduction kernel sums partial gradients into final result

    Args:
        pipelinecache: Pipeline cache for kernel compilation
        input_buf: Input from forward pass (n_elements, size)
        gamma: Scale parameters from forward pass (size,)
        grad_output: Gradient of loss w.r.t. output (n_elements, size)
        grad_input: Output gradient w.r.t. input (n_elements, size) - MUTATED
        grad_gamma: Output gradient w.r.t. gamma (size,) - MUTATED
        grad_beta: Output gradient w.r.t. beta (size,) - MUTATED
        accumulate: If False (default), zeros grad_gamma/grad_beta before accumulation.
                   If True, accumulates into existing values using GPU-native atomic operations.
                   Use True for gradient accumulation across mini-batches.
                   Note: grad_input is always overwritten (never accumulated).

    Raises:
        ValueError: If shapes are incompatible

    Example:
        # Standard usage (single batch)
        run_layernorm_backward(cache, x, gamma, grad_out, grad_in, grad_g, grad_b)

        # Gradient accumulation (multiple mini-batches)
        run_layernorm_backward(cache, x1, gamma, grad_out1, grad_in1, grad_g, grad_b, accumulate=False)
        run_layernorm_backward(cache, x2, gamma, grad_out2, grad_in2, grad_g, grad_b, accumulate=True)
    """
    from gpu_buffer import pool_release_buffer, pool_take_buffer_2d
    from gpu_kernels_backward import (
        LAYERNORM_BACKWARD_REDUCE_ACCUMULATE_KERNEL,
        LAYERNORM_BACKWARD_REDUCE_KERNEL,
    )

    n_elements, size = input_buf.shape

    if n_elements == 0 or size == 0:
        raise ValueError(f"Invalid input_buf shape: {(n_elements, size)}")

    if gamma.shape != (size,):
        raise ValueError(
            f"gamma shape {gamma.shape} doesn't match input size {(size,)}"
        )

    if grad_output.shape != (n_elements, size):
        raise ValueError(
            f"grad_output shape {grad_output.shape} doesn't match input_buf shape {(n_elements, size)}"
        )

    if grad_input.shape != (n_elements, size):
        raise ValueError(
            f"grad_input shape {grad_input.shape} doesn't match input_buf shape {(n_elements, size)}"
        )

    if grad_gamma.shape != (size,):
        raise ValueError(
            f"grad_gamma shape {grad_gamma.shape} doesn't match expected {(size,)}"
        )

    if grad_beta.shape != (size,):
        raise ValueError(
            f"grad_beta shape {grad_beta.shape} doesn't match expected {(size,)}"
        )

    # Allocate temporary buffers for partial gradients
    buffer_pool = (
        pipelinecache.device.buffer_pool
        if hasattr(pipelinecache.device, "buffer_pool")
        else None
    )

    if buffer_pool is not None:
        partial_grad_gamma = pool_take_buffer_2d(buffer_pool, n_elements, size)
        partial_grad_beta = pool_take_buffer_2d(buffer_pool, n_elements, size)
    else:
        # Fallback: create temporary buffers directly
        import numpy as np
        from gpu_buffer import create_gpu_buffer_2d

        partial_grad_gamma = create_gpu_buffer_2d(
            pipelinecache.device,
            n_elements,
            size,
            np.zeros((n_elements, size), dtype=np.float32),
        )
        partial_grad_beta = create_gpu_buffer_2d(
            pipelinecache.device,
            n_elements,
            size,
            np.zeros((n_elements, size), dtype=np.float32),
        )

    # Stage 1: Compute partial gradients (one workgroup per element)
    params = np.array([size, n_elements], dtype=np.uint32)

    dispatch_simple_compute(
        pipelinecache,
        LAYERNORM_BACKWARD_KERNEL,
        params,
        [
            input_buf,
            gamma,
            grad_output,
            grad_input,
            partial_grad_gamma,
            partial_grad_beta,
        ],
        n_elements,  # One workgroup per element
    )

    # Stage 2: Reduce partial gradients into final gamma/beta gradients
    # Choose kernel based on accumulation mode
    if accumulate:
        # Use atomic accumulation kernel (GPU-native, no CPU roundtrip)
        reduction_kernel = LAYERNORM_BACKWARD_REDUCE_ACCUMULATE_KERNEL
    else:
        # Zero first, then use normal reduction
        clear_buffer(grad_gamma)
        clear_buffer(grad_beta)
        reduction_kernel = LAYERNORM_BACKWARD_REDUCE_KERNEL

    params_reduce = np.array([size, n_elements], dtype=np.uint32)
    dispatch_simple_compute(
        pipelinecache,
        reduction_kernel,
        params_reduce,
        [partial_grad_gamma, partial_grad_beta, grad_gamma, grad_beta],
        (size + 255) // 256,
    )

    # Release temporary buffers
    if buffer_pool is not None:
        pool_release_buffer(buffer_pool, partial_grad_gamma)
        pool_release_buffer(buffer_pool, partial_grad_beta)


def run_gelu_backward(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
) -> None:
    """Backward pass for GELU activation (mutation).

    This function MUTATES grad_input by writing the computed gradients.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input from forward pass
        grad_output: Gradient of loss w.r.t. output
        grad_input: Output gradient w.r.t. input (MUTATED)

    Raises:
        ValueError: If shapes are incompatible
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

    params = np.array([total_size], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        GELU_BACKWARD_KERNEL,
        params,
        [input_buf, grad_output, grad_input],
        (total_size + 255) // 256,
    )


def run_bias_backward(
    pipelinecache: PipelineCache,
    grad_output: GPUBuffer2D,
    grad_bias: GPUBuffer1D,
    accumulate: bool = False,
) -> None:
    """
    Backward pass for bias - sum gradients over batch (mutation)

    This function MUTATES grad_bias by writing or accumulating gradients.
    Returns None to signal mutation.

    Args:
        pipelinecache: Pipeline cache for kernel compilation
        grad_output: Gradient of loss w.r.t. output (n_elements, dim)
        grad_bias: Output gradient w.r.t. bias (dim,) - MUTATED
        accumulate: If False (default), zeros grad_bias before accumulation.
                   If True, accumulates into existing grad_bias values.
                   Use True for gradient accumulation across mini-batches.

    Raises:
        ValueError: If shapes are incompatible

    Example:
        # Standard usage (single batch)
        run_bias_backward(cache, grad_out, grad_bias)

        # Gradient accumulation (multiple mini-batches)
        run_bias_backward(cache, grad_out1, grad_bias, accumulate=False)  # First batch
        run_bias_backward(cache, grad_out2, grad_bias, accumulate=True)   # Accumulate
        run_bias_backward(cache, grad_out3, grad_bias, accumulate=True)   # Accumulate
    """
    n_elements, dim = grad_output.shape

    if n_elements == 0 or dim == 0:
        raise ValueError(f"Invalid grad_output shape: {(n_elements, dim)}")

    if grad_bias.shape != (dim,):
        raise ValueError(
            f"grad_bias shape {grad_bias.shape} doesn't match expected {(dim,)}"
        )

    # Zero accumulation buffer only if not accumulating
    if not accumulate:
        clear_buffer(grad_bias)

    total_size = n_elements * dim
    params = np.array([total_size, dim], dtype=np.uint32)

    dispatch_simple_compute(
        pipelinecache,
        BIAS_BACKWARD_KERNEL,
        params,
        [grad_output, grad_bias],
        (dim + 255) // 256,
    )
