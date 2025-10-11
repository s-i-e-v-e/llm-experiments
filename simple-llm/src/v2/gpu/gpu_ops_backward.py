"""Backward pass operations - individual kernel dispatches"""

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
        AssertionError: If dimensions are incompatible
        ValueError: If shapes are invalid
    """
    M, N = grad_C.shape
    K, N2 = B.shape

    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"Invalid dimensions: grad_C=({M}, {N}), B=({K}, {N2})")

    if grad_A.shape != (M, K):
        raise ValueError(
            f"grad_A shape {grad_A.shape} doesn't match expected ({M}, {K})"
        )

    assert N == N2, f"Dimension mismatch: grad_C.shape[1]={N} != B.shape[1]={N2}"

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
        AssertionError: If dimensions are incompatible
        ValueError: If shapes are invalid
    """
    M, K = A.shape
    M2, N = grad_C.shape

    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError(f"Invalid dimensions: A=({M}, {K}), grad_C=({M2}, {N})")

    if grad_B.shape != (K, N):
        raise ValueError(
            f"grad_B shape {grad_B.shape} doesn't match expected ({K}, {N})"
        )

    assert M == M2, f"Dimension mismatch: A.shape[0]={M} != grad_C.shape[0]={M2}"

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
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
    grad_gamma: GPUBuffer1D,
    grad_beta: GPUBuffer1D,
) -> None:
    """Backward pass for layer normalization (mutation).

    This function MUTATES grad_input, grad_gamma, and grad_beta by writing gradients.
    Returns None to signal mutation.

    NOTE: This function automatically zeros grad_gamma and grad_beta before
    accumulating gradients, so caller does not need to pre-zero them.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input from forward pass (nelements, size)
        gamma: Scale parameters from forward pass (size,)
        grad_output: Gradient of loss w.r.t. output (nelements, size)
        grad_input: Output gradient w.r.t. input (nelements, size) (MUTATED)
        grad_gamma: Output gradient w.r.t. gamma (size,) (MUTATED, auto-zeroed)
        grad_beta: Output gradient w.r.t. beta (size,) (MUTATED, auto-zeroed)

    Raises:
        ValueError: If shapes are incompatible
    """
    nelements, size = input_buf.shape

    if nelements <= 0 or size <= 0:
        raise ValueError(f"Invalid input_buf shape: ({nelements}, {size})")

    if gamma.shape != (size,):
        raise ValueError(
            f"gamma shape {gamma.shape} doesn't match input size ({size},)"
        )

    if grad_output.shape != (nelements, size):
        raise ValueError(
            f"grad_output shape {grad_output.shape} doesn't match input_buf shape ({nelements}, {size})"
        )

    if grad_input.shape != (nelements, size):
        raise ValueError(
            f"grad_input shape {grad_input.shape} doesn't match input_buf shape ({nelements}, {size})"
        )

    if grad_gamma.shape != (size,):
        raise ValueError(
            f"grad_gamma shape {grad_gamma.shape} doesn't match expected ({size},)"
        )

    if grad_beta.shape != (size,):
        raise ValueError(
            f"grad_beta shape {grad_beta.shape} doesn't match expected ({size},)"
        )

    # Zero accumulation buffers (safer API - caller doesn't need to remember)
    clear_buffer(grad_gamma)
    clear_buffer(grad_beta)

    params = np.array([size, nelements], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        LAYERNORM_BACKWARD_KERNEL,
        params,
        [input_buf, gamma, grad_output, grad_input, grad_gamma, grad_beta],
        nelements,
    )


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
    pipeline_cache: PipelineCache,
    grad_output: GPUBuffer2D,
    grad_bias: GPUBuffer1D,
) -> None:
    """Backward pass for bias - sum gradients over batch (mutation).

    This function MUTATES grad_bias by writing accumulated gradients.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        grad_output: Gradient of loss w.r.t. output (n_elements, dim)
        grad_bias: Output gradient w.r.t. bias (dim,) (MUTATED)

    Raises:
        ValueError: If shapes are incompatible
    """
    n_elements, dim = grad_output.shape

    if n_elements <= 0 or dim <= 0:
        raise ValueError(f"Invalid grad_output shape: ({n_elements}, {dim})")

    if grad_bias.shape != (dim,):
        raise ValueError(
            f"grad_bias shape {grad_bias.shape} doesn't match expected ({dim},)"
        )

    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        BIAS_BACKWARD_KERNEL,
        params,
        [grad_output, grad_bias],
        (dim + 255) // 256,
    )
