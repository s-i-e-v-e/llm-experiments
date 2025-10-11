"""Backward pass operations - individual kernel dispatches"""

import numpy as np
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
    """Compute gradient w.r.t. A: grad_A = grad_C @ B^T"""
    M, N = grad_C.shape
    K, N2 = B.shape
    assert N == N2, f"Dimension mismatch: {N} != {N2}"

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
    """Compute gradient w.r.t. B: grad_B = A^T @ grad_C"""
    M, K = A.shape
    M2, N = grad_C.shape
    assert M == M2, f"Dimension mismatch: {M} != {M2}"

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
    """
    Backward pass for layer normalization.

    IMPORTANT: grad_gamma and grad_beta MUST be pre-zeroed before calling
    this function, as the kernel accumulates gradients.

    Use clear_buffer() to zero them:
        clear_buffer(grad_gamma)
        clear_buffer(grad_beta)
        run_layernorm_backward(...)

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input from forward pass (nelements, size)
        gamma: Scale parameters from forward pass
        grad_output: Gradient of loss w.r.t. output
        grad_input: Output gradient w.r.t. input
        grad_gamma: Output gradient w.r.t. gamma (MUST BE PRE-ZEROED)
        grad_beta: Output gradient w.r.t. beta (MUST BE PRE-ZEROED)
    """
    nelements, size = input_buf.shape
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
    """Backward pass for GELU activation"""
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
    """Backward pass for bias - sum gradients over batch"""
    n_elements, dim = grad_output.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        BIAS_BACKWARD_KERNEL,
        params,
        [grad_output, grad_bias],
        (dim + 255) // 256,
    )
