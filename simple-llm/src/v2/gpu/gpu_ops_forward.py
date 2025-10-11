"""Forward pass operations - individual kernel dispatches"""

import numpy as np
from gpu_kernels_forward import (
    BIAS_ADD_KERNEL,
    GELU_KERNEL,
    LAYERNORM_KERNEL,
    RESIDUAL_ADD_KERNEL,
    TILED_MATMUL_KERNEL,
)
from gpu_ops import dispatch_simple_compute
from gpu_types import GPUBuffer1D, GPUBuffer2D, PipelineCache

# ============================================================================
# FORWARD PASS OPERATIONS
# ============================================================================


def run_matmul(
    pipeline_cache: PipelineCache,
    A: GPUBuffer2D,
    B: GPUBuffer2D,
    C: GPUBuffer2D,
) -> None:
    """
    Matrix multiplication: C = A @ B.

    Uses tiled algorithm for efficiency. This function MUTATES C.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        A: Input matrix (M, K)
        B: Input matrix (K, N)
        C: Output matrix (M, N) - MUTATED
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Dimension mismatch: {K} != {K2}"

    params = np.array([M, K, N], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        TILED_MATMUL_KERNEL,
        params,
        [A, B, C],
        min((N + 15) // 16, 65535),
        min((M + 15) // 16, 65535),
        1,
    )


def run_layernorm(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    beta: GPUBuffer1D,
    output: GPUBuffer2D,
) -> None:
    """
    Layer normalization with affine transformation.

    This function MUTATES output.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input tensor (n_elements, size)
        gamma: Scale parameters (size,)
        beta: Shift parameters (size,)
        output: Output tensor (n_elements, size) - MUTATED
    """
    n_elements, size = input_buf.shape

    params = np.array([size, n_elements], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        LAYERNORM_KERNEL,
        params,
        [input_buf, gamma, beta, output],
        n_elements,
    )


def run_gelu(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """
    GELU activation function.

    This function MUTATES output.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input tensor
        output: Output tensor - MUTATED
    """
    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        GELU_KERNEL,
        params,
        [input_buf, output],
        (total_size + 255) // 256,
    )


def run_bias_add(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    bias: GPUBuffer1D,
    output: GPUBuffer2D,
) -> None:
    """
    Add bias to input tensor (broadcast over first dimension).

    This function MUTATES output.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input tensor (n_elements, dim)
        bias: Bias vector (dim,)
        output: Output tensor (n_elements, dim) - MUTATED
    """
    n_elements, dim = input_buf.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        BIAS_ADD_KERNEL,
        params,
        [input_buf, bias, output],
        (total_size + 255) // 256,
    )


def run_residual_add(
    pipeline_cache: PipelineCache,
    input_a: GPUBuffer2D,
    input_b: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """
    Element-wise addition for residual connections.

    This function MUTATES output.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_a: First input tensor
        input_b: Second input tensor
        output: Output tensor - MUTATED
    """
    total_size = input_a.size

    params = np.array([total_size], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        RESIDUAL_ADD_KERNEL,
        params,
        [input_a, input_b, output],
        (total_size + 255) // 256,
    )
