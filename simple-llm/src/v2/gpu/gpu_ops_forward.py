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
    """Execute tiled matrix multiplication: C = A @ B"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible shapes: {A.shape} @ {B.shape}"
    assert C.shape == (M, N), f"Output shape mismatch: {C.shape} != ({M}, {N})"

    params = np.array([M, K, N], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        TILED_MATMUL_KERNEL,
        params,
        [A, B, C],
        (N + 15) // 16,
        (M + 15) // 16,
        1,
    )


def run_layernorm(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    beta: GPUBuffer1D,
    output: GPUBuffer2D,
) -> None:
    """Apply layer normalization"""
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
    """Apply GELU activation"""
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
    """Add bias to matrix (broadcasts over rows)"""
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
    """Element-wise addition for residual connections"""
    total_size = input_a.size
    assert input_a.size == input_b.size == output.size

    params = np.array([total_size], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        RESIDUAL_ADD_KERNEL,
        params,
        [input_a, input_b, output],
        (total_size + 255) // 256,
    )
