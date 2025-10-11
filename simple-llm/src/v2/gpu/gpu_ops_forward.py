"""Forward pass operations - individual kernel dispatches"""

import numpy as np
from gpu_device import (
    BindGroupEntry,
    get_or_create_pipeline,
)
from gpu_kernels_forward import (
    BIAS_ADD_KERNEL,
    GELU_KERNEL,
    LAYERNORM_KERNEL,
    RESIDUAL_ADD_KERNEL,
    TILED_MATMUL_KERNEL,
)
from gpu_ops import _create_bind_group, _create_uniform_buffer, _dispatch_compute
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
    params_buffer = _create_uniform_buffer(pipeline_cache, params)

    pipeline = get_or_create_pipeline(pipeline_cache, TILED_MATMUL_KERNEL)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, A.buffer, 0, A.size * 4),
            BindGroupEntry(2, B.buffer, 0, B.size * 4),
            BindGroupEntry(3, C.buffer, 0, C.size * 4),
        ],
    )

    _dispatch_compute(
        pipeline_cache,
        pipeline,
        bind_group,
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
    params_buffer = _create_uniform_buffer(pipeline_cache, params)

    pipeline = get_or_create_pipeline(pipeline_cache, LAYERNORM_KERNEL)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, input_buf.buffer, 0, input_buf.size * 4),
            BindGroupEntry(2, gamma.buffer, 0, gamma.size * 4),
            BindGroupEntry(3, beta.buffer, 0, beta.size * 4),
            BindGroupEntry(4, output.buffer, 0, output.size * 4),
        ],
    )

    _dispatch_compute(pipeline_cache, pipeline, bind_group, n_elements)


def run_gelu(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """Apply GELU activation"""
    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = _create_uniform_buffer(pipeline_cache, params)

    pipeline = get_or_create_pipeline(pipeline_cache, GELU_KERNEL)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, input_buf.buffer, 0, input_buf.size * 4),
            BindGroupEntry(2, output.buffer, 0, output.size * 4),
        ],
    )

    _dispatch_compute(pipeline_cache, pipeline, bind_group, (total_size + 255) // 256)


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
    params_buffer = _create_uniform_buffer(pipeline_cache, params)

    pipeline = get_or_create_pipeline(pipeline_cache, BIAS_ADD_KERNEL)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, input_buf.buffer, 0, input_buf.size * 4),
            BindGroupEntry(2, bias.buffer, 0, bias.size * 4),
            BindGroupEntry(3, output.buffer, 0, output.size * 4),
        ],
    )

    _dispatch_compute(pipeline_cache, pipeline, bind_group, (total_size + 255) // 256)


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
    params_buffer = _create_uniform_buffer(pipeline_cache, params)

    pipeline = get_or_create_pipeline(pipeline_cache, RESIDUAL_ADD_KERNEL)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, input_a.buffer, 0, input_a.size * 4),
            BindGroupEntry(2, input_b.buffer, 0, input_b.size * 4),
            BindGroupEntry(3, output.buffer, 0, output.size * 4),
        ],
    )

    _dispatch_compute(pipeline_cache, pipeline, bind_group, (total_size + 255) // 256)
