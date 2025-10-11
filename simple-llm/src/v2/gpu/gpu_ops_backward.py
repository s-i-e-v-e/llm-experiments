"""Backward pass operations - individual kernel dispatches"""

import numpy as np
from gpu_device import (
    BindGroupEntry,
    get_or_create_pipeline,
)
from gpu_kernels_backward import (
    BIAS_BACKWARD_KERNEL,
    GELU_BACKWARD_KERNEL,
    LAYERNORM_BACKWARD_KERNEL,
    MATMUL_BACKWARD_A_KERNEL,
    MATMUL_BACKWARD_B_KERNEL,
)
from gpu_ops import _create_bind_group, _create_uniform_buffer, _dispatch_compute
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
    params_buffer = _create_uniform_buffer(pipeline_cache, params)

    pipeline = get_or_create_pipeline(pipeline_cache, MATMUL_BACKWARD_A_KERNEL)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, grad_C.buffer, 0, grad_C.size * 4),
            BindGroupEntry(2, B.buffer, 0, B.size * 4),
            BindGroupEntry(3, grad_A.buffer, 0, grad_A.size * 4),
        ],
    )

    _dispatch_compute(
        pipeline_cache,
        pipeline,
        bind_group,
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
    params_buffer = _create_uniform_buffer(pipeline_cache, params)

    pipeline = get_or_create_pipeline(pipeline_cache, MATMUL_BACKWARD_B_KERNEL)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, A.buffer, 0, A.size * 4),
            BindGroupEntry(2, grad_C.buffer, 0, grad_C.size * 4),
            BindGroupEntry(3, grad_B.buffer, 0, grad_B.size * 4),
        ],
    )

    _dispatch_compute(
        pipeline_cache,
        pipeline,
        bind_group,
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
    """Backward pass for layer normalization"""
    n_elements, size = input_buf.shape

    # Zero out gamma and beta gradients (kernel accumulates)
    zero_data = np.zeros(grad_gamma.size, dtype=np.float32)
    pipeline_cache.device.wgpu_device.queue.write_buffer(
        grad_gamma.buffer, 0, zero_data
    )
    pipeline_cache.device.wgpu_device.queue.write_buffer(grad_beta.buffer, 0, zero_data)

    params = np.array([size, n_elements], dtype=np.uint32)
    params_buffer = _create_uniform_buffer(pipeline_cache, params)

    pipeline = get_or_create_pipeline(pipeline_cache, LAYERNORM_BACKWARD_KERNEL)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, input_buf.buffer, 0, input_buf.size * 4),
            BindGroupEntry(2, gamma.buffer, 0, gamma.size * 4),
            BindGroupEntry(3, grad_output.buffer, 0, grad_output.size * 4),
            BindGroupEntry(4, grad_input.buffer, 0, grad_input.size * 4),
            BindGroupEntry(5, grad_gamma.buffer, 0, grad_gamma.size * 4),
            BindGroupEntry(6, grad_beta.buffer, 0, grad_beta.size * 4),
        ],
    )

    _dispatch_compute(pipeline_cache, pipeline, bind_group, n_elements)


def run_gelu_backward(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
) -> None:
    """Backward pass for GELU activation"""
    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = _create_uniform_buffer(pipeline_cache, params)

    pipeline = get_or_create_pipeline(pipeline_cache, GELU_BACKWARD_KERNEL)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, input_buf.buffer, 0, input_buf.size * 4),
            BindGroupEntry(2, grad_output.buffer, 0, grad_output.size * 4),
            BindGroupEntry(3, grad_input.buffer, 0, grad_input.size * 4),
        ],
    )

    _dispatch_compute(pipeline_cache, pipeline, bind_group, (total_size + 255) // 256)


def run_bias_backward(
    pipeline_cache: PipelineCache,
    grad_output: GPUBuffer2D,
    grad_bias: GPUBuffer1D,
) -> None:
    """Backward pass for bias - sum gradients over batch"""
    n_elements, dim = grad_output.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    params_buffer = _create_uniform_buffer(pipeline_cache, params)

    pipeline = get_or_create_pipeline(pipeline_cache, BIAS_BACKWARD_KERNEL)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, grad_output.buffer, 0, grad_output.size * 4),
            BindGroupEntry(2, grad_bias.buffer, 0, grad_bias.size * 4),
        ],
    )

    _dispatch_compute(pipeline_cache, pipeline, bind_group, (dim + 255) // 256)
