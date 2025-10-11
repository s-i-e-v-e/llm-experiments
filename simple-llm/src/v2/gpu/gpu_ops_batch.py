"""Command batching"""

import numpy as np
from gpu_kernels_backward import (
    BIAS_BACKWARD_KERNEL,
    GELU_BACKWARD_KERNEL,
    LAYERNORM_BACKWARD_KERNEL,
    MATMUL_BACKWARD_A_KERNEL,
    MATMUL_BACKWARD_B_KERNEL,
)
from gpu_kernels_forward import (
    BIAS_ADD_KERNEL,
    GELU_KERNEL,
    LAYERNORM_KERNEL,
    RESIDUAL_ADD_KERNEL,
    TILED_MATMUL_KERNEL,
)
from gpu_ops import _add_compute_to_batch_internal
from gpu_types import BatchState, GPUBuffer1D, GPUBuffer2D, PipelineCache

# ============================================================================
# FORWARD PASS BATCH OPERATIONS
# ============================================================================


def batch_add_matmul(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    A: GPUBuffer2D,
    B: GPUBuffer2D,
    C: GPUBuffer2D,
) -> None:
    """Add matmul to batch"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Dimension mismatch: {K} != {K2}"

    params = np.array([M, K, N], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        TILED_MATMUL_KERNEL,
        params,
        [A, B, C],
        min((N + 15) // 16, 65535),
        min((M + 15) // 16, 65535),
        1,
    )


def batch_add_layernorm(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    beta: GPUBuffer1D,
    output: GPUBuffer2D,
) -> None:
    """Add layernorm to batch"""
    n_elements, size = input_buf.shape

    params = np.array([size, n_elements], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        LAYERNORM_KERNEL,
        params,
        [input_buf, gamma, beta, output],
        n_elements,
    )


def batch_add_gelu(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """Add GELU to batch"""
    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        GELU_KERNEL,
        params,
        [input_buf, output],
        (total_size + 255) // 256,
    )


def batch_add_bias_add(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    bias: GPUBuffer1D,
    output: GPUBuffer2D,
) -> None:
    """Add bias addition to batch"""
    n_elements, dim = input_buf.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        BIAS_ADD_KERNEL,
        params,
        [input_buf, bias, output],
        (total_size + 255) // 256,
    )


def batch_add_residual(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_a: GPUBuffer2D,
    input_b: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """Add residual connection to batch"""
    total_size = input_a.size

    params = np.array([total_size], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        RESIDUAL_ADD_KERNEL,
        params,
        [input_a, input_b, output],
        (total_size + 255) // 256,
    )


def batch_add_copy(
    batch_state: BatchState, source: GPUBuffer2D, dest: GPUBuffer2D
) -> None:
    """Add buffer copy operation to batch"""
    assert source.size == dest.size, (
        f"Buffer sizes must match: {source.size} != {dest.size}"
    )

    if batch_state.encoder is None:
        raise RuntimeError("Must call create_command_batch before adding operations")

    batch_state.encoder.copy_buffer_to_buffer(
        source.buffer, 0, dest.buffer, 0, source.size * 4
    )
    batch_state.operation_count += 1


# ============================================================================
# BACKWARD PASS BATCH OPERATIONS
# ============================================================================


def batch_add_matmul_backward_a(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    grad_C: GPUBuffer2D,
    B: GPUBuffer2D,
    grad_A: GPUBuffer2D,
) -> None:
    """Add matmul backward A to batch"""
    M, N = grad_C.shape
    K, N2 = B.shape
    assert N == N2, f"Dimension mismatch: {N} != {N2}"

    params = np.array([M, K, N], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        MATMUL_BACKWARD_A_KERNEL,
        params,
        [grad_C, B, grad_A],
        (K + 15) // 16,
        (M + 15) // 16,
        1,
    )


def batch_add_matmul_backward_b(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    A: GPUBuffer2D,
    grad_C: GPUBuffer2D,
    grad_B: GPUBuffer2D,
) -> None:
    """Add matmul backward B to batch"""
    M, K = A.shape
    M2, N = grad_C.shape
    assert M == M2, f"Dimension mismatch: {M} != {M2}"

    params = np.array([M, K, N], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        MATMUL_BACKWARD_B_KERNEL,
        params,
        [A, grad_C, grad_B],
        (N + 15) // 16,
        (K + 15) // 16,
        1,
    )


def batch_add_layernorm_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
    grad_gamma: GPUBuffer1D,
    grad_beta: GPUBuffer1D,
) -> None:
    """Add layernorm backward to batch"""
    n_elements, size = input_buf.shape

    # Zero out gamma and beta gradients (kernel accumulates)
    zero_data = np.zeros(grad_gamma.size, dtype=np.float32)
    batch_state.device.wgpu_device.queue.write_buffer(grad_gamma.buffer, 0, zero_data)
    batch_state.device.wgpu_device.queue.write_buffer(grad_beta.buffer, 0, zero_data)

    params = np.array([size, n_elements], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        LAYERNORM_BACKWARD_KERNEL,
        params,
        [input_buf, gamma, grad_output, grad_input, grad_gamma, grad_beta],
        n_elements,
    )


def batch_add_gelu_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
) -> None:
    """Add GELU backward to batch"""
    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        GELU_BACKWARD_KERNEL,
        params,
        [input_buf, grad_output, grad_input],
        (total_size + 255) // 256,
    )


def batch_add_bias_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    grad_output: GPUBuffer2D,
    grad_bias: GPUBuffer1D,
) -> None:
    """Add bias backward to batch"""
    n_elements, dim = grad_output.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        BIAS_BACKWARD_KERNEL,
        params,
        [grad_output, grad_bias],
        (dim + 255) // 256,
    )


# ============================================================================
# BATCH SUBMISSION
# ============================================================================


def submit_batch(batch_state: BatchState) -> None:
    """Submit all batched operations"""
    if batch_state.encoder is None:
        raise RuntimeError("Batch already submitted or not initialized")

    command_buffer = batch_state.encoder.finish()
    batch_state.device.wgpu_device.queue.submit([command_buffer])

    # Clear encoder to prevent reuse
    batch_state.encoder = None
    batch_state.retained_buffers.clear()
