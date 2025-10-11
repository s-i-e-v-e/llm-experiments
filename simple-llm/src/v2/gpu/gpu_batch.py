"""Command batching"""

from typing import List

import numpy as np

try:
    import wgpu

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None

from gpu_device import BindGroupEntry, create_bind_group_entries, get_or_create_pipeline
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
from gpu_types import BatchState, Device, GPUBuffer2D, PipelineCache

# ============================================================================
# COMMAND BATCH STATE
# ============================================================================


def create_command_batch(device: Device, enable_profiling: bool = False) -> BatchState:
    """Create command batch state for batched GPU operations"""
    encoder = device.wgpu_device.create_command_encoder()
    return BatchState(
        device=device,
        encoder=encoder,
        retained_buffers=[],
        enable_profiling=enable_profiling,
        operation_count=0,
    )


def _create_and_retain_uniform_buffer(
    batch_state: BatchState, data: np.ndarray
) -> object:
    """Helper: Create uniform buffer and add to retained list"""
    buffer = batch_state.device.wgpu_device.create_buffer_with_data(
        data=data, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(buffer)
    return buffer


def _create_bind_group_for_batch(
    batch_state: BatchState, pipeline: object, entries: List[BindGroupEntry]
) -> object:
    """Helper: Create bind group using type-safe entries"""
    return batch_state.device.wgpu_device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=create_bind_group_entries(entries),
    )


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
    params_buffer = _create_and_retain_uniform_buffer(batch_state, params)

    pipeline = get_or_create_pipeline(pipeline_cache, TILED_MATMUL_KERNEL)

    bind_group = _create_bind_group_for_batch(
        batch_state,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, A.buffer, 0, A.size * 4),
            BindGroupEntry(2, B.buffer, 0, B.size * 4),
            BindGroupEntry(3, C.buffer, 0, C.size * 4),
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    workgroups_x = min((N + 15) // 16, 65535)
    workgroups_y = min((M + 15) // 16, 65535)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1)
    compute_pass.end()

    batch_state.operation_count += 1


def batch_add_layernorm(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer2D,
    beta: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """Add layernorm to batch"""
    n_elements, size = input_buf.shape

    params = np.array([size, n_elements], dtype=np.uint32)
    params_buffer = _create_and_retain_uniform_buffer(batch_state, params)

    pipeline = get_or_create_pipeline(pipeline_cache, LAYERNORM_KERNEL)

    bind_group = _create_bind_group_for_batch(
        batch_state,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, input_buf.buffer, 0, input_buf.size * 4),
            BindGroupEntry(2, gamma.buffer, 0, gamma.size * 4),
            BindGroupEntry(3, beta.buffer, 0, beta.size * 4),
            BindGroupEntry(4, output.buffer, 0, output.size * 4),
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(n_elements, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1


def batch_add_gelu(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """Add GELU to batch"""
    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = _create_and_retain_uniform_buffer(batch_state, params)

    pipeline = get_or_create_pipeline(pipeline_cache, GELU_KERNEL)

    bind_group = _create_bind_group_for_batch(
        batch_state,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, input_buf.buffer, 0, input_buf.size * 4),
            BindGroupEntry(2, output.buffer, 0, output.size * 4),
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1


def batch_add_bias_add(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    bias: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """Add bias addition to batch"""
    n_elements, dim = input_buf.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    params_buffer = _create_and_retain_uniform_buffer(batch_state, params)

    pipeline = get_or_create_pipeline(pipeline_cache, BIAS_ADD_KERNEL)

    bind_group = _create_bind_group_for_batch(
        batch_state,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, input_buf.buffer, 0, input_buf.size * 4),
            BindGroupEntry(2, bias.buffer, 0, bias.size * 4),
            BindGroupEntry(3, output.buffer, 0, output.size * 4),
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1


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
    params_buffer = _create_and_retain_uniform_buffer(batch_state, params)

    pipeline = get_or_create_pipeline(pipeline_cache, RESIDUAL_ADD_KERNEL)

    bind_group = _create_bind_group_for_batch(
        batch_state,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, input_a.buffer, 0, input_a.size * 4),
            BindGroupEntry(2, input_b.buffer, 0, input_b.size * 4),
            BindGroupEntry(3, output.buffer, 0, output.size * 4),
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1


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
    """Add matmul backward w.r.t. A: grad_A = grad_C @ B^T"""
    M, N = grad_C.shape
    K2, N2 = B.shape
    assert N == N2, f"Dimension mismatch: {N} != {N2}"

    params = np.array([M, K2, N], dtype=np.uint32)
    params_buffer = _create_and_retain_uniform_buffer(batch_state, params)

    pipeline = get_or_create_pipeline(pipeline_cache, MATMUL_BACKWARD_A_KERNEL)

    bind_group = _create_bind_group_for_batch(
        batch_state,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, grad_C.buffer, 0, grad_C.size * 4),
            BindGroupEntry(2, B.buffer, 0, B.size * 4),
            BindGroupEntry(3, grad_A.buffer, 0, grad_A.size * 4),
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    workgroups_x = min((K2 + 15) // 16, 65535)
    workgroups_y = min((M + 15) // 16, 65535)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1)
    compute_pass.end()

    batch_state.operation_count += 1


def batch_add_matmul_backward_b(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    A: GPUBuffer2D,
    grad_C: GPUBuffer2D,
    grad_B: GPUBuffer2D,
) -> None:
    """Add matmul backward w.r.t. B: grad_B = A^T @ grad_C"""
    M, K = A.shape
    M2, N = grad_C.shape
    assert M == M2, f"Dimension mismatch: {M} != {M2}"

    params = np.array([M, K, N], dtype=np.uint32)
    params_buffer = _create_and_retain_uniform_buffer(batch_state, params)

    pipeline = get_or_create_pipeline(pipeline_cache, MATMUL_BACKWARD_B_KERNEL)

    bind_group = _create_bind_group_for_batch(
        batch_state,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, A.buffer, 0, A.size * 4),
            BindGroupEntry(2, grad_C.buffer, 0, grad_C.size * 4),
            BindGroupEntry(3, grad_B.buffer, 0, grad_B.size * 4),
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    workgroups_x = min((N + 15) // 16, 65535)
    workgroups_y = min((K + 15) // 16, 65535)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1)
    compute_pass.end()

    batch_state.operation_count += 1


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
    params_buffer = _create_and_retain_uniform_buffer(batch_state, params)

    pipeline = get_or_create_pipeline(pipeline_cache, GELU_BACKWARD_KERNEL)

    bind_group = _create_bind_group_for_batch(
        batch_state,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, input_buf.buffer, 0, input_buf.size * 4),
            BindGroupEntry(2, grad_output.buffer, 0, grad_output.size * 4),
            BindGroupEntry(3, grad_input.buffer, 0, grad_input.size * 4),
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1


def batch_add_layernorm_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer2D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
    grad_gamma: GPUBuffer2D,
    grad_beta: GPUBuffer2D,
) -> None:
    """Add layernorm backward to batch"""
    n_elements, size = input_buf.shape

    params = np.array([size, n_elements], dtype=np.uint32)
    params_buffer = _create_and_retain_uniform_buffer(batch_state, params)

    pipeline = get_or_create_pipeline(pipeline_cache, LAYERNORM_BACKWARD_KERNEL)

    bind_group = _create_bind_group_for_batch(
        batch_state,
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

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(n_elements, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1


def batch_add_bias_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    grad_output: GPUBuffer2D,
    grad_bias: GPUBuffer2D,
) -> None:
    """Add bias backward to batch - sums gradients over batch dimension"""
    n_elements, dim = grad_output.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    params_buffer = _create_and_retain_uniform_buffer(batch_state, params)

    pipeline = get_or_create_pipeline(pipeline_cache, BIAS_BACKWARD_KERNEL)

    bind_group = _create_bind_group_for_batch(
        batch_state,
        pipeline,
        [
            BindGroupEntry(0, params_buffer, 0, params.nbytes),
            BindGroupEntry(1, grad_output.buffer, 0, grad_output.size * 4),
            BindGroupEntry(2, grad_bias.buffer, 0, grad_bias.size * 4),
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((dim + 255) // 256, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1


# ============================================================================
# BATCH SUBMISSION
# ============================================================================


def batch_submit(batch_state: BatchState) -> None:
    """Execute all batched operations and release retained resources"""
    if batch_state.encoder is not None:
        batch_state.device.wgpu_device.queue.submit([batch_state.encoder.finish()])

        if batch_state.enable_profiling and batch_state.operation_count > 0:
            print(
                f"Batched {batch_state.operation_count} operations in single submission"
            )

        batch_state.encoder = None
        batch_state.retained_buffers = []
        batch_state.operation_count = 0
