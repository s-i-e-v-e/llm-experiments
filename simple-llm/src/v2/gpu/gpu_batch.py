"""Command batching"""

import numpy as np

try:
    import wgpu

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None

from gpu_device import get_or_create_pipeline
from gpu_kernels import (
    BIAS_ADD_KERNEL,
    BIAS_BACKWARD_KERNEL,
    GELU_BACKWARD_KERNEL,
    GELU_KERNEL,
    LAYERNORM_BACKWARD_KERNEL,
    LAYERNORM_KERNEL,
    MATMUL_BACKWARD_A_KERNEL,
    MATMUL_BACKWARD_B_KERNEL,
    RESIDUAL_ADD_KERNEL,
    TILED_MATMUL_KERNEL,
)
from gpu_types import BatchState, GPUBuffer

# ============================================================================
# COMMAND BATCH STATE
# ============================================================================


def create_command_batch(device: object, enable_profiling: bool = False) -> BatchState:
    """Create command batch state for batched GPU operations"""
    encoder = device.create_command_encoder()
    return BatchState(
        device=device,
        encoder=encoder,
        retained_buffers=[],
        enable_profiling=enable_profiling,
        operation_count=0,
    )


def batch_add_matmul(
    batch_state: BatchState, A: GPUBuffer, B: GPUBuffer, C: GPUBuffer
) -> BatchState:
    """Add matmul to batch. Returns updated batch_state."""
    M, K = A.shape
    K2, N = B.shape

    params = np.array([M, K, N], dtype=np.uint32)
    params_buffer = batch_state.device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(params_buffer)

    pipeline = get_or_create_pipeline(TILED_MATMUL_KERNEL, batch_state.device)
    bind_group = batch_state.device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {"buffer": A.buffer, "offset": 0, "size": A.size * 4},
            },
            {
                "binding": 2,
                "resource": {"buffer": B.buffer, "offset": 0, "size": B.size * 4},
            },
            {
                "binding": 3,
                "resource": {"buffer": C.buffer, "offset": 0, "size": C.size * 4},
            },
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    # Validate workgroup size against device limits
    workgroups_x = min((N + 15) // 16, 65535)
    workgroups_y = min((M + 15) // 16, 65535)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1)
    compute_pass.end()

    batch_state.operation_count += 1
    return batch_state


def batch_add_layernorm(
    batch_state: BatchState,
    input_buf: GPUBuffer,
    gamma: GPUBuffer,
    beta: GPUBuffer,
    output: GPUBuffer,
) -> BatchState:
    """Add layernorm to batch. Returns updated batch_state."""
    n_elements, size = input_buf.shape

    params = np.array([size, n_elements], dtype=np.uint32)
    params_buffer = batch_state.device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(params_buffer)

    pipeline = get_or_create_pipeline(LAYERNORM_KERNEL, batch_state.device)
    bind_group = batch_state.device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": input_buf.buffer,
                    "offset": 0,
                    "size": input_buf.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": gamma.buffer,
                    "offset": 0,
                    "size": gamma.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": beta.buffer,
                    "offset": 0,
                    "size": beta.size * 4,
                },
            },
            {
                "binding": 4,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(n_elements, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1
    return batch_state


def batch_add_gelu(
    batch_state: BatchState, input_buf: GPUBuffer, output: GPUBuffer
) -> BatchState:
    """Add GELU to batch. Returns updated batch_state."""
    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = batch_state.device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(params_buffer)

    pipeline = get_or_create_pipeline(GELU_KERNEL, batch_state.device)
    bind_group = batch_state.device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": input_buf.buffer,
                    "offset": 0,
                    "size": input_buf.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1
    return batch_state


def batch_add_bias_add(
    batch_state: BatchState, input_buf: GPUBuffer, bias: GPUBuffer, output: GPUBuffer
) -> BatchState:
    """Add bias addition to batch. Returns updated batch_state."""
    n_elements, dim = input_buf.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    params_buffer = batch_state.device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(params_buffer)

    pipeline = get_or_create_pipeline(BIAS_ADD_KERNEL, batch_state.device)
    bind_group = batch_state.device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": input_buf.buffer,
                    "offset": 0,
                    "size": input_buf.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": bias.buffer,
                    "offset": 0,
                    "size": bias.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1
    return batch_state


def batch_add_residual(
    batch_state: BatchState, input_a: GPUBuffer, input_b: GPUBuffer, output: GPUBuffer
) -> BatchState:
    """Add residual connection to batch. Returns updated batch_state."""
    total_size = input_a.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = batch_state.device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(params_buffer)

    pipeline = get_or_create_pipeline(RESIDUAL_ADD_KERNEL, batch_state.device)
    bind_group = batch_state.device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": input_a.buffer,
                    "offset": 0,
                    "size": input_a.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": input_b.buffer,
                    "offset": 0,
                    "size": input_b.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1
    return batch_state


def batch_add_copy(
    batch_state: BatchState, source: GPUBuffer, dest: GPUBuffer
) -> BatchState:
    """Add buffer copy operation to batch. Returns updated batch_state."""
    assert source.size == dest.size, (
        f"Buffer sizes must match: {source.size} != {dest.size}"
    )

    if batch_state.encoder is None:
        raise RuntimeError("Must call create_command_batch() before adding operations")

    batch_state.encoder.copy_buffer_to_buffer(
        source.buffer,
        0,
        dest.buffer,
        0,
        source.size * 4,  # 4 bytes per float32
    )

    batch_state.operation_count += 1
    return batch_state


# ============================================================================
# BACKWARD PASS BATCH OPERATIONS
# ============================================================================


def batch_add_matmul_backward_a(
    batch_state: BatchState, grad_C: GPUBuffer, B: GPUBuffer, grad_A: GPUBuffer
) -> BatchState:
    """Add matmul backward w.r.t. A: grad_A = grad_C @ B^T"""
    M, N = grad_C.shape
    K2, N2 = B.shape
    assert N == N2, f"Dimension mismatch: {N} != {N2}"

    params = np.array([M, K2, N], dtype=np.uint32)
    params_buffer = batch_state.device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(params_buffer)

    pipeline = get_or_create_pipeline(MATMUL_BACKWARD_A_KERNEL, batch_state.device)
    bind_group = batch_state.device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": grad_C.buffer,
                    "offset": 0,
                    "size": grad_C.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {"buffer": B.buffer, "offset": 0, "size": B.size * 4},
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": grad_A.buffer,
                    "offset": 0,
                    "size": grad_A.size * 4,
                },
            },
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
    return batch_state


def batch_add_matmul_backward_b(
    batch_state: BatchState, A: GPUBuffer, grad_C: GPUBuffer, grad_B: GPUBuffer
) -> BatchState:
    """Add matmul backward w.r.t. B: grad_B = A^T @ grad_C"""
    M, K = A.shape
    M2, N = grad_C.shape
    assert M == M2, f"Dimension mismatch: {M} != {M2}"

    params = np.array([M, K, N], dtype=np.uint32)
    params_buffer = batch_state.device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(params_buffer)

    pipeline = get_or_create_pipeline(MATMUL_BACKWARD_B_KERNEL, batch_state.device)
    bind_group = batch_state.device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {"buffer": A.buffer, "offset": 0, "size": A.size * 4},
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": grad_C.buffer,
                    "offset": 0,
                    "size": grad_C.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": grad_B.buffer,
                    "offset": 0,
                    "size": grad_B.size * 4,
                },
            },
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
    return batch_state


def batch_add_gelu_backward(
    batch_state: BatchState,
    input_buf: GPUBuffer,
    grad_output: GPUBuffer,
    grad_input: GPUBuffer,
) -> BatchState:
    """Add GELU backward to batch"""
    total_size = input_buf.size
    assert grad_output.size == total_size and grad_input.size == total_size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = batch_state.device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(params_buffer)

    pipeline = get_or_create_pipeline(GELU_BACKWARD_KERNEL, batch_state.device)
    bind_group = batch_state.device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": input_buf.buffer,
                    "offset": 0,
                    "size": input_buf.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": grad_output.buffer,
                    "offset": 0,
                    "size": grad_output.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": grad_input.buffer,
                    "offset": 0,
                    "size": grad_input.size * 4,
                },
            },
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1
    return batch_state


def batch_add_layernorm_backward(
    batch_state: BatchState,
    input_buf: GPUBuffer,
    gamma: GPUBuffer,
    grad_output: GPUBuffer,
    grad_input: GPUBuffer,
    grad_gamma: GPUBuffer,
    grad_beta: GPUBuffer,
) -> BatchState:
    """Add layernorm backward to batch"""
    n_elements, size = input_buf.shape

    params = np.array([size, n_elements], dtype=np.uint32)
    params_buffer = batch_state.device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(params_buffer)

    pipeline = get_or_create_pipeline(LAYERNORM_BACKWARD_KERNEL, batch_state.device)
    bind_group = batch_state.device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": input_buf.buffer,
                    "offset": 0,
                    "size": input_buf.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": gamma.buffer,
                    "offset": 0,
                    "size": gamma.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": grad_output.buffer,
                    "offset": 0,
                    "size": grad_output.size * 4,
                },
            },
            {
                "binding": 4,
                "resource": {
                    "buffer": grad_input.buffer,
                    "offset": 0,
                    "size": grad_input.size * 4,
                },
            },
            {
                "binding": 5,
                "resource": {
                    "buffer": grad_gamma.buffer,
                    "offset": 0,
                    "size": grad_gamma.size * 4,
                },
            },
            {
                "binding": 6,
                "resource": {
                    "buffer": grad_beta.buffer,
                    "offset": 0,
                    "size": grad_beta.size * 4,
                },
            },
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(n_elements, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1
    return batch_state


def batch_add_bias_backward(
    batch_state: BatchState, grad_output: GPUBuffer, grad_bias: GPUBuffer
) -> BatchState:
    """Add bias backward to batch - sums gradients over batch dimension"""
    n_elements, dim = grad_output.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    params_buffer = batch_state.device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(params_buffer)

    pipeline = get_or_create_pipeline(BIAS_BACKWARD_KERNEL, batch_state.device)
    bind_group = batch_state.device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": grad_output.buffer,
                    "offset": 0,
                    "size": grad_output.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": grad_bias.buffer,
                    "offset": 0,
                    "size": grad_bias.size * 4,
                },
            },
        ],
    )

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((dim + 255) // 256, 1, 1)
    compute_pass.end()

    batch_state.operation_count += 1
    return batch_state


# ============================================================================
# BATCH SUBMISSION
# ============================================================================


def batch_submit(batch_state: BatchState) -> BatchState:
    """Execute all batched operations and release retained resources. Returns batch_state."""
    if batch_state.encoder is not None:
        batch_state.device.queue.submit([batch_state.encoder.finish()])

        if batch_state.enable_profiling and batch_state.operation_count > 0:
            print(
                f"Batched {batch_state.operation_count} operations in single submission"
            )

        batch_state.encoder = None

    batch_state.retained_buffers = []
    batch_state.operation_count = 0

    return batch_state
