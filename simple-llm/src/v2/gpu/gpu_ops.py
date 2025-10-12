"""Kernel dispatches"""

from typing import Dict, List, Tuple

import numpy as np

from .gpu_device import (
    pipeline_get_or_create,
    wgpu,
)
from .gpu_kernels import get_transpose_kernel
from .gpu_types import (
    BatchState,
    BindGroupEntry,
    GPUBindGroup,
    GPUBuffer,
    GPUBuffer1D,
    GPUBuffer2D,
    GPUBufferAny,
    GPUComputePipeline,
    GPUConfig,
    GPUDevice,
    PipelineCache,
)

# ============================================================================
# VALIDATION
# ============================================================================


def validate_buffer_shape_1d(
    buffer: GPUBuffer1D, expected_size: int, name: str
) -> None:
    """Validate 1D buffer has expected size.

    Args:
        buffer: Buffer to validate
        expected_size: Expected size
        name: Buffer name for error messages

    Raises:
        ValueError: If size doesn't match or invalid
    """
    if expected_size <= 0:
        raise ValueError(f"Invalid expected size for {name}: {expected_size}")

    if buffer.shape != (expected_size,):
        raise ValueError(
            f"{name} shape mismatch: got {buffer.shape}, expected ({expected_size},)"
        )


def validate_buffer_shape_2d(
    buffer: GPUBuffer2D, expected_shape: Tuple[int, int], name: str
) -> None:
    """Validate 2D buffer has expected shape.

    Args:
        buffer: Buffer to validate
        expected_shape: Expected (rows, cols)
        name: Buffer name for error messages

    Raises:
        ValueError: If shape doesn't match or dimensions invalid
    """
    if expected_shape[0] <= 0 or expected_shape[1] <= 0:
        raise ValueError(f"Invalid expected shape for {name}: {expected_shape}")

    if buffer.shape != expected_shape:
        raise ValueError(
            f"{name} shape mismatch: got {buffer.shape}, expected {expected_shape}"
        )


def validate_matmul_shapes(
    A: GPUBuffer2D, B: GPUBuffer2D, C: GPUBuffer2D, operation: str
) -> Tuple[int, int, int]:
    """Validate shapes for matrix multiplication operations.

    Args:
        A: First input matrix
        B: Second input matrix
        C: Output matrix
        operation: Operation name for error messages

    Returns:
        Tuple of (M, K, N) dimensions

    Raises:
        ValueError: If shapes are incompatible
    """
    M, K = A.shape
    K2, N = B.shape

    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError(f"{operation}: Invalid dimensions A=({M}, {K}), B=({K2}, {N})")

    if K != K2:
        raise ValueError(
            f"{operation}: Dimension mismatch A.shape[1]={K} != B.shape[0]={K2}"
        )

    if C.shape != (M, N):
        raise ValueError(
            f"{operation}: Output shape {C.shape} doesn't match expected ({M}, {N})"
        )

    return M, K, N


def validate_optimizer_buffers(
    gradients: GPUBufferAny,
    weights: GPUBufferAny,
    m: GPUBufferAny,
    v: GPUBufferAny,
    step: int,
) -> int:
    """Validate optimizer buffer shapes and step.

    Args:
        gradients: Gradient buffer
        weights: Weight buffer
        m: Momentum buffer
        v: Variance buffer
        step: Training step number

    Returns:
        Buffer size

    Raises:
        AssertionError: If shapes don't match
        ValueError: If step is invalid
    """
    if step < 1:
        raise ValueError(f"Step must be >= 1, got {step}")

    assert gradients.size == weights.size, (
        f"Size mismatch: gradients={gradients.size} != weights={weights.size}"
    )
    assert weights.size == m.size == v.size, (
        f"All buffers must have same size: weights={weights.size}, m={m.size}, v={v.size}"
    )

    return weights.size


# ============================================================================
# INFRASTRUCTURE
# ============================================================================


def create_bind_group_entries(entries: List[BindGroupEntry]) -> List[Dict]:
    """Convert typed BindGroupEntry list to wgpu bind group entry format.

    Args:
        entries: List of BindGroupEntry specifications

    Returns:
        New list of dictionaries in wgpu bind group format
    """
    return [
        {
            "binding": entry.binding,
            "resource": {
                "buffer": entry.buffer,
                "offset": entry.offset,
                "size": entry.size,
            },
        }
        for entry in entries
    ]


def INTERNAL__create_uniform_buffer_(
    device: GPUDevice,
    config: GPUConfig,
    pipeline_cache: PipelineCache,
    data: np.ndarray,
) -> GPUBuffer:
    """Internal: Create uniform buffer for parameters.

    Memory management: Buffer is automatically freed by WGPU after GPU finishes
    using it (typically after queue submission completes). No explicit cleanup needed.

    Args:
        pipeline_cache: Pipeline cache state
        data: Numpy array of parameter data

    Returns:
        New WGPU uniform buffer
    """
    return device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM)


def INTERNAL__create_bind_group_internal(
    device: GPUDevice,
    config: GPUConfig,
    pipeline_cache: PipelineCache,
    pipeline: GPUComputePipeline,
    entries: List[BindGroupEntry],
) -> GPUBindGroup:
    """Internal: Create bind group using type-safe entries.

    Args:
        pipeline_cache: Pipeline cache state
        pipeline: Compute pipeline
        entries: Bind group entry specifications

    Returns:
        New WGPU bind group
    """
    return device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=create_bind_group_entries(entries),
    )


def dispatch_compute(
    device: GPUDevice,
    config: GPUConfig,
    pipeline: GPUComputePipeline,
    bind_group: GPUBindGroup,
    workgroups_x: int,
    workgroups_y: int = 1,
    workgroups_z: int = 1,
) -> None:
    """Internal: Create encoder and dispatch compute pass.

    Uiform buffers created in this function are automatically freed
    by WGPU after the queue submission completes. No explicit cleanup needed.

    Args:
        pipeline: Compute pipeline
        bind_group: Bind group
        workgroups_x: Number of workgroups in X
        workgroups_y: Number of workgroups in Y
        workgroups_z: Number of workgroups in Z
    """
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def create_command_batch(
    device: GPUDevice, config: GPUConfig, enable_profiling: bool = False
) -> BatchState:
    """
    Create command batch state for batched GPU operations

    Memory management: Uniform buffers created during batch operations are retained
    in batch_state.retained_buffers to keep them alive until submit_batch is called.

    Args:
        device: GPU device state
        enable_profiling: Whether to enable profiling for this batch

    Returns:
        New batch state with encoder ready for operations

    Raises:
        RuntimeError: If device not initialized or batch limit exceeded
    """
    encoder = device.create_command_encoder()

    return BatchState(
        encoder=encoder,
        retained_buffers=[],
        enable_profiling=enable_profiling,
        operation_count=0,
    )


def INTERNAL__create_and_retain_uniform_buffer_internal(
    device: GPUDevice, config: GPUConfig, batch_state: BatchState, data: np.ndarray
) -> GPUBuffer:
    """Internal: Create uniform buffer and add to retained list.

    Memory management: Buffer is kept alive by batch_state.retained_buffers
    until submit_batch() is called.

    Args:
        batch_state: Batch state
        data: Numpy array of data

    Returns:
        New WGPU uniform buffer
    """
    buffer = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM)
    batch_state.retained_buffers.append(buffer)
    return buffer


def INTERNAL__create_bind_group_for_batch_internal(
    device: GPUDevice,
    config: GPUConfig,
    batch_state: BatchState,
    pipeline: GPUComputePipeline,
    entries: List[BindGroupEntry],
) -> GPUBindGroup:
    """Internal: Create bind group using type-safe entries.

    Args:
        batch_state: Batch state
        pipeline: Compute pipeline
        entries: Bind group entry specifications

    Returns:
        New WGPU bind group
    """
    return device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=create_bind_group_entries(entries),
    )


def add_compute_to_batch(
    device: GPUDevice,
    config: GPUConfig,
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    kernel_code: str,
    params: np.ndarray,
    buffers: List[GPUBufferAny],
    workgroups_x: int,
    workgroups_y: int = 1,
    workgroups_z: int = 1,
) -> None:
    """
    Internal: Add compute operation to batch encoder.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state
        kernel_code: WGSL kernel source
        params: Parameter array
        buffers: GPU buffers to bind
        workgroups_x: Workgroups in X
        workgroups_y: Workgroups in Y
        workgroups_z: Workgroups in Z

    Raises:
        RuntimeError: If batch operation limit exceeded
    """
    # Check batch operation limit from config
    max_ops = config.max_batch_operations

    if batch_state.operation_count >= max_ops:
        raise RuntimeError(
            f"Batch operation limit ({max_ops}) exceeded. "
            f"Call submit_batch() to flush operations."
        )

    # Validate workgroup counts
    max_workgroups = config.max_workgroups_per_dim

    if (
        workgroups_x > max_workgroups
        or workgroups_y > max_workgroups
        or workgroups_z > max_workgroups
    ):
        raise ValueError(
            f"Workgroup counts ({workgroups_x}, {workgroups_y}, {workgroups_z}) "
            f"exceed maximum ({max_workgroups})"
        )

    params_buffer = INTERNAL__create_and_retain_uniform_buffer_internal(
        device, config, batch_state, params
    )
    pipeline = pipeline_get_or_create(device, pipeline_cache, kernel_code)

    # Build bind group entries
    entries = [BindGroupEntry(0, params_buffer, 0, params.nbytes)]
    for i, buf in enumerate(buffers):
        binding_index = i + 1
        entries.append(BindGroupEntry(binding_index, buf.buffer, 0, buf.size * 4))

    bindgroup = INTERNAL__create_bind_group_for_batch_internal(
        device, config, batch_state, pipeline, entries
    )

    assert batch_state.encoder is not None
    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bindgroup)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z)
    compute_pass.end()

    batch_state.operation_count += 1


def submit_batch(device: GPUDevice, batch_state: BatchState) -> None:
    """Submit all batched operations.

    Args:
        batch_state: Batch state

    Raises:
        RuntimeError: If batch already submitted or not initialized
    """
    if batch_state.encoder is None:
        raise RuntimeError("Batch already submitted or not initialized")

    command_buffer = batch_state.encoder.finish()
    device.queue.submit([command_buffer])

    # Clear encoder and buffers to prevent reuse
    batch_state.encoder = None
    batch_state.retained_buffers.clear()


# ============================================================================
# COMMON OPERATIONS
# ============================================================================
def add_copy(batch_state: BatchState, source: GPUBuffer2D, dest: GPUBuffer2D) -> None:
    """Add buffer copy operation.

    Args:
        batch_state: Batch state
        source: Source buffer
        dest: Destination buffer

    Raises:
        ValueError: If buffer sizes don't match
        RuntimeError: If batch not initialized
    """
    if source.size != dest.size:
        raise ValueError(f"Buffer sizes must match: {source.size} != {dest.size}")

    if batch_state.encoder is None:
        raise RuntimeError("Must call create_command_batch before adding operations")

    batch_state.encoder.copy_buffer_to_buffer(
        source.buffer, 0, dest.buffer, 0, source.size * 4
    )
    batch_state.operation_count += 1


def transpose(
    device: GPUDevice,
    config: GPUConfig,
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """
    Transposes a 2D matrix using tiled algorithm with bank conflict avoidance.
    Used for various operations requiring transposed matrices.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state
        input_buf: Input matrix [rows, cols]
        output: Output matrix [cols, rows]

    Raises:
        ValueError: If buffer shapes incompatible for transpose
    """
    rows, cols = input_buf.shape

    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid input shape: ({rows}, {cols})")

    if output.shape != (cols, rows):
        raise ValueError(
            f"Output shape {output.shape} doesn't match transposed "
            f"input shape ({cols}, {rows})"
        )

    tile_size = config.matmul_tile_size  # Reuse matmul tile size

    params = np.array([rows, cols], dtype=np.uint32)

    workgroups_x = (cols + tile_size - 1) // tile_size
    workgroups_y = (rows + tile_size - 1) // tile_size

    add_compute_to_batch(
        device,
        config,
        pipeline_cache,
        batch_state,
        get_transpose_kernel(config),
        params,
        [input_buf, output],
        workgroups_x,
        workgroups_y,
        1,
    )
