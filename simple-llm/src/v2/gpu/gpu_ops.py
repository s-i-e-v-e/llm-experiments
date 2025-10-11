"""Core GPU operations and compute dispatch"""

from typing import List, Tuple

import numpy as np
from gpu_device import (
    create_bind_group_entries,
    get_or_create_pipeline,
    wgpu,
)
from gpu_types import (
    BatchState,
    BindGroupEntry,
    Device,
    GPUBuffer1D,
    GPUBuffer2D,
    GPUBufferAny,
    PipelineCache,
    WGPUBindGroup,
    WGPUBuffer,
    WGPUComputePipeline,
)

# ============================================================================
# VALIDATION UTILITIES (Phase 3)
# ============================================================================


def validate_buffer_shape_2d(
    buffer: GPUBuffer2D, expected_shape: Tuple[int, int], name: str
) -> None:
    """Validate 2D buffer has expected shape.

    This function does NOT mutate buffer.

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


def validate_buffer_shape_1d(
    buffer: GPUBuffer1D, expected_size: int, name: str
) -> None:
    """Validate 1D buffer has expected size.

    This function does NOT mutate buffer.

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


def validate_matmul_shapes(
    A: GPUBuffer2D, B: GPUBuffer2D, C: GPUBuffer2D, operation: str
) -> Tuple[int, int, int]:
    """Validate shapes for matrix multiplication operations.

    This function does NOT mutate any buffers.

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

    This function does NOT mutate any buffers.

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
# HELPER FUNCTIONS
# ============================================================================


def _create_uniform_buffer_internal(
    pipeline_cache: PipelineCache, data: np.ndarray
) -> WGPUBuffer:
    """Internal: Create uniform buffer for parameters.

    This function does NOT mutate pipeline_cache or data.

    Memory management: Buffer is automatically freed by WGPU after GPU finishes
    using it (typically after queue submission completes). No explicit cleanup needed.

    Args:
        pipeline_cache: Pipeline cache state
        data: Numpy array of parameter data

    Returns:
        New WGPU uniform buffer
    """
    return pipeline_cache.device.wgpu_device.create_buffer_with_data(
        data=data, usage=wgpu.BufferUsage.UNIFORM
    )


def _create_bind_group_internal(
    pipeline_cache: PipelineCache,
    pipeline: WGPUComputePipeline,
    entries: List[BindGroupEntry],
) -> WGPUBindGroup:
    """Internal: Create bind group using type-safe entries.

    This function does NOT mutate any inputs.

    Args:
        pipeline_cache: Pipeline cache state
        pipeline: Compute pipeline
        entries: Bind group entry specifications

    Returns:
        New WGPU bind group
    """
    return pipeline_cache.device.wgpu_device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=create_bind_group_entries(entries),
    )


def _dispatch_compute_internal(
    pipeline_cache: PipelineCache,
    pipeline: WGPUComputePipeline,
    bind_group: WGPUBindGroup,
    workgroups_x: int,
    workgroups_y: int = 1,
    workgroups_z: int = 1,
) -> None:
    """Internal: Create encoder and dispatch compute pass.

    Uiform buffers created in this function are automatically freed
    by WGPU after the queue submission completes. No explicit cleanup needed.

    This function does NOT mutate pipeline_cache.

    Args:
        pipeline_cache: Pipeline cache state
        pipeline: Compute pipeline
        bind_group: Bind group
        workgroups_x: Number of workgroups in X
        workgroups_y: Number of workgroups in Y
        workgroups_z: Number of workgroups in Z
    """
    encoder = pipeline_cache.device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z)
    compute_pass.end()
    pipeline_cache.device.wgpu_device.queue.submit([encoder.finish()])


def dispatch_simple_compute(
    pipelinecache: PipelineCache,
    kernel_code: str,
    params: np.ndarray,
    buffers: List[GPUBufferAny],
    workgroups_x: int,
    workgroups_y: int = 1,
    workgroups_z: int = 1,
) -> None:
    """
    Unified compute dispatch - eliminates repetitive pipeline/bind group/dispatch pattern

    This function may MUTATE pipelinecache by adding cached pipelines.
    This function does NOT mutate params or buffers.

    Memory management: Uniform parameter buffer is created temporarily and automatically
    freed by WGPU after GPU completes execution. No memory leak.

    Args:
        pipelinecache: Pipeline cache state (may be MUTATED for caching)
        kernel_code: WGSL kernel source code
        params: Numpy array of parameters (uploaded as uniform buffer at binding 0)
        buffers: List of GPU buffers to bind (sequential bindings starting at 1)
        workgroups_x: Number of workgroups in X dimension
        workgroups_y: Number of workgroups in Y dimension (default 1)
        workgroups_z: Number of workgroups in Z dimension (default 1)

    Raises:
        ValueError: If workgroup counts exceed device limits
    """
    # Validate workgroup counts against config limits
    max_workgroups = pipelinecache.device.config.max_workgroups_per_dim

    if workgroups_x > max_workgroups:
        raise ValueError(
            f"workgroups_x ({workgroups_x}) exceeds maximum ({max_workgroups}). "
            f"Consider tiling the computation."
        )

    if workgroups_y > max_workgroups:
        raise ValueError(
            f"workgroups_y ({workgroups_y}) exceeds maximum ({max_workgroups})"
        )

    if workgroups_z > max_workgroups:
        raise ValueError(
            f"workgroups_z ({workgroups_z}) exceeds maximum ({max_workgroups})"
        )

    # Create uniform buffer for parameters
    params_buffer = _create_uniform_buffer_internal(pipelinecache, params)

    # Get or create pipeline
    pipeline = get_or_create_pipeline(pipelinecache, kernel_code)

    # Build bind group entries: binding 0 is params, rest are buffers
    entries = [BindGroupEntry(0, params_buffer, 0, params.nbytes)]
    for i, buf in enumerate(buffers):
        binding_index = i + 1
        entries.append(BindGroupEntry(binding_index, buf.buffer, 0, buf.size * 4))

    bindgroup = _create_bind_group_internal(pipelinecache, pipeline, entries)

    # Dispatch compute
    _dispatch_compute_internal(
        pipelinecache, pipeline, bindgroup, workgroups_x, workgroups_y, workgroups_z
    )


def _validate_buffer_shapes(
    buffers: List[Tuple[GPUBufferAny, Tuple[int, ...], str]],
) -> None:
    """Validate buffer shapes match expected dimensions.

    This function does NOT mutate buffers.

    Args:
        buffers: List of (buffer, expected_shape, name) tuples

    Raises:
        ValueError: If any buffer shape doesn't match expected
    """
    for buffer, expected_shape, name in buffers:
        if buffer.shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: got {buffer.shape}, expected {expected_shape}"
            )


# ============================================================================
# COMMAND BATCH STATE
# ============================================================================


def create_command_batch(device: Device, enable_profiling: bool = False) -> BatchState:
    """
    Create command batch state for batched GPU operations

    This function does NOT mutate device.

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
    encoder = device.wgpu_device.create_command_encoder()

    return BatchState(
        device=device,
        encoder=encoder,
        retained_buffers=[],
        enable_profiling=enable_profiling,
        operation_count=0,
    )


def _create_and_retain_uniform_buffer_internal(
    batch_state: BatchState, data: np.ndarray
) -> WGPUBuffer:
    """Internal: Create uniform buffer and add to retained list (mutation).

    This function MUTATES batch_state.retained_buffers by appending the new buffer.

    Memory management: Buffer is kept alive by batch_state.retained_buffers
    until submit_batch() is called.

    Args:
        batch_state: Batch state (MUTATED)
        data: Numpy array of data

    Returns:
        New WGPU uniform buffer
    """
    buffer = batch_state.device.wgpu_device.create_buffer_with_data(
        data=data, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(buffer)
    return buffer


def _create_bind_group_for_batch_internal(
    batch_state: BatchState,
    pipeline: WGPUComputePipeline,
    entries: List[BindGroupEntry],
) -> WGPUBindGroup:
    """Internal: Create bind group using type-safe entries.

    This function does NOT mutate batch_state.

    Args:
        batch_state: Batch state
        pipeline: Compute pipeline
        entries: Bind group entry specifications

    Returns:
        New WGPU bind group
    """
    return batch_state.device.wgpu_device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=create_bind_group_entries(entries),
    )


def _add_compute_to_batch_internal(
    pipelinecache: PipelineCache,
    batch_state: BatchState,
    kernel_code: str,
    params: np.ndarray,
    buffers: List[GPUBufferAny],
    workgroups_x: int,
    workgroups_y: int = 1,
    workgroups_z: int = 1,
) -> None:
    """
    Internal: Add compute operation to batch encoder (mutation)

    This function MUTATES batch_state by adding operation and retaining uniform buffer.

    Args:
        pipelinecache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
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
    max_ops = batch_state.device.config.max_batch_operations

    if batch_state.operation_count >= max_ops:
        raise RuntimeError(
            f"Batch operation limit ({max_ops}) exceeded. "
            f"Call submit_batch() to flush operations."
        )

    # Validate workgroup counts
    max_workgroups = batch_state.device.config.max_workgroups_per_dim

    if (
        workgroups_x > max_workgroups
        or workgroups_y > max_workgroups
        or workgroups_z > max_workgroups
    ):
        raise ValueError(
            f"Workgroup counts ({workgroups_x}, {workgroups_y}, {workgroups_z}) "
            f"exceed maximum ({max_workgroups})"
        )

    params_buffer = _create_and_retain_uniform_buffer_internal(batch_state, params)
    pipeline = get_or_create_pipeline(pipelinecache, kernel_code)

    # Build bind group entries
    entries = [BindGroupEntry(0, params_buffer, 0, params.nbytes)]
    for i, buf in enumerate(buffers):
        binding_index = i + 1
        entries.append(BindGroupEntry(binding_index, buf.buffer, 0, buf.size * 4))

    bindgroup = _create_bind_group_for_batch_internal(batch_state, pipeline, entries)

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bindgroup)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z)
    compute_pass.end()

    batch_state.operation_count += 1


def submit_batch(batch_state: BatchState) -> None:
    """Submit all batched operations (mutation).

    This function MUTATES batch_state by clearing encoder and retained buffers.
    Returns None to signal mutation.

    Args:
        batch_state: Batch state (MUTATED)

    Raises:
        RuntimeError: If batch already submitted or not initialized
    """
    if batch_state.encoder is None:
        raise RuntimeError("Batch already submitted or not initialized")

    command_buffer = batch_state.encoder.finish()
    batch_state.device.wgpu_device.queue.submit([command_buffer])

    # Clear encoder and buffers
    batch_state.encoder = None
    batch_state.retained_buffers.clear()
