"""Core GPU operations and compute dispatch"""

from typing import List, Union

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
    GPUBuffer3D,
    PipelineCache,
    WGPUBindGroup,
    WGPUBuffer,
    WGPUComputePipeline,
)

GPUBufferAny = Union[GPUBuffer1D, GPUBuffer2D, GPUBuffer3D]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _create_uniform_buffer_internal(
    pipeline_cache: PipelineCache, data: np.ndarray
) -> WGPUBuffer:
    """Internal: Create uniform buffer for parameters"""
    return pipeline_cache.device.wgpu_device.create_buffer_with_data(
        data=data, usage=wgpu.BufferUsage.UNIFORM
    )


def _create_bind_group_internal(
    pipeline_cache: PipelineCache,
    pipeline: WGPUComputePipeline,
    entries: List[BindGroupEntry],
) -> WGPUBindGroup:
    """Internal: Create bind group using type-safe entries"""
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
    """Internal: Create encoder and dispatch compute pass"""
    encoder = pipeline_cache.device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z)
    compute_pass.end()
    pipeline_cache.device.wgpu_device.queue.submit([encoder.finish()])


def dispatch_simple_compute(
    pipeline_cache: PipelineCache,
    kernel_code: str,
    params: np.ndarray,
    buffers: List[GPUBufferAny],
    workgroups_x: int,
    workgroups_y: int = 1,
    workgroups_z: int = 1,
) -> None:
    """
    Unified compute dispatch - eliminates repetitive pipeline/bind group/dispatch pattern.

    Args:
        pipeline_cache: Pipeline cache state
        kernel_code: WGSL kernel source code
        params: Numpy array of parameters (will be uploaded as uniform buffer at binding 0)
        buffers: List of GPU buffers to bind (sequential bindings starting at 1)
        workgroups_x: Number of workgroups in X dimension
        workgroups_y: Number of workgroups in Y dimension (default: 1)
        workgroups_z: Number of workgroups in Z dimension (default: 1)
    """
    # Create uniform buffer for parameters
    params_buffer = _create_uniform_buffer_internal(pipeline_cache, params)

    # Get or create pipeline
    pipeline = get_or_create_pipeline(pipeline_cache, kernel_code)

    # Build bind group entries: binding 0 is params, rest are buffers
    entries = [BindGroupEntry(0, params_buffer, 0, params.nbytes)]

    for i, buf in enumerate(buffers):
        binding_index = i + 1
        entries.append(BindGroupEntry(binding_index, buf.buffer, 0, buf.size * 4))

    bind_group = _create_bind_group_internal(pipeline_cache, pipeline, entries)

    # Dispatch compute
    _dispatch_compute_internal(
        pipeline_cache, pipeline, bind_group, workgroups_x, workgroups_y, workgroups_z
    )


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


def _create_and_retain_uniform_buffer_internal(
    batch_state: BatchState, data: np.ndarray
) -> WGPUBuffer:
    """Internal: Create uniform buffer and add to retained list"""
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
    """Internal: Create bind group using type-safe entries"""
    return batch_state.device.wgpu_device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=create_bind_group_entries(entries),
    )


def _add_compute_to_batch_internal(
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

    Similar to dispatch_simple_compute but adds to batch instead of immediate dispatch.
    """
    params_buffer = _create_and_retain_uniform_buffer_internal(batch_state, params)
    pipeline = get_or_create_pipeline(pipeline_cache, kernel_code)

    # Build bind group entries: binding 0 is params, rest are buffers
    entries = [BindGroupEntry(0, params_buffer, 0, params.nbytes)]
    for i, buf in enumerate(buffers):
        binding_index = i + 1
        entries.append(BindGroupEntry(binding_index, buf.buffer, 0, buf.size * 4))

    bind_group = _create_bind_group_for_batch_internal(batch_state, pipeline, entries)

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z)
    compute_pass.end()

    batch_state.operation_count += 1
