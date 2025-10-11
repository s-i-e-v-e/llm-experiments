import numpy as np
from gpu_device import (
    BindGroupEntry,
    create_bind_group_entries,
    wgpu,
)
from gpu_types import PipelineCache

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _create_uniform_buffer(pipeline_cache: PipelineCache, data: np.ndarray) -> object:
    """Helper: Create uniform buffer for parameters"""
    return pipeline_cache.device.wgpu_device.create_buffer_with_data(
        data=data, usage=wgpu.BufferUsage.UNIFORM
    )


def _create_bind_group(
    pipeline_cache: PipelineCache, pipeline: object, entries: list[BindGroupEntry]
) -> object:
    """Helper: Create bind group using type-safe entries"""
    return pipeline_cache.device.wgpu_device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=create_bind_group_entries(entries),
    )


def _dispatch_compute(
    pipeline_cache: PipelineCache,
    pipeline: object,
    bind_group: object,
    workgroups_x: int,
    workgroups_y: int = 1,
    workgroups_z: int = 1,
) -> None:
    """Helper: Create encoder and dispatch compute pass"""
    encoder = pipeline_cache.device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z)
    compute_pass.end()
    pipeline_cache.device.wgpu_device.queue.submit([encoder.finish()])
