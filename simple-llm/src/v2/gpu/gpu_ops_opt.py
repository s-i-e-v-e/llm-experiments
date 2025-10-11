"""Optimizer operations"""

import numpy as np

try:
    import wgpu

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None

from gpu_device import BindGroupEntry, create_bind_group_entries, get_or_create_pipeline
from gpu_kernels_opt import ADAMW_OPTIMIZER_KERNEL
from gpu_types import GPUBuffer2D, PipelineCache

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


# ============================================================================
# OPTIMIZER OPERATIONS
# ============================================================================


def run_adamw_update(
    pipeline_cache: PipelineCache,
    gradients: GPUBuffer2D,
    weights: GPUBuffer2D,
    m: GPUBuffer2D,
    v: GPUBuffer2D,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    step: int,
) -> None:
    """Execute AdamW optimizer update"""
    total_size = weights.size

    opt_params = np.array(
        [lr, beta1, beta2, weight_decay, eps, float(step)], dtype=np.float32
    )
    opt_params_buffer = _create_uniform_buffer(pipeline_cache, opt_params)

    size_params = np.array([total_size], dtype=np.uint32)
    size_buffer = _create_uniform_buffer(pipeline_cache, size_params)

    pipeline = get_or_create_pipeline(pipeline_cache, ADAMW_OPTIMIZER_KERNEL)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, opt_params_buffer, 0, opt_params.nbytes),
            BindGroupEntry(1, gradients.buffer, 0, gradients.size * 4),
            BindGroupEntry(2, weights.buffer, 0, weights.size * 4),
            BindGroupEntry(3, m.buffer, 0, m.size * 4),
            BindGroupEntry(4, v.buffer, 0, v.size * 4),
            BindGroupEntry(5, size_buffer, 0, size_params.nbytes),
        ],
    )

    _dispatch_compute(pipeline_cache, pipeline, bind_group, (total_size + 255) // 256)
