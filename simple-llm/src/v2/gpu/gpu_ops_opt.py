"""Individual kernel dispatch functions"""

import numpy as np

try:
    import wgpu

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None

from gpu_device import get_device, get_or_create_pipeline
from gpu_kernels import (
    ADAMW_OPTIMIZER_KERNEL,
)


def run_adamw_update(
    gradients: GPUBuffer,
    weights: GPUBuffer,
    m: GPUBuffer,
    v: GPUBuffer,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    step: int,
    device: Optional[object] = None,
) -> None:
    """Execute AdamW optimizer update"""
    device = device or get_device()

    total_size = weights.size

    opt_params = np.array(
        [lr, beta1, beta2, weight_decay, eps, float(step)], dtype=np.float32
    )
    opt_params_buffer = device.create_buffer_with_data(
        data=opt_params, usage=wgpu.BufferUsage.UNIFORM
    )

    size_buffer = device.create_buffer_with_data(
        data=np.array([total_size], dtype=np.uint32), usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(ADAMW_OPTIMIZER_KERNEL, device)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": opt_params_buffer,
                    "offset": 0,
                    "size": opt_params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": gradients.buffer,
                    "offset": 0,
                    "size": gradients.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": weights.buffer,
                    "offset": 0,
                    "size": weights.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {"buffer": m.buffer, "offset": 0, "size": m.size * 4},
            },
            {
                "binding": 4,
                "resource": {"buffer": v.buffer, "offset": 0, "size": v.size * 4},
            },
            {"binding": 5, "resource": {"buffer": size_buffer, "offset": 0, "size": 4}},
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])
