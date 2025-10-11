"""Individual kernel dispatch functions"""

import numpy as np

try:
    import wgpu

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None

from gpu_device import get_or_create_pipeline
from gpu_kernels import (
    BIAS_BACKWARD_KERNEL,
    GELU_BACKWARD_KERNEL,
    LAYERNORM_BACKWARD_KERNEL,
    MATMUL_BACKWARD_A_KERNEL,
    MATMUL_BACKWARD_B_KERNEL,
)
from gpu_types import GPUBuffer, PipelineCache

# ============================================================================
# BACKWARD PASS OPERATIONS
# ============================================================================


def run_matmul_backward(
    pipeline_cache: PipelineCache,
    A: GPUBuffer,
    B: GPUBuffer,
    grad_C: GPUBuffer,
    grad_A: GPUBuffer,
    grad_B: GPUBuffer,
) -> None:
    """
    Backward pass for matrix multiplication
    Given: A, B, grad_C
    Compute: grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
    """
    device = pipeline_cache.device

    M, K = A.shape
    K2, N = B.shape

    # Compute grad_A = grad_C @ B^T
    params = np.array([M, K, N], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline_A = get_or_create_pipeline(pipeline_cache, MATMUL_BACKWARD_A_KERNEL)
    bind_group_A = device.wgpu_device.create_bind_group(
        layout=pipeline_A.get_bind_group_layout(0),
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

    encoder = device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline_A)
    compute_pass.set_bind_group(0, bind_group_A)
    compute_pass.dispatch_workgroups((K + 15) // 16, (M + 15) // 16, 1)
    compute_pass.end()

    # Compute grad_B = A^T @ grad_C
    pipeline_B = get_or_create_pipeline(pipeline_cache, MATMUL_BACKWARD_B_KERNEL)
    bind_group_B = device.wgpu_device.create_bind_group(
        layout=pipeline_B.get_bind_group_layout(0),
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

    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline_B)
    compute_pass.set_bind_group(0, bind_group_B)
    compute_pass.dispatch_workgroups((N + 15) // 16, (K + 15) // 16, 1)
    compute_pass.end()

    device.wgpu_device.queue.submit([encoder.finish()])


def run_layernorm_backward(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer,
    gamma: GPUBuffer,
    grad_output: GPUBuffer,
    grad_input: GPUBuffer,
    grad_gamma: GPUBuffer,
    grad_beta: GPUBuffer,
) -> None:
    """Backward pass for layer normalization"""
    device = pipeline_cache.device

    n_elements, size = input_buf.shape

    # Zero out gamma and beta gradients BEFORE kernel (they accumulate inside kernel)
    zero_data = np.zeros(grad_gamma.size, dtype=np.float32)
    device.wgpu_device.queue.write_buffer(grad_gamma.buffer, 0, zero_data)
    device.wgpu_device.queue.write_buffer(grad_beta.buffer, 0, zero_data)

    params = np.array([size, n_elements], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(pipeline_cache, LAYERNORM_BACKWARD_KERNEL)

    bind_group = device.wgpu_device.create_bind_group(
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

    encoder = device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    # One workgroup per batch element
    compute_pass.dispatch_workgroups(n_elements, 1, 1)
    compute_pass.end()
    device.wgpu_device.queue.submit([encoder.finish()])


def run_gelu_backward(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer,
    grad_output: GPUBuffer,
    grad_input: GPUBuffer,
) -> None:
    """Backward pass for GELU activation"""
    device = pipeline_cache.device

    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(pipeline_cache, GELU_BACKWARD_KERNEL)

    bind_group = device.wgpu_device.create_bind_group(
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

    encoder = device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()
    device.wgpu_device.queue.submit([encoder.finish()])


def run_bias_backward(
    pipeline_cache: PipelineCache, grad_output: GPUBuffer, grad_bias: GPUBuffer
) -> None:
    """Backward pass for bias - sum gradients over batch"""
    device = pipeline_cache.device

    n_elements, dim = grad_output.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(pipeline_cache, BIAS_BACKWARD_KERNEL)

    bind_group = device.wgpu_device.create_bind_group(
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

    encoder = device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((dim + 255) // 256, 1, 1)
    compute_pass.end()
    device.wgpu_device.queue.submit([encoder.finish()])
