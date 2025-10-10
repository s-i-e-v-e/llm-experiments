import numpy as np

from v2.gpu_kernels import (
    BIAS_ADD_KERNEL,
    GELU_KERNEL,
    LAYERNORM_KERNEL,
    RESIDUAL_ADD_KERNEL,
    TILED_MATMUL_KERNEL,
)
from v2.gpu_util import (
    GPUBuffer,
    _get_or_create_pipeline,
    wgpu,
)

# ============================================================================
# BATCHED OPERATIONS (Minimize GPU Submissions)
# ============================================================================


class CommandBatcher:
    """Batch multiple GPU operations into single submission"""

    def __init__(self, device):
        self.device = device
        self.encoder = None

    def begin(self):
        """Start batching operations"""
        self.encoder = self.device.create_command_encoder()
        return self

    def add_matmul(self, A: GPUBuffer, B: GPUBuffer, C: GPUBuffer):
        """Add matmul to batch"""
        M, K = A.shape
        K2, N = B.shape

        params = np.array([M, K, N], dtype=np.uint32)
        params_buffer = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM
        )

        pipeline = _get_or_create_pipeline(TILED_MATMUL_KERNEL, self.device)
        bind_group = self.device.create_bind_group(
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

        compute_pass = self.encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups((N + 15) // 16, (M + 15) // 16, 1)
        compute_pass.end()

    def add_layernorm(
        self, input_buf: GPUBuffer, gamma: GPUBuffer, beta: GPUBuffer, output: GPUBuffer
    ):
        """Add layernorm to batch"""
        n_elements, size = input_buf.shape

        params = np.array([size, n_elements], dtype=np.uint32)
        params_buffer = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM
        )

        pipeline = _get_or_create_pipeline(LAYERNORM_KERNEL, self.device)
        bind_group = self.device.create_bind_group(
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

        compute_pass = self.encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(n_elements, 1, 1)
        compute_pass.end()

    def add_gelu(self, input_buf: GPUBuffer, output: GPUBuffer):
        """Add GELU to batch"""
        total_size = input_buf.size

        params = np.array([total_size], dtype=np.uint32)
        params_buffer = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM
        )

        pipeline = _get_or_create_pipeline(GELU_KERNEL, self.device)
        bind_group = self.device.create_bind_group(
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

        compute_pass = self.encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
        compute_pass.end()

    def add_residual(self, input_a: GPUBuffer, input_b: GPUBuffer, output: GPUBuffer):
        """Add residual connection to batch"""
        total_size = input_a.size

        params = np.array([total_size], dtype=np.uint32)
        params_buffer = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM
        )

        pipeline = _get_or_create_pipeline(RESIDUAL_ADD_KERNEL, self.device)
        bind_group = self.device.create_bind_group(
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

        compute_pass = self.encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
        compute_pass.end()

    def add_bias_add(self, input_buf: GPUBuffer, bias: GPUBuffer, output: GPUBuffer):
        """Add bias addition to batch"""
        n_elements, dim = input_buf.shape
        total_size = n_elements * dim

        params = np.array([total_size, dim], dtype=np.uint32)
        params_buffer = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM
        )

        pipeline = _get_or_create_pipeline(BIAS_ADD_KERNEL, self.device)
        bind_group = self.device.create_bind_group(
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

        compute_pass = self.encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
        compute_pass.end()

    def submit(self):
        """Execute all batched operations"""
        self.device.queue.submit([self.encoder.finish()])
        self.encoder = None
