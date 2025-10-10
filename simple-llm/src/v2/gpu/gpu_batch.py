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


class CommandBatcher:
    """Batch multiple GPU operations into single submission with proper resource retention"""

    def __init__(self, device):
        self.device = device
        self.encoder = None
        self._retained_buffers = []  # Keep references until submit

    def begin(self):
        """Start batching operations"""
        self.encoder = self.device.create_command_encoder()
        self._retained_buffers = []
        return self

    def add_matmul(self, A: GPUBuffer, B: GPUBuffer, C: GPUBuffer):
        """Add matmul to batch"""
        M, K = A.shape
        K2, N = B.shape

        params = np.array([M, K, N], dtype=np.uint32)
        params_buffer = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM
        )
        self._retained_buffers.append(params_buffer)

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

        # Validate workgroup size against device limits
        workgroups_x = min((N + 15) // 16, 65535)
        workgroups_y = min((M + 15) // 16, 65535)
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1)
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
        self._retained_buffers.append(params_buffer)

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
        self._retained_buffers.append(params_buffer)

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

    def add_bias_backward(self, grad_output: GPUBuffer, grad_bias: GPUBuffer):
        """Add bias backward to batch - sums gradients over batch dimension"""
        from v2.gpu_kernels import BIAS_BACKWARD_KERNEL

        n_elements, dim = grad_output.shape
        total_size = n_elements * dim

        params = np.array([total_size, dim], dtype=np.uint32)
        params_buffer = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM
        )
        self._retained_buffers.append(params_buffer)

        pipeline = _get_or_create_pipeline(BIAS_BACKWARD_KERNEL, self.device)
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

        compute_pass = self.encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups((dim + 255) // 256, 1, 1)
        compute_pass.end()

    def add_residual(self, input_a: GPUBuffer, input_b: GPUBuffer, output: GPUBuffer):
        """Add residual connection to batch"""
        total_size = input_a.size

        params = np.array([total_size], dtype=np.uint32)
        params_buffer = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM
        )
        self._retained_buffers.append(params_buffer)

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
        self._retained_buffers.append(params_buffer)

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

    def add_copy(self, source: GPUBuffer, dest: GPUBuffer):
        """Add buffer copy operation to batch"""
        assert source.size == dest.size, (
            f"Buffer sizes must match: {source.size} != {dest.size}"
        )

        if self.encoder is None:
            raise RuntimeError("Must call begin() before adding operations")

        self.encoder.copy_buffer_to_buffer(
            source.buffer,
            0,
            dest.buffer,
            0,
            source.size * 4,  # 4 bytes per float32
        )

    def add_matmul_backward_a(self, grad_C: GPUBuffer, B: GPUBuffer, grad_A: GPUBuffer):
        """Add matmul backward w.r.t. A: grad_A = grad_C @ B^T"""
        from v2.gpu_kernels import MATMUL_BACKWARD_A_KERNEL

        M, N = grad_C.shape
        K2, N2 = B.shape
        assert N == N2, f"Dimension mismatch: {N} != {N2}"

        params = np.array([M, K2, N], dtype=np.uint32)
        params_buffer = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM
        )
        self._retained_buffers.append(params_buffer)

        pipeline = _get_or_create_pipeline(MATMUL_BACKWARD_A_KERNEL, self.device)
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

        compute_pass = self.encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        workgroups_x = min((K2 + 15) // 16, 65535)
        workgroups_y = min((M + 15) // 16, 65535)
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1)
        compute_pass.end()

    def add_matmul_backward_b(self, A: GPUBuffer, grad_C: GPUBuffer, grad_B: GPUBuffer):
        """Add matmul backward w.r.t. B: grad_B = A^T @ grad_C"""
        from v2.gpu_kernels import MATMUL_BACKWARD_B_KERNEL

        M, K = A.shape
        M2, N = grad_C.shape
        assert M == M2, f"Dimension mismatch: {M} != {M2}"

        params = np.array([M, K, N], dtype=np.uint32)
        params_buffer = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM
        )
        self._retained_buffers.append(params_buffer)

        pipeline = _get_or_create_pipeline(MATMUL_BACKWARD_B_KERNEL, self.device)
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

        compute_pass = self.encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        workgroups_x = min((N + 15) // 16, 65535)
        workgroups_y = min((K + 15) // 16, 65535)
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1)
        compute_pass.end()

    def add_gelu_backward(
        self, input_buf: GPUBuffer, grad_output: GPUBuffer, grad_input: GPUBuffer
    ):
        """Add GELU backward to batch"""
        from v2.gpu_kernels import GELU_BACKWARD_KERNEL

        total_size = input_buf.size
        assert grad_output.size == total_size and grad_input.size == total_size

        params = np.array([total_size], dtype=np.uint32)
        params_buffer = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM
        )
        self._retained_buffers.append(params_buffer)

        pipeline = _get_or_create_pipeline(GELU_BACKWARD_KERNEL, self.device)
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

        compute_pass = self.encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
        compute_pass.end()

    def add_layernorm_backward(
        self,
        input_buf: GPUBuffer,
        gamma: GPUBuffer,
        grad_output: GPUBuffer,
        grad_input: GPUBuffer,
        grad_gamma: GPUBuffer,
        grad_beta: GPUBuffer,
    ):
        """Add layernorm backward to batch"""
        from v2.gpu_kernels import LAYERNORM_BACKWARD_KERNEL

        n_elements, size = input_buf.shape

        params = np.array([size, n_elements], dtype=np.uint32)
        params_buffer = self.device.create_buffer_with_data(
            data=params, usage=wgpu.BufferUsage.UNIFORM
        )
        self._retained_buffers.append(params_buffer)

        pipeline = _get_or_create_pipeline(LAYERNORM_BACKWARD_KERNEL, self.device)
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

        compute_pass = self.encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(n_elements, 1, 1)
        compute_pass.end()

    def submit(self):
        """Execute all batched operations and release retained resources"""
        if self.encoder is not None:
            self.device.queue.submit([self.encoder.finish()])

            if self.enable_profiling and self.operation_count > 0:
                print(f"Batched {self.operation_count} operations in single submission")

            self.encoder = None
        self._retained_buffers = []
