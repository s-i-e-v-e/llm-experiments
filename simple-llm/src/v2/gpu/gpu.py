"""
Fixed high-performance GPU transformer using WGSL kernels.
Complete implementation with all data structures and helper functions.
"""

import numpy as np

from v2.gpu_batch import (
    CommandBatcher,
)
from v2.gpu_kernels import (
    ADAMW_OPTIMIZER_KERNEL,
    BIAS_ADD_KERNEL,
    BIAS_BACKWARD_KERNEL,
    CROSS_ENTROPY_LOSS_KERNEL,
    EMBEDDING_KERNEL,
    EXTRACT_LAST_TOKENS_KERNEL,
    FLASHATTENTION_FORWARD_KERNEL,
    GELU_BACKWARD_KERNEL,
    GELU_KERNEL,
    LAYERNORM_BACKWARD_KERNEL,
    LAYERNORM_KERNEL,
    MATMUL_BACKWARD_A_KERNEL,
    MATMUL_BACKWARD_B_KERNEL,
    MULTIHEAD_ATTENTION_KERNEL,
    RESIDUAL_ADD_KERNEL,
    TILED_MATMUL_KERNEL,
    TRANSPOSE_KERNEL,
)
from v2.gpu_util import (
    BufferPool,
    GPUBuffer,
    GPUModelParams,
    GPUOptimizerState,
    _get_or_create_pipeline,
    create_gpu_buffer,
    create_gpu_model_params,
    create_optimizer_state,
    dict_to_gpu_layer,
    get_device,
    gpu_layer_to_dict,
    gpu_to_numpy,
    wgpu,
)

__all__ = [
    "GPUModelParams",
    "GPUOptimizerState",
    "create_gpu_model_params",
    "dict_to_gpu_layer",
    "create_optimizer_state",
    "EMBEDDING_KERNEL",
    "CROSS_ENTROPY_LOSS_KERNEL",
    "CommandBatcher",
    "gpu_to_numpy",
    "gpu_layer_to_dict",
    "BufferPool",
]

# ============================================================================
# KERNEL EXECUTION FUNCTIONS
# ============================================================================


def run_matmul(A: GPUBuffer, B: GPUBuffer, C: GPUBuffer, device=None):
    """Execute tiled matrix multiplication: C = A @ B"""
    device = device or get_device()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible shapes: {A.shape} @ {B.shape}"
    assert C.shape == (M, N), f"Output shape mismatch: {C.shape} != ({M}, {N})"

    params = np.array([M, K, N], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(TILED_MATMUL_KERNEL, device)

    bind_group = device.create_bind_group(
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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((N + 15) // 16, (M + 15) // 16, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def run_layernorm(
    input_buf: GPUBuffer,
    gamma: GPUBuffer,
    beta: GPUBuffer,
    output: GPUBuffer,
    device=None,
):
    """Execute layer normalization"""
    device = device or get_device()

    n_elements, size = input_buf.shape

    params = np.array([size, n_elements], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(LAYERNORM_KERNEL, device)

    bind_group = device.create_bind_group(
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
                "resource": {"buffer": beta.buffer, "offset": 0, "size": beta.size * 4},
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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(n_elements, 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def run_gelu(input_buf: GPUBuffer, output: GPUBuffer, device=None):
    """Apply GELU activation"""
    device = device or get_device()

    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(GELU_KERNEL, device)

    bind_group = device.create_bind_group(
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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def run_residual_add(
    input_a: GPUBuffer, input_b: GPUBuffer, output: GPUBuffer, device=None
):
    """Element-wise addition for residual connections"""
    device = device or get_device()

    total_size = input_a.size
    assert input_a.size == input_b.size == output.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(RESIDUAL_ADD_KERNEL, device)

    bind_group = device.create_bind_group(
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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def run_bias_add(input_buf: GPUBuffer, bias: GPUBuffer, output: GPUBuffer, device=None):
    """Add bias vector to each row of input matrix"""
    device = device or get_device()

    n_elements, dim = input_buf.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(BIAS_ADD_KERNEL, device)

    bind_group = device.create_bind_group(
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
                "resource": {"buffer": bias.buffer, "offset": 0, "size": bias.size * 4},
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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


# ============================================================================
# BACKWARD PASS EXECUTION FUNCTIONS
# ============================================================================


def run_matmul_backward(
    A: GPUBuffer,
    B: GPUBuffer,
    grad_C: GPUBuffer,
    grad_A: GPUBuffer,
    grad_B: GPUBuffer,
    device=None,
):
    """
    Backward pass for matrix multiplication
    Given: A, B, grad_C
    Compute: grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
    """
    device = device or get_device()

    M, K = A.shape
    K2, N = B.shape

    # Compute grad_A = grad_C @ B^T
    params = np.array([M, K, N], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline_A = _get_or_create_pipeline(MATMUL_BACKWARD_A_KERNEL, device)
    bind_group_A = device.create_bind_group(
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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline_A)
    compute_pass.set_bind_group(0, bind_group_A)
    compute_pass.dispatch_workgroups((K + 15) // 16, (M + 15) // 16, 1)
    compute_pass.end()

    # Compute grad_B = A^T @ grad_C
    pipeline_B = _get_or_create_pipeline(MATMUL_BACKWARD_B_KERNEL, device)
    bind_group_B = device.create_bind_group(
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

    device.queue.submit([encoder.finish()])


def run_layernorm_backward(
    input_buf: GPUBuffer,
    gamma: GPUBuffer,
    grad_output: GPUBuffer,
    grad_input: GPUBuffer,
    grad_gamma: GPUBuffer,
    grad_beta: GPUBuffer,
    device=None,
):
    """Backward pass for layer normalization"""
    device = device or get_device()

    n_elements, size = input_buf.shape

    # Zero out gamma and beta gradients BEFORE kernel (they accumulate inside kernel)
    zero_data = np.zeros(grad_gamma.size, dtype=np.float32)
    device.queue.write_buffer(grad_gamma.buffer, 0, zero_data)
    device.queue.write_buffer(grad_beta.buffer, 0, zero_data)

    params = np.array([size, n_elements], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(LAYERNORM_BACKWARD_KERNEL, device)

    bind_group = device.create_bind_group(
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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    # One workgroup per batch element
    compute_pass.dispatch_workgroups(n_elements, 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def run_gelu_backward(
    input_buf: GPUBuffer, grad_output: GPUBuffer, grad_input: GPUBuffer, device=None
):
    """Backward pass for GELU activation"""
    device = device or get_device()

    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(GELU_BACKWARD_KERNEL, device)

    bind_group = device.create_bind_group(
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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def run_bias_backward(grad_output: GPUBuffer, grad_bias: GPUBuffer, device=None):
    """Backward pass for bias - sum gradients over batch"""
    device = device or get_device()

    n_elements, dim = grad_output.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(BIAS_BACKWARD_KERNEL, device)

    bind_group = device.create_bind_group(
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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((dim + 255) // 256, 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


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
    device=None,
):
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

    pipeline = _get_or_create_pipeline(ADAMW_OPTIMIZER_KERNEL, device)

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


# ============================================================================
# ATTENTION AND INFERENCE HELPER FUNCTIONS
# ============================================================================


def run_multihead_attention(
    Q: GPUBuffer,
    K: GPUBuffer,
    V: GPUBuffer,
    output: GPUBuffer,
    n_heads: int,
    device=None,
):
    """
    Execute multi-head self-attention.

    Args:
        Q, K, V: [batch_size * seq_len, embedding_dim]
        output: [batch_size * seq_len, embedding_dim]
        n_heads: Number of attention heads
    """
    device = device or get_device()

    batch_seq, embedding_dim = Q.shape
    head_dim = embedding_dim // n_heads

    # Infer batch_size and seq_len (need to pass this properly in real implementation)
    # For now, assume batch_size = 1 for inference
    batch_size = 1
    seq_len = batch_seq

    params = np.array([batch_size, seq_len, n_heads, head_dim], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(MULTIHEAD_ATTENTION_KERNEL, device)

    bind_group = device.create_bind_group(
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
                "resource": {"buffer": Q.buffer, "offset": 0, "size": Q.size * 4},
            },
            {
                "binding": 2,
                "resource": {"buffer": K.buffer, "offset": 0, "size": K.size * 4},
            },
            {
                "binding": 3,
                "resource": {"buffer": V.buffer, "offset": 0, "size": V.size * 4},
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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    # Launch one workgroup per (batch, head, query_position)
    compute_pass.dispatch_workgroups(seq_len, n_heads, batch_size)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def run_transpose(input_buf: GPUBuffer, output: GPUBuffer, device=None):
    """Transpose a matrix"""
    device = device or get_device()

    rows, cols = input_buf.shape
    assert output.shape == (cols, rows), (
        f"Output shape {output.shape} != ({cols}, {rows})"
    )

    params = np.array([rows, cols], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(TRANSPOSE_KERNEL, device)

    bind_group = device.create_bind_group(
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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((cols + 15) // 16, (rows + 15) // 16, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def run_extract_last_tokens(
    input_buf: GPUBuffer, output: GPUBuffer, batch_size: int, seq_len: int, device=None
):
    """Extract last token from each sequence"""
    device = device or get_device()

    embedding_dim = input_buf.size // (batch_size * seq_len)

    params = np.array([batch_size, seq_len, embedding_dim], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(EXTRACT_LAST_TOKENS_KERNEL, device)

    bind_group = device.create_bind_group(
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

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((embedding_dim + 255) // 256, batch_size, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


# ============================================================================
# FLASHATTENTION EXECUTION FUNCTIONS
# ============================================================================


def run_flashattention(
    Q: GPUBuffer,
    K: GPUBuffer,
    V: GPUBuffer,
    output: GPUBuffer,
    n_heads: int,
    save_for_backward: bool = False,
    device=None,
):
    """
    Execute FlashAttention with tiling and online softmax.

    Args:
        Q, K, V: [batch_size * seq_len, embedding_dim]
        output: [batch_size * seq_len, embedding_dim]
        n_heads: Number of attention heads
        save_for_backward: If True, save L and M statistics

    Returns:
        If save_for_backward: (output, L_buffer, M_buffer)
        Else: output
    """
    device = device or get_device()

    batch_seq, embedding_dim = Q.shape
    head_dim = embedding_dim // n_heads

    # For now, assume batch_size = 1 for simplicity
    batch_size = 1
    seq_len = batch_seq

    # Block sizes (tuned for typical WGSL shared memory limits)
    Bc = 32  # Block size for K/V
    Br = 32  # Block size for Q

    # Number of Q blocks to process
    num_q_blocks = (seq_len + Br - 1) // Br

    # Create statistics buffers if needed
    if save_for_backward:
        L_buffer = create_gpu_buffer((batch_size, seq_len, n_heads), device=device)
        M_buffer = create_gpu_buffer((batch_size, seq_len, n_heads), device=device)
    else:
        # Dummy buffers (won't be used)
        L_buffer = create_gpu_buffer((1,), device=device)
        M_buffer = create_gpu_buffer((1,), device=device)

    params = np.array([batch_size, seq_len, n_heads, head_dim, Bc, Br], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(FLASHATTENTION_FORWARD_KERNEL, device)

    bind_group = device.create_bind_group(
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
                "resource": {"buffer": Q.buffer, "offset": 0, "size": Q.size * 4},
            },
            {
                "binding": 2,
                "resource": {"buffer": K.buffer, "offset": 0, "size": K.size * 4},
            },
            {
                "binding": 3,
                "resource": {"buffer": V.buffer, "offset": 0, "size": V.size * 4},
            },
            {
                "binding": 4,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
            {
                "binding": 5,
                "resource": {
                    "buffer": L_buffer.buffer,
                    "offset": 0,
                    "size": L_buffer.size * 4,
                },
            },
            {
                "binding": 6,
                "resource": {
                    "buffer": M_buffer.buffer,
                    "offset": 0,
                    "size": M_buffer.size * 4,
                },
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    # Launch one workgroup per (batch, head, Q_block)
    compute_pass.dispatch_workgroups(num_q_blocks, n_heads, batch_size)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    if save_for_backward:
        return output, L_buffer, M_buffer
    return output


def run_simple_attention(
    Q: GPUBuffer,
    K: GPUBuffer,
    V: GPUBuffer,
    output: GPUBuffer,
    n_heads: int,
    device=None,
):
    """
    Simplified attention without tiling - faster for small sequences.
    Use this instead of FlashAttention for seq_len < 512
    """
    device = device or get_device()

    batch_seq, embedding_dim = Q.shape
    head_dim = embedding_dim // n_heads

    # Just use the multihead attention kernel we already have
    return run_multihead_attention(Q, K, V, output, n_heads, device)
