"""Individual kernel dispatch functions"""

from typing import Tuple

import numpy as np

try:
    import wgpu

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None

from gpu_buffer import create_gpu_buffer
from gpu_device import get_or_create_pipeline
from gpu_kernels import (
    BIAS_ADD_KERNEL,
    EXTRACT_LAST_TOKENS_KERNEL,
    FLASHATTENTION_FORWARD_KERNEL,
    GELU_KERNEL,
    LAYERNORM_KERNEL,
    MULTIHEAD_ATTENTION_KERNEL,
    RESIDUAL_ADD_KERNEL,
    TILED_MATMUL_KERNEL,
    TRANSPOSE_KERNEL,
)
from gpu_types import GPUBuffer, PipelineCache


# ============================================================================
# FORWARD PASS OPERATIONS
# ============================================================================
def run_matmul(
    pipeline_cache: PipelineCache, A: GPUBuffer, B: GPUBuffer, C: GPUBuffer
) -> None:
    """
    Represents a core matrix multiplication, fundamental for linear transformations in both attention and feed-forward layers.

    Execute tiled matrix multiplication: C = A @ B
    """
    device = pipeline_cache.device

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible shapes: {A.shape} @ {B.shape}"
    assert C.shape == (M, N), f"Output shape mismatch: {C.shape} != ({M}, {N})"

    params = np.array([M, K, N], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(pipeline_cache, TILED_MATMUL_KERNEL)

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

    encoder = device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((N + 15) // 16, (M + 15) // 16, 1)
    compute_pass.end()
    device.wgpu_device.queue.submit([encoder.finish()])


def run_layernorm(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer,
    gamma: GPUBuffer,
    beta: GPUBuffer,
    output: GPUBuffer,
) -> None:
    """
    Applies layer normalization to stabilize the activations.
    """
    device = pipeline_cache.device

    n_elements, size = input_buf.shape

    params = np.array([size, n_elements], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(pipeline_cache, LAYERNORM_KERNEL)

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

    encoder = device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(n_elements, 1, 1)
    compute_pass.end()
    device.wgpu_device.queue.submit([encoder.finish()])


def run_gelu(
    pipeline_cache: PipelineCache, input_buf: GPUBuffer, output: GPUBuffer
) -> None:
    """
    An element-wise activation function used in the feed-forward networks.
    """
    device = pipeline_cache.device

    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(pipeline_cache, GELU_KERNEL)

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
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
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


def run_residual_add(
    pipeline_cache: PipelineCache,
    input_a: GPUBuffer,
    input_b: GPUBuffer,
    output: GPUBuffer,
) -> None:
    """
    Implements the skip/residual connection by adding the input of a block to its output.

    Element-wise addition for residual connections
    """
    device = pipeline_cache.device

    total_size = input_a.size
    assert input_a.size == input_b.size == output.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(pipeline_cache, RESIDUAL_ADD_KERNEL)

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

    encoder = device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()
    device.wgpu_device.queue.submit([encoder.finish()])


def run_bias_add(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer,
    bias: GPUBuffer,
    output: GPUBuffer,
) -> None:
    """Add bias vector to each row of input matrix"""
    device = pipeline_cache.device

    n_elements, dim = input_buf.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(pipeline_cache, BIAS_ADD_KERNEL)

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

    encoder = device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()
    device.wgpu_device.queue.submit([encoder.finish()])


# ============================================================================
# ATTENTION AND HELPER OPERATIONS
# ============================================================================


def run_multihead_attention(
    pipeline_cache: PipelineCache,
    Q: GPUBuffer,
    K: GPUBuffer,
    V: GPUBuffer,
    output: GPUBuffer,
    n_heads: int,
) -> None:
    """
    A standard implementation of the attention mechanism.

    Execute multi-head self-attention.

    Args:
        Q, K, V: [batch_size * seq_len, embedding_dim]
        output: [batch_size * seq_len, embedding_dim]
        n_heads: Number of attention heads
    """
    device = pipeline_cache.device

    batch_seq, embedding_dim = Q.shape
    head_dim = embedding_dim // n_heads

    # Infer batch_size and seq_len (need to pass this properly in real implementation)
    # For now, assume batch_size = 1 for inference
    batch_size = 1
    seq_len = batch_seq

    params = np.array([batch_size, seq_len, n_heads, head_dim], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(pipeline_cache, MULTIHEAD_ATTENTION_KERNEL)

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

    encoder = device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    # Launch one workgroup per (batch, head, query_position)
    compute_pass.dispatch_workgroups(seq_len, n_heads, batch_size)
    compute_pass.end()
    device.wgpu_device.queue.submit([encoder.finish()])


def run_flashattention(
    pipeline_cache: PipelineCache,
    Q: GPUBuffer,
    K: GPUBuffer,
    V: GPUBuffer,
    output: GPUBuffer,
    n_heads: int,
    save_for_backward: bool = False,
) -> GPUBuffer | Tuple[GPUBuffer, GPUBuffer, GPUBuffer]:
    """
    A memory-efficient, optimized implementation of the attention mechanism.

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
    device = pipeline_cache.device

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
        L_buffer = create_gpu_buffer(device, (batch_size, seq_len, n_heads))
        M_buffer = create_gpu_buffer(device, (batch_size, seq_len, n_heads))
    else:
        # Dummy buffers (won't be used)
        L_buffer = create_gpu_buffer(device, (1,))
        M_buffer = create_gpu_buffer(device, (1,))

    params = np.array([batch_size, seq_len, n_heads, head_dim, Bc, Br], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(pipeline_cache, FLASHATTENTION_FORWARD_KERNEL)

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

    encoder = device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    # Launch one workgroup per (batch, head, Q_block)
    compute_pass.dispatch_workgroups(num_q_blocks, n_heads, batch_size)
    compute_pass.end()
    device.wgpu_device.queue.submit([encoder.finish()])

    if save_for_backward:
        return output, L_buffer, M_buffer
    return output


def run_simple_attention(
    pipeline_cache: PipelineCache,
    Q: GPUBuffer,
    K: GPUBuffer,
    V: GPUBuffer,
    output: GPUBuffer,
    n_heads: int,
) -> None:
    """
    Simplified attention without tiling - faster for small sequences.
    Use this instead of FlashAttention for seq_len < 512
    """
    # Just use the multihead attention kernel we already have
    return run_multihead_attention(pipeline_cache, Q, K, V, output, n_heads)


def run_transpose(
    pipeline_cache: PipelineCache, input_buf: GPUBuffer, output: GPUBuffer
) -> None:
    """A sub-operation, often used within attention to align matrix dimensions for multiplication (e.g., transposing the Key matrix)."""
    device = pipeline_cache.device

    rows, cols = input_buf.shape
    assert output.shape == (cols, rows), (
        f"Output shape {output.shape} != ({cols}, {rows})"
    )

    params = np.array([rows, cols], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(pipeline_cache, TRANSPOSE_KERNEL)

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
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
        ],
    )

    encoder = device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((cols + 15) // 16, (rows + 15) // 16, 1)
    compute_pass.end()
    device.wgpu_device.queue.submit([encoder.finish()])


def run_extract_last_tokens(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer,
    output: GPUBuffer,
    batch_size: int,
    seq_len: int,
) -> None:
    """A utility function used at the end of the forward pass to select the final hidden states, which are then used for next-token prediction."""
    device = pipeline_cache.device

    embedding_dim = input_buf.size // (batch_size * seq_len)

    params = np.array([batch_size, seq_len, embedding_dim], dtype=np.uint32)
    params_buffer = device.wgpu_device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = get_or_create_pipeline(pipeline_cache, EXTRACT_LAST_TOKENS_KERNEL)

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
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
        ],
    )

    encoder = device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((embedding_dim + 255) // 256, batch_size, 1)
    compute_pass.end()
    device.wgpu_device.queue.submit([encoder.finish()])
