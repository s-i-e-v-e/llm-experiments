"""Forward pass operations - individual kernel dispatches"""

import numpy as np
from gpu_kernels_forward import (
    BIAS_ADD_KERNEL,
    CROSS_ENTROPY_LOSS_KERNEL,
    EMBEDDING_KERNEL,
    GELU_KERNEL,
    LAYERNORM_KERNEL,
    MULTIHEAD_ATTENTION_KERNEL,
    RESIDUAL_ADD_KERNEL,
    TILED_MATMUL_KERNEL,
)
from gpu_ops import dispatch_simple_compute
from gpu_types import GPUBuffer1D, GPUBuffer2D, PipelineCache

# ============================================================================
# FORWARD PASS OPERATIONS
# ============================================================================


def run_matmul(
    pipeline_cache: PipelineCache,
    A: GPUBuffer2D,
    B: GPUBuffer2D,
    C: GPUBuffer2D,
) -> None:
    """
    Matrix multiplication: C = A @ B.

    Uses tiled algorithm for efficiency. This function MUTATES C.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        A: Input matrix (M, K)
        B: Input matrix (K, N)
        C: Output matrix (M, N) - MUTATED
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Dimension mismatch: {K} != {K2}"
    assert C.shape == (M, N), f"Output shape mismatch: {C.shape} != ({M}, {N})"

    params = np.array([M, K, N], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        TILED_MATMUL_KERNEL,
        params,
        [A, B, C],
        min((N + 15) // 16, 65535),
        min((M + 15) // 16, 65535),
        1,
    )


def run_embedding(
    pipeline_cache: PipelineCache,
    embedding_table: GPUBuffer2D,
    pos_encoding: GPUBuffer2D,
    input_ids: GPUBuffer1D,
    output: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
) -> None:
    """
    Embedding lookup with positional encoding.

    Looks up token embeddings and adds positional encodings.
    This function MUTATES output.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        embedding_table: Token embedding table (vocab_size, embedding_dim)
        pos_encoding: Positional encoding table (context_size, embedding_dim)
        input_ids: Input token IDs (batch_size * seq_len,) - stored as uint32
        output: Output embeddings (batch_size * seq_len, embedding_dim) - MUTATED
        batch_size: Batch size
        seq_len: Sequence length
    """
    vocab_size, embedding_dim = embedding_table.shape
    context_size, embedding_dim2 = pos_encoding.shape
    assert embedding_dim == embedding_dim2, "Embedding dimension mismatch"
    assert seq_len <= context_size, (
        f"Sequence length {seq_len} exceeds context size {context_size}"
    )
    assert input_ids.size == batch_size * seq_len, "Input IDs size mismatch"
    assert output.shape == (batch_size * seq_len, embedding_dim), (
        "Output shape mismatch"
    )

    params = np.array([batch_size, seq_len, embedding_dim], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        EMBEDDING_KERNEL,
        params,
        [embedding_table, pos_encoding, input_ids, output],
        (batch_size * seq_len + 255) // 256,
    )


def run_multihead_attention(
    pipeline_cache: PipelineCache,
    Q: GPUBuffer2D,
    K: GPUBuffer2D,
    V: GPUBuffer2D,
    output: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
) -> None:
    """
    Multi-head self-attention with causal masking.

    Computes scaled dot-product attention for each head, with causal masking
    to prevent attending to future positions.

    This function MUTATES output.

    PERFORMANCE NOTES:
    - Works for any seq_len and head_dim (no hardcoded limits)
    - Recomputes attention scores multiple times (tradeoff: memory vs compute)
    - For very long sequences (>2048), consider tiling or FlashAttention

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        Q: Query matrix (batch_size * seq_len, n_heads * head_dim)
        K: Key matrix (batch_size * seq_len, n_heads * head_dim)
        V: Value matrix (batch_size * seq_len, n_heads * head_dim)
        output: Output matrix (batch_size * seq_len, n_heads * head_dim) - MUTATED
        batch_size: Batch size
        seq_len: Sequence length (no upper limit)
        n_heads: Number of attention heads
        head_dim: Dimension per head (no upper limit)
    """
    embedding_dim = n_heads * head_dim
    total_tokens = batch_size * seq_len

    assert Q.shape == (total_tokens, embedding_dim), f"Q shape mismatch: {Q.shape}"
    assert K.shape == (total_tokens, embedding_dim), f"K shape mismatch: {K.shape}"
    assert V.shape == (total_tokens, embedding_dim), f"V shape mismatch: {V.shape}"
    assert output.shape == (total_tokens, embedding_dim), "Output shape mismatch"

    params = np.array([batch_size, seq_len, n_heads, head_dim], dtype=np.uint32)

    # Dispatch one workgroup per (batch, head, query_position)
    # Each workgroup processes one query position for one head
    dispatch_simple_compute(
        pipeline_cache,
        MULTIHEAD_ATTENTION_KERNEL,
        params,
        [Q, K, V, output],
        workgroups_x=seq_len,  # One workgroup per query position
        workgroups_y=n_heads,  # One per head
        workgroups_z=batch_size,  # One per batch
    )


def run_layernorm(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    beta: GPUBuffer1D,
    output: GPUBuffer2D,
) -> None:
    """
    Layer normalization with affine transformation.

    This function MUTATES output.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input tensor (n_elements, size)
        gamma: Scale parameters (size,)
        beta: Shift parameters (size,)
        output: Output tensor (n_elements, size) - MUTATED
    """
    n_elements, size = input_buf.shape

    params = np.array([size, n_elements], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        LAYERNORM_KERNEL,
        params,
        [input_buf, gamma, beta, output],
        n_elements,
    )


def run_gelu(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """
    GELU activation function.

    This function MUTATES output.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input tensor
        output: Output tensor - MUTATED
    """
    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        GELU_KERNEL,
        params,
        [input_buf, output],
        (total_size + 255) // 256,
    )


def run_bias_add(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    bias: GPUBuffer1D,
    output: GPUBuffer2D,
) -> None:
    """
    Add bias to input tensor (broadcast over first dimension).

    This function MUTATES output.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input tensor (n_elements, dim)
        bias: Bias vector (dim,)
        output: Output tensor (n_elements, dim) - MUTATED
    """
    n_elements, dim = input_buf.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        BIAS_ADD_KERNEL,
        params,
        [input_buf, bias, output],
        (total_size + 255) // 256,
    )


def run_residual_add(
    pipeline_cache: PipelineCache,
    input_a: GPUBuffer2D,
    input_b: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """
    Element-wise addition for residual connections.

    This function MUTATES output.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_a: First input tensor
        input_b: Second input tensor
        output: Output tensor - MUTATED
    """
    total_size = input_a.size

    params = np.array([total_size], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        RESIDUAL_ADD_KERNEL,
        params,
        [input_a, input_b, output],
        (total_size + 255) // 256,
    )


def run_cross_entropy_loss(
    pipeline_cache: PipelineCache,
    logits: GPUBuffer2D,
    targets: GPUBuffer1D,
    loss_output: GPUBuffer1D,
    grad_logits: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
) -> None:
    """
    Combined cross-entropy loss and gradient computation.

    Computes both loss values and gradients in one pass for efficiency.
    This function MUTATES loss_output and grad_logits.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        logits: Model predictions (batch_size * seq_len, vocab_size)
        targets: Target token IDs (batch_size * seq_len,) - uint32
        loss_output: Loss for each prediction (batch_size * seq_len,) - MUTATED
        grad_logits: Gradient w.r.t. logits (batch_size * seq_len, vocab_size) - MUTATED
        batch_size: Batch size
        seq_len: Sequence length
    """
    total_predictions = batch_size * seq_len
    vocab_size = logits.shape[1]

    assert logits.shape == (total_predictions, vocab_size), "Logits shape mismatch"
    assert targets.size == total_predictions, "Targets size mismatch"
    assert loss_output.size == total_predictions, "Loss output size mismatch"
    assert grad_logits.shape == (total_predictions, vocab_size), (
        "Grad logits shape mismatch"
    )

    params = np.array([batch_size, seq_len, vocab_size], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        CROSS_ENTROPY_LOSS_KERNEL,
        params,
        [logits, targets, loss_output, grad_logits],
        (total_predictions + 255) // 256,
    )
