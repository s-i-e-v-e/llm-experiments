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
from gpu_ops import (
    _validate_buffer_shapes,
    dispatch_simple_compute,
    validate_buffer_shape_1d,
    validate_buffer_shape_2d,
    validate_matmul_shapes,
)
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
    """Matrix multiplication: C = A @ B (mutation).

    Uses tiled algorithm for efficiency. For matrices larger than
    ~1M x 1M, automatically tiles into multiple kernel launches.

    This function MUTATES C. Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        A: Input matrix (M, K)
        B: Input matrix (K, N)
        C: Output matrix (M, N) (MUTATED)

    Raises:
        ValueError: If dimensions are incompatible
        NotImplementedError: If matrix exceeds maximum supported size
    """
    M, K, N = validate_matmul_shapes(A, B, C, "run_matmul")

    # Workgroup limit in WGSL
    MAX_WORKGROUPS = 65535
    TILE_SIZE = 16

    # Calculate required workgroups
    wg_x = (N + TILE_SIZE - 1) // TILE_SIZE
    wg_y = (M + TILE_SIZE - 1) // TILE_SIZE

    # Check if we need tiling
    if wg_x <= MAX_WORKGROUPS and wg_y <= MAX_WORKGROUPS:
        # Single dispatch
        params = np.array([M, K, N], dtype=np.uint32)
        dispatch_simple_compute(
            pipeline_cache,
            TILED_MATMUL_KERNEL,
            params,
            [A, B, C],
            wg_x,
            wg_y,
            1,
        )
    else:
        # Matrix too large
        M_tile_size = MAX_WORKGROUPS * TILE_SIZE
        N_tile_size = MAX_WORKGROUPS * TILE_SIZE
        raise NotImplementedError(
            f"Matrix too large: ({M}, {K}) @ ({K}, {N}). "
            f"Maximum supported size is ({M_tile_size}, K) @ (K, {N_tile_size}). "
            f"Consider splitting the computation manually or using a different backend."
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
    """Embedding lookup with positional encoding (mutation).

    Looks up token embeddings and adds positional encodings.
    This function MUTATES output. Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        embedding_table: Token embedding table (vocab_size, embedding_dim)
        pos_encoding: Positional encoding table (context_size, embedding_dim)
        input_ids: Input token IDs (batch_size * seq_len,) - stored as uint32
        output: Output embeddings (batch_size * seq_len, embedding_dim) (MUTATED)
        batch_size: Batch size
        seq_len: Sequence length

    Raises:
        ValueError: If buffer shapes don't match or seq_len exceeds context_size
    """
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError(f"Invalid batch_size={batch_size} or seq_len={seq_len}")

    vocab_size, embedding_dim = embedding_table.shape
    context_size, embedding_dim2 = pos_encoding.shape

    if embedding_dim != embedding_dim2:
        raise ValueError(
            f"Embedding dimension mismatch: table has {embedding_dim}, "
            f"pos_encoding has {embedding_dim2}"
        )
    if seq_len > context_size:
        raise ValueError(
            f"Sequence length {seq_len} exceeds context size {context_size}"
        )

    total_tokens = batch_size * seq_len
    validate_buffer_shape_1d(input_ids, total_tokens, "input_ids")
    validate_buffer_shape_2d(output, (total_tokens, embedding_dim), "output")

    params = np.array([batch_size, seq_len, embedding_dim], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        EMBEDDING_KERNEL,
        params,
        [embedding_table, pos_encoding, input_ids, output],
        (total_tokens + 255) // 256,
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
    """Multi-head self-attention with causal masking (mutation).

    Computes scaled dot-product attention for each head, with causal masking
    to prevent attending to future positions.

    This function MUTATES output. Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        Q: Query matrix (batch_size * seq_len, n_heads * head_dim)
        K: Key matrix (batch_size * seq_len, n_heads * head_dim)
        V: Value matrix (batch_size * seq_len, n_heads * head_dim)
        output: Output matrix (batch_size * seq_len, n_heads * head_dim) (MUTATED)
        batch_size: Batch size
        seq_len: Sequence length (no upper limit)
        n_heads: Number of attention heads
        head_dim: Dimension per head (no upper limit)

    Raises:
        ValueError: If buffer shapes don't match
    """
    if batch_size <= 0 or seq_len <= 0 or n_heads <= 0 or head_dim <= 0:
        raise ValueError(
            f"Invalid dimensions: batch_size={batch_size}, seq_len={seq_len}, "
            f"n_heads={n_heads}, head_dim={head_dim}"
        )

    embedding_dim = n_heads * head_dim
    total_tokens = batch_size * seq_len

    _validate_buffer_shapes(
        [
            (Q, (total_tokens, embedding_dim), "Q"),
            (K, (total_tokens, embedding_dim), "K"),
            (V, (total_tokens, embedding_dim), "V"),
            (output, (total_tokens, embedding_dim), "output"),
        ]
    )

    params = np.array([batch_size, seq_len, n_heads, head_dim], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        MULTIHEAD_ATTENTION_KERNEL,
        params,
        [Q, K, V, output],
        workgroups_x=seq_len,
        workgroups_y=n_heads,
        workgroups_z=batch_size,
    )


def run_layernorm(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    beta: GPUBuffer1D,
    output: GPUBuffer2D,
) -> None:
    """Layer normalization with affine transformation (mutation).

    This function MUTATES output. Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input tensor (n_elements, size)
        gamma: Scale parameters (size,)
        beta: Shift parameters (size,)
        output: Output tensor (n_elements, size) (MUTATED)

    Raises:
        ValueError: If buffer shapes don't match
    """
    n_elements, size = input_buf.shape

    if n_elements <= 0 or size <= 0:
        raise ValueError(f"Invalid input shape: ({n_elements}, {size})")

    validate_buffer_shape_1d(gamma, size, "gamma")
    validate_buffer_shape_1d(beta, size, "beta")
    validate_buffer_shape_2d(output, (n_elements, size), "output")

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
    """GELU activation function (mutation).

    This function MUTATES output. Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input tensor
        output: Output tensor (MUTATED)

    Raises:
        ValueError: If buffer shapes don't match
    """
    if input_buf.shape != output.shape:
        raise ValueError(
            f"Input/output shape mismatch: {input_buf.shape} != {output.shape}"
        )

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
    """Add bias to input tensor (mutation).

    Broadcasts bias over first dimension.
    This function MUTATES output. Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input tensor (n_elements, dim)
        bias: Bias vector (dim,)
        output: Output tensor (n_elements, dim) (MUTATED)

    Raises:
        ValueError: If buffer shapes don't match
    """
    n_elements, dim = input_buf.shape

    if n_elements <= 0 or dim <= 0:
        raise ValueError(f"Invalid input shape: ({n_elements}, {dim})")

    total_size = n_elements * dim

    validate_buffer_shape_1d(bias, dim, "bias")
    validate_buffer_shape_2d(output, (n_elements, dim), "output")

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
    """Element-wise addition for residual connections (mutation).

    This function MUTATES output. Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_a: First input tensor
        input_b: Second input tensor
        output: Output tensor (MUTATED)

    Raises:
        ValueError: If buffer shapes don't match
    """
    if not (input_a.shape == input_b.shape == output.shape):
        raise ValueError(
            f"Shape mismatch: input_a={input_a.shape}, "
            f"input_b={input_b.shape}, output={output.shape}"
        )

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
    """Combined cross-entropy loss and gradient computation (mutation).

    Computes both loss values and gradients in one pass for efficiency.
    This function MUTATES loss_output and grad_logits. Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        logits: Model predictions (batch_size * seq_len, vocab_size)
        targets: Target token IDs (batch_size * seq_len,) - uint32
        loss_output: Loss for each prediction (batch_size * seq_len,) (MUTATED)
        grad_logits: Gradient w.r.t. logits (batch_size * seq_len, vocab_size) (MUTATED)
        batch_size: Batch size
        seq_len: Sequence length

    Raises:
        ValueError: If buffer shapes don't match
    """
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError(f"Invalid batch_size={batch_size} or seq_len={seq_len}")

    total_predictions = batch_size * seq_len
    vocab_size = logits.shape[1]

    validate_buffer_shape_2d(logits, (total_predictions, vocab_size), "logits")
    validate_buffer_shape_1d(targets, total_predictions, "targets")
    validate_buffer_shape_1d(loss_output, total_predictions, "loss_output")
    validate_buffer_shape_2d(
        grad_logits, (total_predictions, vocab_size), "grad_logits"
    )

    params = np.array([batch_size, seq_len, vocab_size], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        CROSS_ENTROPY_LOSS_KERNEL,
        params,
        [logits, targets, loss_output, grad_logits],
        (total_predictions + 255) // 256,
    )
