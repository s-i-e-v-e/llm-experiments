"""Forward pass operations - individual kernel dispatches"""

import numpy as np

# Import config-based kernel generators
from gpu_kernels_forward import (
    # Keep constants for backward compatibility
    CROSS_ENTROPY_LOSS_KERNEL,
    get_attention_kernel_from_config,
    get_bias_add_kernel_from_config,
    get_embedding_kernel_from_config,
    get_extract_last_tokens_kernel_from_config,
    get_flash_attention_kernel_from_config,
    get_gelu_kernel_from_config,
    get_layernorm_kernel_from_config,
    get_matmul_kernel_from_config,
    get_residual_add_kernel_from_config,
)
from gpu_ops import (
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

    MAX_WORKGROUPS = 65535
    config = pipeline_cache.device.config
    TILE_SIZE = config.matmul_tile_size

    wg_x = (N + TILE_SIZE - 1) // TILE_SIZE
    wg_y = (M + TILE_SIZE - 1) // TILE_SIZE

    if wg_x <= MAX_WORKGROUPS and wg_y <= MAX_WORKGROUPS:
        params = np.array([M, K, N], dtype=np.uint32)
        dispatch_simple_compute(
            pipeline_cache,
            get_matmul_kernel_from_config(config),
            params,
            [A, B, C],
            wg_x,
            wg_y,
            1,
        )
    else:
        M_tile_size = MAX_WORKGROUPS * TILE_SIZE
        N_tile_size = MAX_WORKGROUPS * TILE_SIZE
        raise NotImplementedError(
            f"Matrix too large: ({M}, {K}) @ ({K}, {N}). "
            f"Maximum supported size is ({M_tile_size}, K) @ (K, {N_tile_size})."
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

    config = pipeline_cache.device.config
    params = np.array([batch_size, seq_len, embedding_dim], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        get_embedding_kernel_from_config(config),
        params,
        [embedding_table, pos_encoding, input_ids, output],
        (total_tokens + 255) // 256,
    )


def run_attention(
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

    validate_buffer_shape_2d(Q, (total_tokens, embedding_dim), "Q")
    validate_buffer_shape_2d(K, (total_tokens, embedding_dim), "K")
    validate_buffer_shape_2d(V, (total_tokens, embedding_dim), "V")
    validate_buffer_shape_2d(output, (total_tokens, embedding_dim), "output")

    config = pipeline_cache.device.config
    params = np.array([batch_size, seq_len, n_heads, head_dim], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        get_attention_kernel_from_config(config),
        params,
        [Q, K, V, output],
        seq_len,
        n_heads,
        batch_size,
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

    if n_elements == 0 or size == 0:
        raise ValueError(f"Invalid input_buf shape: {(n_elements, size)}")

    validate_buffer_shape_1d(gamma, size, "gamma")
    validate_buffer_shape_1d(beta, size, "beta")
    validate_buffer_shape_2d(output, (n_elements, size), "output")

    config = pipeline_cache.device.config
    params = np.array([size, n_elements], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        get_layernorm_kernel_from_config(config),
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
            f"Input shape {input_buf.shape} doesn't match output shape {output.shape}"
        )

    total_size = input_buf.size
    config = pipeline_cache.device.config
    params = np.array([total_size], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        get_gelu_kernel_from_config(config),
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

    if n_elements == 0 or dim == 0:
        raise ValueError(f"Invalid input_buf shape: {(n_elements, dim)}")

    validate_buffer_shape_1d(bias, dim, "bias")
    validate_buffer_shape_2d(output, (n_elements, dim), "output")

    total_size = n_elements * dim
    config = pipeline_cache.device.config
    params = np.array([total_size, dim], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        get_bias_add_kernel_from_config(config),
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
    if input_a.shape != input_b.shape:
        raise ValueError(
            f"Input shapes don't match: {input_a.shape} != {input_b.shape}"
        )

    if output.shape != input_a.shape:
        raise ValueError(
            f"Output shape {output.shape} doesn't match input shape {input_a.shape}"
        )

    total_size = input_a.size
    config = pipeline_cache.device.config
    params = np.array([total_size], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        get_residual_add_kernel_from_config(config),
        params,
        [input_a, input_b, output],
        (total_size + 255) // 256,
    )


def run_extract_last_tokens(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    output: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
    embedding_dim: int,
) -> None:
    """Extract last token from each sequence (mutation)."""
    if batch_size <= 0 or seq_len <= 0 or embedding_dim <= 0:
        raise ValueError(
            f"Invalid dimensions: batch_size={batch_size}, seq_len={seq_len}, "
            f"embedding_dim={embedding_dim}"
        )

    total_tokens = batch_size * seq_len
    validate_buffer_shape_2d(input_buf, (total_tokens, embedding_dim), "input_buf")
    validate_buffer_shape_2d(output, (batch_size, embedding_dim), "output")

    config = pipeline_cache.device.config
    params = np.array([batch_size, seq_len, embedding_dim], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        get_extract_last_tokens_kernel_from_config(config),
        params,
        [input_buf, output],
        (embedding_dim + 255) // 256,
        batch_size,
        1,
    )


def run_cross_entropy_loss(
    pipeline_cache: PipelineCache,
    logits: GPUBuffer2D,
    targets: GPUBuffer1D,
    loss_output: GPUBuffer1D,
    grad_logits: GPUBuffer2D,
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
    batch_seq, vocab_size = logits.shape

    if batch_seq <= 0 or vocab_size <= 0:
        raise ValueError(f"Invalid logits shape: {(batch_seq, vocab_size)}")

    validate_buffer_shape_1d(targets, batch_seq, "targets")
    validate_buffer_shape_1d(loss_output, batch_seq, "loss_output")
    validate_buffer_shape_2d(grad_logits, (batch_seq, vocab_size), "grad_logits")

    # Note: This kernel is not parameterized (workgroup_size=256 hardcoded)
    # because it's self-contained and rarely needs tuning
    params = np.array([batch_seq, 1, vocab_size], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        CROSS_ENTROPY_LOSS_KERNEL,
        params,
        [logits, targets, loss_output, grad_logits],
        (batch_seq + 255) // 256,
    )


def run_flash_attention(
    pipeline_cache: PipelineCache,
    Q: GPUBuffer2D,
    K: GPUBuffer2D,
    V: GPUBuffer2D,
    O: GPUBuffer2D,
    L: GPUBuffer1D,
    M: GPUBuffer1D,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
) -> None:
    """FlashAttention forward pass (mutation)."""
    config = pipeline_cache.device.config

    # Validate head_dim against config
    if head_dim > config.flash_attn_max_head_dim:
        raise ValueError(
            f"head_dim {head_dim} exceeds maximum supported by config: "
            f"{config.flash_attn_max_head_dim}. "
            f"Increase flash_attn_max_head_dim in GPUConfig or use smaller head_dim."
        )

    if batch_size <= 0 or seq_len <= 0 or n_heads <= 0 or head_dim <= 0:
        raise ValueError(
            f"Invalid dimensions: batch_size={batch_size}, seq_len={seq_len}, "
            f"n_heads={n_heads}, head_dim={head_dim}"
        )

    embedding_dim = n_heads * head_dim
    total_tokens = batch_size * seq_len

    validate_buffer_shape_2d(Q, (total_tokens, embedding_dim), "Q")
    validate_buffer_shape_2d(K, (total_tokens, embedding_dim), "K")
    validate_buffer_shape_2d(V, (total_tokens, embedding_dim), "V")
    validate_buffer_shape_2d(O, (total_tokens, embedding_dim), "O")

    # L and M store statistics per (batch, seq, head)
    stats_size = batch_size * seq_len * n_heads
    validate_buffer_shape_1d(L, stats_size, "L")
    validate_buffer_shape_1d(M, stats_size, "M")

    params = np.array(
        [
            batch_size,
            seq_len,
            n_heads,
            head_dim,
            config.flash_attn_bc,
            config.flash_attn_br,
        ],
        dtype=np.uint32,
    )

    Br = config.flash_attn_br
    num_q_blocks = (seq_len + Br - 1) // Br

    dispatch_simple_compute(
        pipeline_cache,
        get_flash_attention_kernel_from_config(config),
        params,
        [Q, K, V, O, L, M],
        num_q_blocks,
        n_heads,
        batch_size,
    )
