import numpy as np
from gpu_kernels import (
    get_attention_kernel,
    get_bias_add_kernel,
    get_cross_entropy_loss_kernel,
    get_embedding_kernel,
    get_extract_last_tokens_kernel,
    get_flash_attention_kernel,
    get_gelu_kernel,
    get_layernorm_kernel,
    get_matmul_kernel,
    get_residual_add_kernel,
    get_softmax_kernel,
)
from gpu_types import (
    BatchState,
    GPUBuffer1D,
    GPUBuffer2D,
    PipelineCache,
)

from .gpu_ops import (
    _add_compute_to_batch_internal,
    validate_buffer_shape_1d,
    validate_buffer_shape_2d,
    validate_matmul_shapes,
)

# ============================================================================
# FORWARD PASS OPERATIONS
# ============================================================================


def matmul(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    A: GPUBuffer2D,
    B: GPUBuffer2D,
    C: GPUBuffer2D,
) -> None:
    """Matrix multiplication: C = A @ B (mutation).

    Uses tiled algorithm for efficiency. For matrices larger than
    ~1M x 1M, automatically tiles into multiple kernel launches.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED; operation added)
        A: Input matrix (M, K)
        B: Input matrix (K, N)
        C: Output matrix (M, N) (MUTATED)

    Raises:
        ValueError: If dimensions are incompatible
        NotImplementedError: If matrix exceeds maximum supported size
    """
    config = pipeline_cache.device.config

    M, K, N = validate_matmul_shapes(A, B, C, "matmul")

    MAX_WORKGROUPS = 65535
    TILE_SIZE = config.matmul_tile_size

    wg_x = (N + TILE_SIZE - 1) // TILE_SIZE
    wg_y = (M + TILE_SIZE - 1) // TILE_SIZE

    if wg_x <= MAX_WORKGROUPS and wg_y <= MAX_WORKGROUPS:
        params = np.array([M, K, N], dtype=np.uint32)
        _add_compute_to_batch_internal(
            pipeline_cache,
            batch_state,
            get_matmul_kernel(config),
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


def embedding(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    embedding_table: GPUBuffer2D,
    pos_encoding: GPUBuffer2D,
    input_ids: GPUBuffer1D,
    output: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
) -> None:
    """Embedding lookup with positional encoding

    Looks up token embeddings and adds positional encodings.


    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED; operation added)
        embedding_table: Token embedding table (vocab_size, embedding_dim)
        pos_encoding: Positional encoding table (context_size, embedding_dim)
        input_ids: Input token IDs (batch_size * seq_len,) - stored as uint32
        output: Output embeddings (batch_size * seq_len, embedding_dim) (MUTATED)
        batch_size: Batch size
        seq_len: Sequence length

    Raises:
        ValueError: If buffer shapes don't match or seq_len exceeds context_size
    """
    config = pipeline_cache.device.config

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
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_embedding_kernel(config),
        params,
        [embedding_table, pos_encoding, input_ids, output],
        (total_tokens + 255) // 256,
    )


def attention(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    Q: GPUBuffer2D,
    K: GPUBuffer2D,
    V: GPUBuffer2D,
    output: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
) -> None:
    """Multi-head self-attention with causal masking

    Computes scaled dot-product attention for each head, with causal masking
    to prevent attending to future positions.

    This function MUTATES output. Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        Q: Query matrix (batch_size * seq_len, n_heads * head_dim)
        K: Key matrix (batch_size * seq_len, n_heads * head_dim)
        V: Value matrix (batch_size * seq_len, n_heads * head_dim)
        output: Output matrix (batch_size * seq_len, n_heads * head_dim) (MUTATED)
        batch_size: Batch size
        seq_len: Sequence length (no upper limit)
        n_heads: Number of attention heads
        head_dim: Dimension per head (no upper limit)

    Raises:
        ValueError: If buffer shapes don't match or dimensions invalid
    """
    config = pipeline_cache.device.config

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

    params = np.array([batch_size, seq_len, n_heads, head_dim], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_attention_kernel(config),
        params,
        [Q, K, V, output],
        workgroups_x=seq_len,
        workgroups_y=n_heads,
        workgroups_z=batch_size,
    )


def layernorm(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    beta: GPUBuffer1D,
    output: GPUBuffer2D,
) -> None:
    """Layer normalization with affine transformation

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_buf: Input tensor (n_elements, size)
        gamma: Scale parameters (size,)
        beta: Shift parameters (size,)
        output: Output tensor (n_elements, size) (MUTATED)

    Raises:
        ValueError: If buffer shapes don't match
    """
    config = pipeline_cache.device.config

    n_elements, size = input_buf.shape

    if n_elements == 0 or size == 0:
        raise ValueError(f"Invalid input_buf shape: {(n_elements, size)}")

    validate_buffer_shape_1d(gamma, size, "gamma")
    validate_buffer_shape_1d(beta, size, "beta")
    validate_buffer_shape_2d(output, (n_elements, size), "output")

    params = np.array([size, n_elements], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_layernorm_kernel(config),
        params,
        [input_buf, gamma, beta, output],
        n_elements,
    )


def gelu(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """GELU activation function.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_buf: Input tensor
        output: Output tensor (MUTATED)

    Raises:
        ValueError: If buffer shapes don't match
    """
    config = pipeline_cache.device.config

    if input_buf.shape != output.shape:
        raise ValueError(
            f"Input shape {input_buf.shape} doesn't match output shape {output.shape}"
        )

    total_size = input_buf.size
    params = np.array([total_size], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_gelu_kernel(config),
        params,
        [input_buf, output],
        (total_size + 255) // 256,
    )


def softmax(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    logits: GPUBuffer2D,
    probs: GPUBuffer2D,
) -> None:
    """
    Apply softmax to logits to get probability distribution.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        batch_state: Current batch state with encoder (MUTATED)
        logits: Input logits [batch_size, vocab_size]
        probs: Output probabilities [batch_size, vocab_size] (MUTATED)

    Note:
        Uses numerically stable softmax with max subtraction.
        For generation/sampling after computing final layer logits.
    """
    config = pipeline_cache.device.config

    # Validate shapes
    validate_buffer_shape_2d(logits)
    validate_buffer_shape_2d(probs)

    if logits.shape != probs.shape:
        raise ValueError(
            f"Softmax shape mismatch: logits {logits.shape} != probs {probs.shape}"
        )

    batch_size, vocab_size = logits.shape

    params = np.array([batch_size, vocab_size], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_softmax_kernel(config),
        params,
        [logits.buffer, probs.buffer],
        batch_size,
    )


def bias_add(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    bias: GPUBuffer1D,
    output: GPUBuffer2D,
) -> None:
    """Add bias to input tensor.

    Broadcasts bias over first dimension.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_buf: Input tensor (n_elements, dim)
        bias: Bias vector (dim,)
        output: Output tensor (n_elements, dim) (MUTATED)

    Raises:
        ValueError: If buffer shapes don't match
    """
    config = pipeline_cache.device.config

    n_elements, dim = input_buf.shape

    if n_elements == 0 or dim == 0:
        raise ValueError(f"Invalid input_buf shape: {(n_elements, dim)}")

    validate_buffer_shape_1d(bias, dim, "bias")
    validate_buffer_shape_2d(output, (n_elements, dim), "output")

    total_size = n_elements * dim
    params = np.array([total_size, dim], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_bias_add_kernel(config),
        params,
        [input_buf, bias, output],
        (total_size + 255) // 256,
    )


def residual_add(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_a: GPUBuffer2D,
    input_b: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """Element-wise addition for residual connections.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_a: First input tensor
        input_b: Second input tensor
        output: Output tensor (MUTATED)

    Raises:
        ValueError: If buffer shapes don't match
    """
    config = pipeline_cache.device.config

    if not (input_a.shape == input_b.shape == output.shape):
        raise ValueError(
            f"Shape mismatch: input_a={input_a.shape}, "
            f"input_b={input_b.shape}, output={output.shape}"
        )

    total_size = input_a.size
    params = np.array([total_size], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_residual_add_kernel(config),
        params,
        [input_a, input_b, output],
        (total_size + 255) // 256,
    )


def extract_last_tokens(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    output: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
    embedding_dim: int,
) -> None:
    """
    Extracts the last token embedding from each sequence in the batch.
    Used for generation/inference to get final hidden states.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_buf: Input tensor [batch_size*seq_len, embedding_dim]
        output: Output tensor [batch_size, embedding_dim] (MUTATED)
        batch_size: Batch size
        seq_len: Sequence length
        embedding_dim: Embedding dimension

    Raises:
        ValueError: If buffer shapes don't match or dimensions invalid
    """
    config = pipeline_cache.device.config

    if batch_size <= 0 or seq_len <= 0 or embedding_dim <= 0:
        raise ValueError(
            f"Invalid dimensions: batch_size={batch_size}, seq_len={seq_len}, "
            f"embedding_dim={embedding_dim}"
        )

    total_tokens = batch_size * seq_len
    validate_buffer_shape_2d(input_buf, (total_tokens, embedding_dim), "input_buf")
    validate_buffer_shape_2d(output, (batch_size, embedding_dim), "output")

    params = np.array([batch_size, seq_len, embedding_dim], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_extract_last_tokens_kernel(config),
        params,
        [input_buf, output],
        (embedding_dim + 255) // 256,
        batch_size,
        1,
    )


def cross_entropy_loss(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    logits: GPUBuffer2D,
    targets: GPUBuffer1D,
    loss_output: GPUBuffer1D,
    grad_logits: GPUBuffer2D,
) -> None:
    """Combined cross-entropy loss and gradient computation.

    Computes both loss values and gradients in one pass for efficiency.
    This function MUTATES loss_output and grad_logits. Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        logits: Model predictions (batch_size * seq_len, vocab_size)
        targets: Target token IDs (batch_size * seq_len,) - uint32
        loss_output: Loss for each prediction (batch_size * seq_len,) (MUTATED)
        grad_logits: Gradient w.r.t. logits (batch_size * seq_len, vocab_size) (MUTATED)
        batch_size: Batch size
        seq_len: Sequence length

    Raises:
        ValueError: If buffer shapes don't match
    """
    config = pipeline_cache.device.config
    batch_seq, vocab_size = logits.shape

    if batch_seq <= 0 or vocab_size <= 0:
        raise ValueError(f"Invalid logits shape: {(batch_seq, vocab_size)}")

    validate_buffer_shape_1d(targets, batch_seq, "targets")
    validate_buffer_shape_1d(loss_output, batch_seq, "loss_output")
    validate_buffer_shape_2d(grad_logits, (batch_seq, vocab_size), "grad_logits")
    validate_buffer_shape_2d(logits, (batch_seq, vocab_size), "logits")

    params = np.array([batch_seq, 1, vocab_size], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_cross_entropy_loss_kernel(config),
        params,
        [logits, targets, loss_output, grad_logits],
        (batch_seq + 255) // 256,
    )


def flash_attention(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
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
    """
    Add FlashAttention forward pass to batch.

    This function MUTATES batch_state by adding operations, and will MUTATE
    O, L, M buffers when batch is submitted.


    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        Q: Query matrix [batch_size*seq_len, n_heads*head_dim]
        K: Key matrix [batch_size*seq_len, n_heads*head_dim]
        V: Value matrix [batch_size*seq_len, n_heads*head_dim]
        O: Output matrix [batch_size*seq_len, n_heads*head_dim] (MUTATED)
        L: Softmax normalization [batch_size*seq_len*n_heads] (MUTATED)
        M: Max values [batch_size*seq_len*n_heads] (MUTATED)
        batch_size: Batch size
        seq_len: Sequence length
        n_heads: Number of attention heads
        head_dim: Dimension per head

    Raises:
        ValueError: If buffer shapes don't match or dimensions invalid
    """
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

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_flash_attention_kernel(config),
        params,
        [Q, K, V, O, L, M],
        num_q_blocks,
        n_heads,
        batch_size,
    )
