"""Command batching"""

import numpy as np
from gpu_buffer import clear_buffer
from gpu_kernels_backward import (
    BIAS_BACKWARD_KERNEL,
    GELU_BACKWARD_KERNEL,
    LAYERNORM_BACKWARD_KERNEL,
    MATMUL_BACKWARD_A_KERNEL,
    MATMUL_BACKWARD_B_KERNEL,
)
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
from gpu_kernels_opt import ADAMW_OPTIMIZER_KERNEL
from gpu_ops import (
    _add_compute_to_batch_internal,
    _validate_buffer_shapes,
    validate_buffer_shape_1d,
    validate_buffer_shape_2d,
    validate_matmul_shapes,
    validate_optimizer_buffers,
)
from gpu_types import BatchState, GPUBuffer1D, GPUBuffer2D, PipelineCache

# ============================================================================
# FORWARD PASS BATCH OPERATIONS
# ============================================================================


def batch_add_embedding(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    embedding_table: GPUBuffer2D,
    pos_encoding: GPUBuffer2D,
    input_ids: GPUBuffer1D,
    output: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
) -> None:
    """Add embedding lookup to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        embedding_table: Token embedding table (vocab_size, embedding_dim)
        pos_encoding: Positional encoding table (context_size, embedding_dim)
        input_ids: Input token IDs (batch_size * seq_len,)
        output: Output embeddings (batch_size * seq_len, embedding_dim)
        batch_size: Batch size
        seq_len: Sequence length

    Raises:
        ValueError: If buffer shapes don't match
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
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        EMBEDDING_KERNEL,
        params,
        [embedding_table, pos_encoding, input_ids, output],
        (total_tokens + 255) // 256,
    )


def batch_add_attention(
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
    """Add multi-head attention to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        Q: Query matrix (batch_size * seq_len, n_heads * head_dim)
        K: Key matrix (batch_size * seq_len, n_heads * head_dim)
        V: Value matrix (batch_size * seq_len, n_heads * head_dim)
        output: Output matrix (batch_size * seq_len, n_heads * head_dim)
        batch_size: Batch size
        seq_len: Sequence length
        n_heads: Number of attention heads
        head_dim: Dimension per head

    Raises:
        ValueError: If buffer shapes don't match or dimensions invalid
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
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        MULTIHEAD_ATTENTION_KERNEL,
        params,
        [Q, K, V, output],
        workgroups_x=seq_len,
        workgroups_y=n_heads,
        workgroups_z=batch_size,
    )


def batch_add_matmul(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    A: GPUBuffer2D,
    B: GPUBuffer2D,
    C: GPUBuffer2D,
) -> None:
    """Add matmul to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        A: Input matrix (M, K)
        B: Input matrix (K, N)
        C: Output matrix (M, N)

    Raises:
        ValueError: If dimensions are incompatible
    """
    M, K, N = validate_matmul_shapes(A, B, C, "batch_add_matmul")

    params = np.array([M, K, N], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        TILED_MATMUL_KERNEL,
        params,
        [A, B, C],
        min((N + 15) // 16, 65535),
        min((M + 15) // 16, 65535),
        1,
    )


def batch_add_layernorm(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    beta: GPUBuffer1D,
    output: GPUBuffer2D,
) -> None:
    """Add layernorm to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_buf: Input tensor (n_elements, size)
        gamma: Scale parameters (size,)
        beta: Shift parameters (size,)
        output: Output tensor (n_elements, size)

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
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        LAYERNORM_KERNEL,
        params,
        [input_buf, gamma, beta, output],
        n_elements,
    )


def batch_add_gelu(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """Add GELU to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_buf: Input tensor
        output: Output tensor

    Raises:
        ValueError: If buffer shapes don't match
    """
    if input_buf.shape != output.shape:
        raise ValueError(
            f"Input/output shape mismatch: {input_buf.shape} != {output.shape}"
        )

    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        GELU_KERNEL,
        params,
        [input_buf, output],
        (total_size + 255) // 256,
    )


def batch_add_bias_add(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    bias: GPUBuffer1D,
    output: GPUBuffer2D,
) -> None:
    """Add bias addition to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_buf: Input tensor (n_elements, dim)
        bias: Bias vector (dim,)
        output: Output tensor (n_elements, dim)

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
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        BIAS_ADD_KERNEL,
        params,
        [input_buf, bias, output],
        (total_size + 255) // 256,
    )


def batch_add_residual(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_a: GPUBuffer2D,
    input_b: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """Add residual connection to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_a: First input tensor
        input_b: Second input tensor
        output: Output tensor

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
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        RESIDUAL_ADD_KERNEL,
        params,
        [input_a, input_b, output],
        (total_size + 255) // 256,
    )


def batch_add_cross_entropy_loss(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    logits: GPUBuffer2D,
    targets: GPUBuffer1D,
    loss_output: GPUBuffer1D,
    grad_logits: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
) -> None:
    """Add cross-entropy loss computation to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        logits: Model predictions (batch_size * seq_len, vocab_size)
        targets: Target token IDs (batch_size * seq_len,)
        loss_output: Loss for each prediction (batch_size * seq_len,)
        grad_logits: Gradient w.r.t. logits (batch_size * seq_len, vocab_size)
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
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        CROSS_ENTROPY_LOSS_KERNEL,
        params,
        [logits, targets, loss_output, grad_logits],
        (total_predictions + 255) // 256,
    )


def batch_add_copy(
    batch_state: BatchState, source: GPUBuffer2D, dest: GPUBuffer2D
) -> None:
    """Add buffer copy operation to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        batch_state: Batch state (MUTATED)
        source: Source buffer
        dest: Destination buffer

    Raises:
        ValueError: If buffer sizes don't match
        RuntimeError: If batch not initialized
    """
    if source.size != dest.size:
        raise ValueError(f"Buffer sizes must match: {source.size} != {dest.size}")

    if batch_state.encoder is None:
        raise RuntimeError("Must call create_command_batch before adding operations")

    batch_state.encoder.copy_buffer_to_buffer(
        source.buffer, 0, dest.buffer, 0, source.size * 4
    )
    batch_state.operation_count += 1


# ============================================================================
# BACKWARD PASS BATCH OPERATIONS
# ============================================================================


def batch_add_matmul_backward_a(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    grad_C: GPUBuffer2D,
    B: GPUBuffer2D,
    grad_A: GPUBuffer2D,
) -> None:
    """Add matmul backward A to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        grad_C: Gradient of loss w.r.t. C (M, N)
        B: Forward pass B matrix (K, N)
        grad_A: Output gradient w.r.t. A (M, K)

    Raises:
        ValueError: If dimensions are incompatible
    """
    M, N = grad_C.shape
    K, N2 = B.shape

    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"Invalid dimensions: grad_C=({M}, {N}), B=({K}, {N2})")

    if N != N2:
        raise ValueError(f"Dimension mismatch: grad_C.shape[1]={N} != B.shape[1]={N2}")

    validate_buffer_shape_2d(grad_A, (M, K), "grad_A")

    params = np.array([M, K, N], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        MATMUL_BACKWARD_A_KERNEL,
        params,
        [grad_C, B, grad_A],
        (K + 15) // 16,
        (M + 15) // 16,
        1,
    )


def batch_add_matmul_backward_b(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    A: GPUBuffer2D,
    grad_C: GPUBuffer2D,
    grad_B: GPUBuffer2D,
) -> None:
    """Add matmul backward B to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        A: Forward pass A matrix (M, K)
        grad_C: Gradient of loss w.r.t. C (M, N)
        grad_B: Output gradient w.r.t. B (K, N)

    Raises:
        ValueError: If dimensions are incompatible
    """
    M, K = A.shape
    M2, N = grad_C.shape

    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError(f"Invalid dimensions: A=({M}, {K}), grad_C=({M2}, {N})")

    if M != M2:
        raise ValueError(f"Dimension mismatch: A.shape[0]={M} != grad_C.shape[0]={M2}")

    validate_buffer_shape_2d(grad_B, (K, N), "grad_B")

    params = np.array([M, K, N], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        MATMUL_BACKWARD_B_KERNEL,
        params,
        [A, grad_C, grad_B],
        (N + 15) // 16,
        (K + 15) // 16,
        1,
    )


def batch_add_layernorm_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
    grad_gamma: GPUBuffer1D,
    grad_beta: GPUBuffer1D,
) -> None:
    """Add layernorm backward to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Also MUTATES grad_gamma and grad_beta by zeroing them first.
    Returns None to signal mutation.

    NOTE: This function automatically zeros grad_gamma and grad_beta before
    adding the operation, so caller does not need to pre-zero them.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_buf: Input from forward pass (n_elements, size)
        gamma: Scale parameters from forward pass (size,)
        grad_output: Gradient of loss w.r.t. output (n_elements, size)
        grad_input: Output gradient w.r.t. input (n_elements, size)
        grad_gamma: Output gradient w.r.t. gamma (size,) (MUTATED, auto-zeroed)
        grad_beta: Output gradient w.r.t. beta (size,) (MUTATED, auto-zeroed)

    Raises:
        ValueError: If buffer shapes don't match
    """
    n_elements, size = input_buf.shape

    if n_elements <= 0 or size <= 0:
        raise ValueError(f"Invalid input shape: ({n_elements}, {size})")

    validate_buffer_shape_1d(gamma, size, "gamma")
    validate_buffer_shape_2d(grad_output, (n_elements, size), "grad_output")
    validate_buffer_shape_2d(grad_input, (n_elements, size), "grad_input")
    validate_buffer_shape_1d(grad_gamma, size, "grad_gamma")
    validate_buffer_shape_1d(grad_beta, size, "grad_beta")

    # Zero accumulation buffers (safer API)
    clear_buffer(grad_gamma)
    clear_buffer(grad_beta)

    params = np.array([size, n_elements], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        LAYERNORM_BACKWARD_KERNEL,
        params,
        [input_buf, gamma, grad_output, grad_input, grad_gamma, grad_beta],
        n_elements,
    )


def batch_add_gelu_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
) -> None:
    """Add GELU backward to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        input_buf: Input from forward pass
        grad_output: Gradient of loss w.r.t. output
        grad_input: Output gradient w.r.t. input

    Raises:
        ValueError: If buffer shapes don't match
    """
    if input_buf.shape != grad_output.shape:
        raise ValueError(
            f"input_buf shape {input_buf.shape} doesn't match grad_output shape {grad_output.shape}"
        )

    if grad_input.shape != input_buf.shape:
        raise ValueError(
            f"grad_input shape {grad_input.shape} doesn't match input_buf shape {input_buf.shape}"
        )

    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        GELU_BACKWARD_KERNEL,
        params,
        [input_buf, grad_output, grad_input],
        (total_size + 255) // 256,
    )


def batch_add_bias_backward(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    grad_output: GPUBuffer2D,
    grad_bias: GPUBuffer1D,
) -> None:
    """Add bias backward to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        grad_output: Gradient of loss w.r.t. output (n_elements, dim)
        grad_bias: Output gradient w.r.t. bias (dim,)

    Raises:
        ValueError: If buffer shapes don't match
    """
    n_elements, dim = grad_output.shape

    if n_elements <= 0 or dim <= 0:
        raise ValueError(f"Invalid grad_output shape: ({n_elements}, {dim})")

    validate_buffer_shape_1d(grad_bias, dim, "grad_bias")

    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        BIAS_BACKWARD_KERNEL,
        params,
        [grad_output, grad_bias],
        (dim + 255) // 256,
    )


# ============================================================================
# OPTIMIZER BATCH OPERATIONS
# ============================================================================


def batch_add_adamw_update(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    gradients: GPUBuffer2D,
    weights: GPUBuffer2D,
    m: GPUBuffer2D,
    v: GPUBuffer2D,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    step: int,
) -> None:
    """Add AdamW optimizer update to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        gradients: Gradient buffer (same shape as weights)
        weights: Parameter buffer
        m: First moment (momentum) buffer
        v: Second moment (variance) buffer
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        weight_decay: Weight decay coefficient
        eps: Small constant for numerical stability
        step: Current training step (1-indexed)

    Raises:
        AssertionError: If buffer shapes don't match
        ValueError: If step is invalid
    """
    size = validate_optimizer_buffers(gradients, weights, m, v, step)

    params = np.array(
        [lr, beta1, beta2, weight_decay, eps, float(step), size],
        dtype=np.float32,
    )

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        ADAMW_OPTIMIZER_KERNEL,
        params,
        [gradients, weights, m, v],
        (size + 255) // 256,
    )


def batch_add_adamw_update_1d(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    gradients: GPUBuffer1D,
    weights: GPUBuffer1D,
    m: GPUBuffer1D,
    v: GPUBuffer1D,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    step: int,
) -> None:
    """Add AdamW optimizer update for 1D buffers to batch (mutation).

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        gradients: Gradient buffer (same size as weights)
        weights: Parameter buffer
        m: First moment (momentum) buffer
        v: Second moment (variance) buffer
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        weight_decay: Weight decay coefficient
        eps: Small constant for numerical stability
        step: Current training step (1-indexed)

    Raises:
        AssertionError: If buffer sizes don't match
        ValueError: If step is invalid
    """
    size = validate_optimizer_buffers(gradients, weights, m, v, step)

    params = np.array(
        [lr, beta1, beta2, weight_decay, eps, float(step), size],
        dtype=np.float32,
    )

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        ADAMW_OPTIMIZER_KERNEL,
        params,
        [gradients, weights, m, v],
        (size + 255) // 256,
    )


# ============================================================================
# BATCH SUBMISSION
# ============================================================================


def submit_batch(batch_state: BatchState) -> None:
    """Submit all batched operations (mutation).

    This function MUTATES batch_state by submitting and clearing the encoder.
    Returns None to signal mutation.

    Args:
        batch_state: Batch state (MUTATED)

    Raises:
        RuntimeError: If batch not initialized or already submitted
    """
    if batch_state.encoder is None:
        raise RuntimeError("Batch already submitted or not initialized")

    command_buffer = batch_state.encoder.finish()
    batch_state.device.wgpu_device.queue.submit([command_buffer])

    # Clear encoder to prevent reuse (mutation)
    batch_state.encoder = None
    batch_state.retained_buffers.clear()
