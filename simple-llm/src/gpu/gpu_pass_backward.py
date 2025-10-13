import numpy as np

from .gpu_buffer import gpu_buffer_zerofy
from .gpu_kernels import (
    get_bias_backward_kernel,
    get_bias_backward_reduce_kernel,
    get_dropout_backward_kernel,
    get_embedding_backward_kernel,
    get_embedding_backward_reduce_kernel,
    get_flash_attention_backward_kernel,
    get_flash_attention_backward_reduce_kernel,
    get_gelu_backward_kernel,
    get_layernorm_backward_kernel,
    get_layernorm_backward_reduce_accumulate_kernel,
    get_layernorm_backward_reduce_kernel,
    get_matmul_backward_a_kernel,
    get_matmul_backward_b_kernel,
)
from .gpu_ops import (
    batch_add,
    validate_buffer_shape_1d,
    validate_buffer_shape_2d,
)
from .gpu_types import (
    GPUBuffer1D,
    GPUBuffer2D,
    GPUContext,
)

# ============================================================================
# BACKWARD PASS OPERATIONS
# ============================================================================

"""Backward pass operations - individual kernel dispatches

MUTATION SEMANTICS:
- All backward operations MUTATE their output gradient buffers
- Accumulation buffers (grad_gamma, grad_beta, grad_bias) are automatically
  zeroed before kernel dispatch - caller does NOT need to pre-zero them
- Workspace buffers (grad_input, etc.) are NOT auto-zeroed - caller must
  ensure these are properly initialized if reusing buffers
"""


def matmul_backward_a(
    ctx: GPUContext,
    grad_C: GPUBuffer2D,
    B: GPUBuffer2D,
    grad_A: GPUBuffer2D,
) -> None:
    """Compute gradient w.r.t. A: grad_A = grad_C @ B^T.

    Args:
        grad_C: Gradient of loss w.r.t. C (M, N)
        B: Forward pass B matrix (K, N)
        grad_A: Output gradient w.r.t. A (M, K)

    Raises:
        ValueError: If dimensions are incompatible or invalid
    """
    M, N = grad_C.shape
    K, N2 = B.shape

    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"Invalid dimensions: grad_C=({M}, {N}), B=({K}, {N2})")

    if N != N2:
        raise ValueError(f"Dimension mismatch: grad_C.shape[1]={N} != B.shape[1]={N2}")

    TILE_SIZE = ctx.config.matmul_tile_size

    validate_buffer_shape_2d(grad_A, (M, K), "grad_A")

    params = np.array([M, N, K], dtype=np.uint32)
    batch_add(
        ctx,
        get_matmul_backward_a_kernel(ctx),
        params,
        [grad_C, B, grad_A],
        (K + TILE_SIZE - 1) // TILE_SIZE,
        (M + TILE_SIZE - 1) // TILE_SIZE,
        1,
    )


def matmul_backward_b(
    ctx: GPUContext,
    A: GPUBuffer2D,
    grad_C: GPUBuffer2D,
    grad_B: GPUBuffer2D,
) -> None:
    """Compute gradient w.r.t. B: grad_B = A^T @ grad_C.

    Args:
        A: Forward pass A matrix (M, K)
        grad_C: Gradient of loss w.r.t. C (M, N)
        grad_B: Output gradient w.r.t. B (K, N)

    Raises:
        ValueError: If dimensions are incompatible or invalid
    """
    M, K = A.shape
    M2, N = grad_C.shape

    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError(f"Invalid dimensions: A=({M}, {K}), grad_C=({M2}, {N})")

    if M != M2:
        raise ValueError(f"Dimension mismatch: A.shape[0]={M} != grad_C.shape[0]={M2}")

    validate_buffer_shape_2d(grad_B, (K, N), "grad_B")

    TILE_SIZE = ctx.config.matmul_tile_size
    params = np.array([M, K, N], dtype=np.uint32)
    batch_add(
        ctx,
        get_matmul_backward_b_kernel(ctx),
        params,
        [A, grad_C, grad_B],
        (N + TILE_SIZE - 1) // TILE_SIZE,
        (K + TILE_SIZE - 1) // TILE_SIZE,
        1,
    )


def embedding_backward(
    ctx: GPUContext,
    input_ids: GPUBuffer1D,
    grad_output: GPUBuffer2D,
    grad_embedding: GPUBuffer2D,
    reduction_workspace: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
) -> None:
    """
    Embedding backward pass using workspace pattern.

    Two-stage process:
    1. Copy grad_output to workspace (flat indexing, parallelized)
    2. Reduce workspace to grad_embedding (2D dispatch: vocab × dim)

    Args:
        ctx: GPU context
        input_ids: Input token IDs [batch_size*seq_len]
        grad_output: Gradient w.r.t. embeddings [batch_size*seq_len, embedding_dim]
        grad_embedding: Output gradient w.r.t. embedding table [vocab_size, embedding_dim]
        reduction_workspace: Workspace [batch_size*seq_len, embedding_dim]
        batch_size: Batch size
        seq_len: Sequence length

    Raises:
        ValueError: If buffer shapes don't match
    """
    import numpy as np

    from .gpu_buffer import gpu_buffer_zerofy
    from .gpu_ops import batch_add, validate_buffer_shape_1d, validate_buffer_shape_2d

    total_tokens, embedding_dim = grad_output.shape
    vocab_size, embedding_dim2 = grad_embedding.shape

    if embedding_dim != embedding_dim2:
        raise ValueError(f"Embedding dim mismatch: {embedding_dim} != {embedding_dim2}")

    if total_tokens != batch_size * seq_len:
        raise ValueError(
            f"Total tokens {total_tokens} != batch_size*seq_len {batch_size * seq_len}"
        )

    validate_buffer_shape_1d(input_ids, total_tokens, "input_ids")
    validate_buffer_shape_2d(grad_output, (total_tokens, embedding_dim), "grad_output")
    validate_buffer_shape_2d(
        reduction_workspace, (total_tokens, embedding_dim), "reduction_workspace"
    )

    # Zero the output buffer
    gpu_buffer_zerofy(ctx, grad_embedding)

    # Stage 1: Copy grad_output to workspace (FLAT indexing, parallel over all elements)
    total_elements = total_tokens * embedding_dim
    stage1_params = np.array([total_tokens, embedding_dim, 0, 0], dtype=np.uint32)

    batch_add(
        ctx,
        get_embedding_backward_kernel(ctx),
        stage1_params,
        [grad_output, reduction_workspace],
        (total_elements + 255) // 256,  # FIXED: Flat indexing over all elements
        1,
        1,
    )

    # Stage 2: Reduce workspace to embedding gradients (2D dispatch: vocab_id × dim_idx)
    stage2_params = np.array(
        [total_tokens, embedding_dim, vocab_size, 0], dtype=np.uint32
    )

    batch_add(
        ctx,
        get_embedding_backward_reduce_kernel(ctx),
        stage2_params,
        [input_ids, reduction_workspace, grad_embedding],
        vocab_size,  # FIXED: workgroup_id.x = vocab_id
        embedding_dim,  # FIXED: workgroup_id.y = dim_idx
        1,
    )


def dropout_backward(
    ctx: GPUContext,
    grad_output: GPUBuffer2D,
    mask: GPUBuffer2D,
    grad_input: GPUBuffer2D,
    keep_prob: float,
) -> None:
    """Dropout backward using saved mask.

    Args:
        grad_output: Gradient w.r.t. output [rows, cols]
        mask: Mask from forward pass [rows, cols]
        grad_input: Output gradient w.r.t. input [rows, cols]
        keep_prob: Probability of keeping each element (from forward pass)

    Raises:
        ValueError: If buffer shapes don't match
    """
    rows, cols = grad_output.shape

    validate_buffer_shape_2d(grad_input, (rows, cols), "grad_input")
    validate_buffer_shape_2d(mask, (rows, cols), "mask")

    total_size = rows * cols
    params = np.array([total_size, keep_prob], dtype=np.float32)

    batch_add(
        ctx,
        get_dropout_backward_kernel(ctx),
        params,
        [grad_output, mask, grad_input],
        (total_size + 255) // 256,
    )


def layernorm_backward(
    ctx: GPUContext,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
    grad_gamma: GPUBuffer1D,
    grad_beta: GPUBuffer1D,
    reduction_workspace_gamma: GPUBuffer2D,
    reduction_workspace_beta: GPUBuffer2D,
    accumulate: bool = False,
) -> None:
    """
    Backward pass for layer normalization using workspace pattern.

    Uses two-stage reduction to avoid race conditions:
    - Stage 1: Each workgroup computes partial gamma/beta gradients → workspace
    - Stage 2: Reduction kernel sums partial gradients into final result

    Example:
        # Gradient accumulation (multiple mini-batches)
        layernorm_backward(ctx, x1, gamma, grad_out1, grad_in1, grad_g, grad_b, ws_g, ws_b, accumulate=False)
        layernorm_backward(ctx, x2, gamma, grad_out2, grad_in2, grad_g, grad_b, ws_g, ws_b, accumulate=True)

    Args:
        ctx: GPU context
        input_buf: Input from forward pass (n_elements, size)
        gamma: Scale parameters from forward pass (size,)
        grad_output: Gradient of loss w.r.t. output (n_elements, size)
        grad_input: Output gradient w.r.t. input (n_elements, size)
        grad_gamma: Output gradient w.r.t. gamma (size,)
        grad_beta: Output gradient w.r.t. beta (size,)
        reduction_workspace_gamma: Workspace for partial gamma gradients (n_elements, size)
        reduction_workspace_beta: Workspace for partial beta gradients (n_elements, size)
        accumulate: If False (default), zeros grad_gamma/grad_beta before operation.
                   If True, accumulates into existing values.

    Raises:
        ValueError: If buffer shapes don't match
    """
    n_elements, size = input_buf.shape

    if n_elements == 0 or size == 0:
        raise ValueError(f"Invalid input_buf shape: {(n_elements, size)}")

    validate_buffer_shape_1d(gamma, size, "gamma")
    validate_buffer_shape_2d(grad_output, (n_elements, size), "grad_output")
    validate_buffer_shape_2d(grad_input, (n_elements, size), "grad_input")
    validate_buffer_shape_1d(grad_gamma, size, "grad_gamma")
    validate_buffer_shape_1d(grad_beta, size, "grad_beta")
    validate_buffer_shape_2d(
        reduction_workspace_gamma, (n_elements, size), "reduction_workspace_gamma"
    )
    validate_buffer_shape_2d(
        reduction_workspace_beta, (n_elements, size), "reduction_workspace_beta"
    )

    # Stage 1: Compute partial gradients
    params = np.array([size, n_elements], dtype=np.uint32)

    batch_add(
        ctx,
        get_layernorm_backward_kernel(ctx),
        params,
        [
            input_buf,
            gamma,
            grad_output,
            grad_input,
            reduction_workspace_gamma,
            reduction_workspace_beta,
        ],
        n_elements,
    )

    # Stage 2: Reduce partial gradients
    if accumulate:
        # Use atomic accumulation kernel
        reduction_kernel = get_layernorm_backward_reduce_accumulate_kernel(ctx)
    else:
        gpu_buffer_zerofy(ctx, grad_gamma)
        gpu_buffer_zerofy(ctx, grad_beta)
        reduction_kernel = get_layernorm_backward_reduce_kernel(ctx)

    params_reduce = np.array([size, n_elements], dtype=np.uint32)

    batch_add(
        ctx,
        reduction_kernel,
        params_reduce,
        [reduction_workspace_gamma, reduction_workspace_beta, grad_gamma, grad_beta],
        size,
    )


def gelu_backward(
    ctx: GPUContext,
    input_buf: GPUBuffer2D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
) -> None:
    """Backward pass for GELU activation

    Args:
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
    batch_add(
        ctx,
        get_gelu_backward_kernel(ctx),
        params,
        [input_buf, grad_output, grad_input],
        (total_size + 255) // 256,
    )


def bias_backward(
    ctx: GPUContext,
    grad_output: GPUBuffer2D,
    grad_bias: GPUBuffer1D,
    reduction_workspace: GPUBuffer1D,
    accumulate: bool = False,
) -> None:
    """
    Backward pass for bias - sum gradients over batch using two-stage reduction.

    Stage 1: Each workgroup computes partial sums for bias dimensions
    Stage 2: Reduce all partials to final gradient

    Example:
        # Gradient accumulation (multiple mini-batches)
        bias_backward(ctx, grad_out1, grad_bias, workspace, accumulate=False)
        bias_backward(ctx, grad_out2, grad_bias, workspace, accumulate=True)

    Args:
        ctx: GPU context
        grad_output: Gradient of loss w.r.t. output (n_elements, dim)
        grad_bias: Output gradient w.r.t. bias (dim,)
        reduction_workspace: Workspace for partial reductions
        accumulate: If False, zeros grad_bias before operation

    Raises:
        ValueError: If buffer shapes don't match
    """
    n_elements, dim = grad_output.shape

    if n_elements == 0 or dim == 0:
        raise ValueError(f"Invalid grad_output shape: {(n_elements, dim)}")

    validate_buffer_shape_1d(grad_bias, dim, "grad_bias")

    # Zero accumulation buffer only if not accumulating
    if not accumulate:
        gpu_buffer_zerofy(ctx, grad_bias)

    # Stage 1: Compute partial sums
    workgroup_size = 256
    num_workgroups = (n_elements + workgroup_size - 1) // workgroup_size
    total_partials = num_workgroups * dim

    # Validate workspace size
    validate_buffer_shape_1d(reduction_workspace, total_partials, "reduction_workspace")

    params = np.array([n_elements, dim], dtype=np.uint32)

    batch_add(
        ctx,
        get_bias_backward_kernel(ctx),
        params,
        [grad_output, reduction_workspace],
        num_workgroups,
    )

    # Stage 2: Reduce partials to final gradient
    reduce_params = np.array([num_workgroups, dim], dtype=np.uint32)

    batch_add(
        ctx,
        get_bias_backward_reduce_kernel(ctx),
        reduce_params,
        [reduction_workspace, grad_bias],
        dim,  # One workgroup per bias dimension
    )


def flash_attention_backward(
    ctx: GPUContext,
    Q: GPUBuffer2D,
    K: GPUBuffer2D,
    V: GPUBuffer2D,
    O: GPUBuffer2D,
    L: GPUBuffer1D,
    M: GPUBuffer1D,
    grad_O: GPUBuffer2D,
    grad_Q: GPUBuffer2D,
    grad_K_workspace: GPUBuffer2D,
    grad_V_workspace: GPUBuffer2D,
    grad_K: GPUBuffer2D,
    grad_V: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
) -> None:
    """
    Compute gradients for flash attention using two-stage reduction.

    Stage 1: Each Q-block computes partial grad_K and grad_V contributions
    Stage 2: Reduce all partials to final gradients

    Args:
        ctx: GPU context
        Q: Query from forward pass [batch_size*seq_len, n_heads*head_dim]
        K: Key from forward pass [batch_size*seq_len, n_heads*head_dim]
        V: Value from forward pass [batch_size*seq_len, n_heads*head_dim]
        O: Output from forward pass [batch_size*seq_len, n_heads*head_dim]
        L: Softmax normalization from forward [batch_size*seq_len*n_heads]
        M: Max values from forward [batch_size*seq_len*n_heads]
        grad_O: Gradient w.r.t. output [batch_size*seq_len, n_heads*head_dim]
        grad_Q: Output gradient w.r.t. Q [batch_size*seq_len, n_heads*head_dim]
        grad_K_workspace: Workspace for partial grad_K [num_q_blocks*batch_size*seq_len, n_heads*head_dim]
        grad_V_workspace: Workspace for partial grad_V [num_q_blocks*batch_size*seq_len, n_heads*head_dim]
        grad_K: Output gradient w.r.t. K [batch_size*seq_len, n_heads*head_dim]
        grad_V: Output gradient w.r.t. V [batch_size*seq_len, n_heads*head_dim]
        batch_size: Batch size
        seq_len: Sequence length
        n_heads: Number of attention heads
        head_dim: Dimension per head

    Raises:
        ValueError: If buffer shapes don't match or dimensions invalid
    """
    config = ctx.config

    # Validate head_dim against config
    if head_dim > config.flash_attn_max_head_dim:
        raise ValueError(
            f"head_dim {head_dim} exceeds maximum supported by config: "
            f"{config.flash_attn_max_head_dim}"
        )

    if batch_size <= 0 or seq_len <= 0 or n_heads <= 0 or head_dim <= 0:
        raise ValueError(
            f"Invalid dimensions: batch_size={batch_size}, seq_len={seq_len}, "
            f"n_heads={n_heads}, head_dim={head_dim}"
        )

    embedding_dim = n_heads * head_dim
    total_tokens = batch_size * seq_len
    stats_size = batch_size * seq_len * n_heads

    validate_buffer_shape_2d(Q, (total_tokens, embedding_dim), "Q")
    validate_buffer_shape_2d(K, (total_tokens, embedding_dim), "K")
    validate_buffer_shape_2d(V, (total_tokens, embedding_dim), "V")
    validate_buffer_shape_2d(O, (total_tokens, embedding_dim), "O")
    validate_buffer_shape_1d(L, stats_size, "L")
    validate_buffer_shape_1d(M, stats_size, "M")
    validate_buffer_shape_2d(grad_O, (total_tokens, embedding_dim), "grad_O")
    validate_buffer_shape_2d(grad_Q, (total_tokens, embedding_dim), "grad_Q")
    validate_buffer_shape_2d(grad_K, (total_tokens, embedding_dim), "grad_K")
    validate_buffer_shape_2d(grad_V, (total_tokens, embedding_dim), "grad_V")

    Br = config.flash_attn_br
    num_q_blocks = (seq_len + Br - 1) // Br

    # Validate workspace shapes
    workspace_tokens = num_q_blocks * total_tokens
    validate_buffer_shape_2d(
        grad_K_workspace, (workspace_tokens, embedding_dim), "grad_K_workspace"
    )
    validate_buffer_shape_2d(
        grad_V_workspace, (workspace_tokens, embedding_dim), "grad_V_workspace"
    )

    # Stage 1: Compute partial gradients per Q-block
    # Each Q-block writes to its dedicated workspace slice
    for q_block_idx in range(num_q_blocks):
        params = np.array(
            [
                batch_size,
                seq_len,
                n_heads,
                head_dim,
                config.flash_attn_bc,
                config.flash_attn_br,
                q_block_idx,  # Current Q-block index for workspace offset
            ],
            dtype=np.uint32,
        )

        batch_add(
            ctx,
            get_flash_attention_backward_kernel(ctx),
            params,
            [grad_O, Q, K, V, O, L, M, grad_Q, grad_K_workspace, grad_V_workspace],
            1,  # Process one Q-block at a time
            n_heads,
            batch_size,
        )

    # Stage 2: Reduce workspace partials to final gradients
    reduce_params = np.array(
        [
            batch_size,
            seq_len,
            n_heads,
            head_dim,
            num_q_blocks,
        ],
        dtype=np.uint32,
    )

    batch_add(
        ctx,
        get_flash_attention_backward_reduce_kernel(ctx),
        reduce_params,
        [grad_K_workspace, grad_V_workspace, grad_K, grad_V],
        total_tokens,  # One workgroup per token position
        1,
        1,
    )
