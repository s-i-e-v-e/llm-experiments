"""Kernel dispatches"""

from typing import List, Tuple

import numpy as np
from gpu_buffer import clear_buffer
from gpu_device import (
    create_bind_group_entries,
    get_or_create_pipeline,
    wgpu,
)
from gpu_kernels import (
    get_attention_backward_kernel_from_config,
    get_attention_kernel_from_config,
    get_bias_add_kernel_from_config,
    get_bias_backward_kernel_from_config,
    get_embedding_kernel_from_config,
    get_extract_last_tokens_kernel_from_config,
    get_flash_attention_backward_kernel_from_config,
    get_flash_attention_kernel_from_config,
    get_gelu_backward_kernel_from_config,
    get_gelu_kernel_from_config,
    get_layernorm_backward_kernel_from_config,
    get_layernorm_backward_reduce_kernel_from_config,
    get_layernorm_kernel_from_config,
    get_matmul_backward_a_kernel_from_config,
    get_matmul_backward_b_kernel_from_config,
    get_matmul_kernel_from_config,
    get_residual_add_kernel_from_config,
)
from gpu_types import (
    BatchState,
    BindGroupEntry,
    Device,
    GPUBuffer1D,
    GPUBuffer2D,
    GPUBufferAny,
    PipelineCache,
    WGPUBindGroup,
    WGPUBuffer,
    WGPUComputePipeline,
)

# ============================================================================
# VALIDATION
# ============================================================================


def validate_buffer_shape_2d(
    buffer: GPUBuffer2D, expected_shape: Tuple[int, int], name: str
) -> None:
    """Validate 2D buffer has expected shape.

    This function does NOT mutate buffer.

    Args:
        buffer: Buffer to validate
        expected_shape: Expected (rows, cols)
        name: Buffer name for error messages

    Raises:
        ValueError: If shape doesn't match or dimensions invalid
    """
    if expected_shape[0] <= 0 or expected_shape[1] <= 0:
        raise ValueError(f"Invalid expected shape for {name}: {expected_shape}")

    if buffer.shape != expected_shape:
        raise ValueError(
            f"{name} shape mismatch: got {buffer.shape}, expected {expected_shape}"
        )


def validate_buffer_shape_1d(
    buffer: GPUBuffer1D, expected_size: int, name: str
) -> None:
    """Validate 1D buffer has expected size.

    This function does NOT mutate buffer.

    Args:
        buffer: Buffer to validate
        expected_size: Expected size
        name: Buffer name for error messages

    Raises:
        ValueError: If size doesn't match or invalid
    """
    if expected_size <= 0:
        raise ValueError(f"Invalid expected size for {name}: {expected_size}")

    if buffer.shape != (expected_size,):
        raise ValueError(
            f"{name} shape mismatch: got {buffer.shape}, expected ({expected_size},)"
        )


def validate_matmul_shapes(
    A: GPUBuffer2D, B: GPUBuffer2D, C: GPUBuffer2D, operation: str
) -> Tuple[int, int, int]:
    """Validate shapes for matrix multiplication operations.

    This function does NOT mutate any buffers.

    Args:
        A: First input matrix
        B: Second input matrix
        C: Output matrix
        operation: Operation name for error messages

    Returns:
        Tuple of (M, K, N) dimensions

    Raises:
        ValueError: If shapes are incompatible
    """
    M, K = A.shape
    K2, N = B.shape

    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError(f"{operation}: Invalid dimensions A=({M}, {K}), B=({K2}, {N})")

    if K != K2:
        raise ValueError(
            f"{operation}: Dimension mismatch A.shape[1]={K} != B.shape[0]={K2}"
        )

    if C.shape != (M, N):
        raise ValueError(
            f"{operation}: Output shape {C.shape} doesn't match expected ({M}, {N})"
        )

    return M, K, N


def validate_optimizer_buffers(
    gradients: GPUBufferAny,
    weights: GPUBufferAny,
    m: GPUBufferAny,
    v: GPUBufferAny,
    step: int,
) -> int:
    """Validate optimizer buffer shapes and step.

    This function does NOT mutate any buffers.

    Args:
        gradients: Gradient buffer
        weights: Weight buffer
        m: Momentum buffer
        v: Variance buffer
        step: Training step number

    Returns:
        Buffer size

    Raises:
        AssertionError: If shapes don't match
        ValueError: If step is invalid
    """
    if step < 1:
        raise ValueError(f"Step must be >= 1, got {step}")

    assert gradients.size == weights.size, (
        f"Size mismatch: gradients={gradients.size} != weights={weights.size}"
    )
    assert weights.size == m.size == v.size, (
        f"All buffers must have same size: weights={weights.size}, m={m.size}, v={v.size}"
    )

    return weights.size


def _validate_buffer_shapes(
    buffers: List[Tuple[GPUBufferAny, Tuple[int, ...], str]],
) -> None:
    """Validate buffer shapes match expected dimensions.

    This function does NOT mutate buffers.

    Args:
        buffers: List of (buffer, expected_shape, name) tuples

    Raises:
        ValueError: If any buffer shape doesn't match expected
    """
    for buffer, expected_shape, name in buffers:
        if buffer.shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: got {buffer.shape}, expected {expected_shape}"
            )


# ============================================================================
# INFRASTRUCTURE
# ============================================================================


def _create_uniform_buffer_internal(
    pipeline_cache: PipelineCache, data: np.ndarray
) -> WGPUBuffer:
    """Internal: Create uniform buffer for parameters.

    This function does NOT mutate pipeline_cache or data.

    Memory management: Buffer is automatically freed by WGPU after GPU finishes
    using it (typically after queue submission completes). No explicit cleanup needed.

    Args:
        pipeline_cache: Pipeline cache state
        data: Numpy array of parameter data

    Returns:
        New WGPU uniform buffer
    """
    return pipeline_cache.device.wgpu_device.create_buffer_with_data(
        data=data, usage=wgpu.BufferUsage.UNIFORM
    )


def _create_bind_group_internal(
    pipeline_cache: PipelineCache,
    pipeline: WGPUComputePipeline,
    entries: List[BindGroupEntry],
) -> WGPUBindGroup:
    """Internal: Create bind group using type-safe entries.

    This function does NOT mutate any inputs.

    Args:
        pipeline_cache: Pipeline cache state
        pipeline: Compute pipeline
        entries: Bind group entry specifications

    Returns:
        New WGPU bind group
    """
    return pipeline_cache.device.wgpu_device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=create_bind_group_entries(entries),
    )


def _dispatch_compute_internal(
    pipeline_cache: PipelineCache,
    pipeline: WGPUComputePipeline,
    bind_group: WGPUBindGroup,
    workgroups_x: int,
    workgroups_y: int = 1,
    workgroups_z: int = 1,
) -> None:
    """Internal: Create encoder and dispatch compute pass.

    Uiform buffers created in this function are automatically freed
    by WGPU after the queue submission completes. No explicit cleanup needed.

    This function does NOT mutate pipeline_cache.

    Args:
        pipeline_cache: Pipeline cache state
        pipeline: Compute pipeline
        bind_group: Bind group
        workgroups_x: Number of workgroups in X
        workgroups_y: Number of workgroups in Y
        workgroups_z: Number of workgroups in Z
    """
    encoder = pipeline_cache.device.wgpu_device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z)
    compute_pass.end()
    pipeline_cache.device.wgpu_device.queue.submit([encoder.finish()])


def dispatch_simple_compute(
    pipelinecache: PipelineCache,
    kernel_code: str,
    params: np.ndarray,
    buffers: List[GPUBufferAny],
    workgroups_x: int,
    workgroups_y: int = 1,
    workgroups_z: int = 1,
) -> None:
    """
    Unified compute dispatch - eliminates repetitive pipeline/bind group/dispatch pattern

    This function may MUTATE pipelinecache by adding cached pipelines.
    This function does NOT mutate params or buffers.

    Memory management: Uniform parameter buffer is created temporarily and automatically
    freed by WGPU after GPU completes execution. No memory leak.

    Args:
        pipelinecache: Pipeline cache state (may be MUTATED for caching)
        kernel_code: WGSL kernel source code
        params: Numpy array of parameters (uploaded as uniform buffer at binding 0)
        buffers: List of GPU buffers to bind (sequential bindings starting at 1)
        workgroups_x: Number of workgroups in X dimension
        workgroups_y: Number of workgroups in Y dimension (default 1)
        workgroups_z: Number of workgroups in Z dimension (default 1)

    Raises:
        ValueError: If workgroup counts exceed device limits
    """
    # Validate workgroup counts against config limits
    max_workgroups = pipelinecache.device.config.max_workgroups_per_dim

    if workgroups_x > max_workgroups:
        raise ValueError(
            f"workgroups_x ({workgroups_x}) exceeds maximum ({max_workgroups}). "
            f"Consider tiling the computation."
        )

    if workgroups_y > max_workgroups:
        raise ValueError(
            f"workgroups_y ({workgroups_y}) exceeds maximum ({max_workgroups})"
        )

    if workgroups_z > max_workgroups:
        raise ValueError(
            f"workgroups_z ({workgroups_z}) exceeds maximum ({max_workgroups})"
        )

    # Create uniform buffer for parameters
    params_buffer = _create_uniform_buffer_internal(pipelinecache, params)

    # Get or create pipeline
    pipeline = get_or_create_pipeline(pipelinecache, kernel_code)

    # Build bind group entries: binding 0 is params, rest are buffers
    entries = [BindGroupEntry(0, params_buffer, 0, params.nbytes)]
    for i, buf in enumerate(buffers):
        binding_index = i + 1
        entries.append(BindGroupEntry(binding_index, buf.buffer, 0, buf.size * 4))

    bindgroup = _create_bind_group_internal(pipelinecache, pipeline, entries)

    # Dispatch compute
    _dispatch_compute_internal(
        pipelinecache, pipeline, bindgroup, workgroups_x, workgroups_y, workgroups_z
    )


def create_command_batch(device: Device, enable_profiling: bool = False) -> BatchState:
    """
    Create command batch state for batched GPU operations

    This function does NOT mutate device.

    Memory management: Uniform buffers created during batch operations are retained
    in batch_state.retained_buffers to keep them alive until submit_batch is called.

    Args:
        device: GPU device state
        enable_profiling: Whether to enable profiling for this batch

    Returns:
        New batch state with encoder ready for operations

    Raises:
        RuntimeError: If device not initialized or batch limit exceeded
    """
    encoder = device.wgpu_device.create_command_encoder()

    return BatchState(
        device=device,
        encoder=encoder,
        retained_buffers=[],
        enable_profiling=enable_profiling,
        operation_count=0,
    )


def _create_and_retain_uniform_buffer_internal(
    batch_state: BatchState, data: np.ndarray
) -> WGPUBuffer:
    """Internal: Create uniform buffer and add to retained list (mutation).

    This function MUTATES batch_state.retained_buffers by appending the new buffer.

    Memory management: Buffer is kept alive by batch_state.retained_buffers
    until submit_batch() is called.

    Args:
        batch_state: Batch state (MUTATED)
        data: Numpy array of data

    Returns:
        New WGPU uniform buffer
    """
    buffer = batch_state.device.wgpu_device.create_buffer_with_data(
        data=data, usage=wgpu.BufferUsage.UNIFORM
    )
    batch_state.retained_buffers.append(buffer)
    return buffer


def _create_bind_group_for_batch_internal(
    batch_state: BatchState,
    pipeline: WGPUComputePipeline,
    entries: List[BindGroupEntry],
) -> WGPUBindGroup:
    """Internal: Create bind group using type-safe entries.

    This function does NOT mutate batch_state.

    Args:
        batch_state: Batch state
        pipeline: Compute pipeline
        entries: Bind group entry specifications

    Returns:
        New WGPU bind group
    """
    return batch_state.device.wgpu_device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=create_bind_group_entries(entries),
    )


def _add_compute_to_batch_internal(
    pipelinecache: PipelineCache,
    batch_state: BatchState,
    kernel_code: str,
    params: np.ndarray,
    buffers: List[GPUBufferAny],
    workgroups_x: int,
    workgroups_y: int = 1,
    workgroups_z: int = 1,
) -> None:
    """
    Internal: Add compute operation to batch encoder (mutation)

    This function MUTATES batch_state by adding operation and retaining uniform buffer.

    Args:
        pipelinecache: Pipeline cache for kernel compilation
        batch_state: Batch state (MUTATED)
        kernel_code: WGSL kernel source
        params: Parameter array
        buffers: GPU buffers to bind
        workgroups_x: Workgroups in X
        workgroups_y: Workgroups in Y
        workgroups_z: Workgroups in Z

    Raises:
        RuntimeError: If batch operation limit exceeded
    """
    # Check batch operation limit from config
    max_ops = batch_state.device.config.max_batch_operations

    if batch_state.operation_count >= max_ops:
        raise RuntimeError(
            f"Batch operation limit ({max_ops}) exceeded. "
            f"Call submit_batch() to flush operations."
        )

    # Validate workgroup counts
    max_workgroups = batch_state.device.config.max_workgroups_per_dim

    if (
        workgroups_x > max_workgroups
        or workgroups_y > max_workgroups
        or workgroups_z > max_workgroups
    ):
        raise ValueError(
            f"Workgroup counts ({workgroups_x}, {workgroups_y}, {workgroups_z}) "
            f"exceed maximum ({max_workgroups})"
        )

    params_buffer = _create_and_retain_uniform_buffer_internal(batch_state, params)
    pipeline = get_or_create_pipeline(pipelinecache, kernel_code)

    # Build bind group entries
    entries = [BindGroupEntry(0, params_buffer, 0, params.nbytes)]
    for i, buf in enumerate(buffers):
        binding_index = i + 1
        entries.append(BindGroupEntry(binding_index, buf.buffer, 0, buf.size * 4))

    bindgroup = _create_bind_group_for_batch_internal(batch_state, pipeline, entries)

    compute_pass = batch_state.encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bindgroup)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z)
    compute_pass.end()

    batch_state.operation_count += 1


def submit_batch(batch_state: BatchState) -> None:
    """Submit all batched operations (mutation).

    This function MUTATES batch_state by clearing encoder and retained buffers.
    Returns None to signal mutation.

    Args:
        batch_state: Batch state (MUTATED)

    Raises:
        RuntimeError: If batch already submitted or not initialized
    """
    if batch_state.encoder is None:
        raise RuntimeError("Batch already submitted or not initialized")

    command_buffer = batch_state.encoder.finish()
    batch_state.device.wgpu_device.queue.submit([command_buffer])

    # Clear encoder and buffers
    batch_state.encoder = None
    batch_state.retained_buffers.clear()


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


def run_softmax(
    pipeline_cache: PipelineCache,
    logits: GPUBuffer2D,
    probs: GPUBuffer2D,
) -> None:
    """
    Apply softmax to logits to get probability distribution.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        logits: Input logits [batch_size, vocab_size]
        probs: Output probabilities [batch_size, vocab_size]

    Note:
        Uses numerically stable softmax with max subtraction.
        For generation/sampling after computing final layer logits.
    """
    from gpu_kernels import create_softmax_kernel
    from gpu_operations import (
        dispatch_simple_compute,
        validate_buffer_shape_2d,
    )

    # Validate shapes
    validate_buffer_shape_2d(logits)
    validate_buffer_shape_2d(probs)

    if logits.shape != probs.shape:
        raise ValueError(
            f"Softmax shape mismatch: logits {logits.shape} != probs {probs.shape}"
        )

    batch_size, vocab_size = logits.shape

    # Create uniform buffer with parameters
    params_data = bytearray(8)  # 2 u32s
    params_data[0:4] = batch_size.to_bytes(4, "little")
    params_data[4:8] = vocab_size.to_bytes(4, "little")

    # Generate kernel code
    kernel_code = create_softmax_kernel(workgroup_size=256)

    # Dispatch computation
    dispatch_simple_compute(
        device=pipeline_cache.device,
        pipeline_cache=pipeline_cache,
        kernel_name="softmax",
        kernel_code=kernel_code,
        buffers=[logits.buffer, probs.buffer],
        uniform_data=params_data,
        workgroup_count=(batch_size, 1, 1),
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
- Functions return None to signal mutation
"""


def run_matmul_backward_a(
    pipeline_cache: PipelineCache,
    grad_C: GPUBuffer2D,
    B: GPUBuffer2D,
    grad_A: GPUBuffer2D,
) -> None:
    """Compute gradient w.r.t. A: grad_A = grad_C @ B^T (mutation).

    This function MUTATES grad_A by writing the computed gradients.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        grad_C: Gradient of loss w.r.t. C (M, N)
        B: Forward pass B matrix (K, N)
        grad_A: Output gradient w.r.t. A (M, K) (MUTATED)

    Raises:
        ValueError: If dimensions are incompatible or invalid
    """
    M, N = grad_C.shape
    K, N2 = B.shape

    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"Invalid dimensions: grad_C=({M}, {N}), B=({K}, {N2})")

    if N != N2:
        raise ValueError(f"Dimension mismatch: grad_C.shape[1]={N} != B.shape[1]={N2}")

    if grad_A.shape != (M, K):
        raise ValueError(
            f"grad_A shape {grad_A.shape} doesn't match expected ({M}, {K})"
        )

    config = pipeline_cache.device.config
    TILE_SIZE = config.matmul_tile_size
    params = np.array([M, K, N], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        get_matmul_backward_a_kernel_from_config(config),
        params,
        [grad_C, B, grad_A],
        (K + TILE_SIZE - 1) // TILE_SIZE,
        (M + TILE_SIZE - 1) // TILE_SIZE,
        1,
    )


def run_matmul_backward_b(
    pipeline_cache: PipelineCache,
    A: GPUBuffer2D,
    grad_C: GPUBuffer2D,
    grad_B: GPUBuffer2D,
) -> None:
    """Compute gradient w.r.t. B: grad_B = A^T @ grad_C (mutation).

    This function MUTATES grad_B by writing the computed gradients.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        A: Forward pass A matrix (M, K)
        grad_C: Gradient of loss w.r.t. C (M, N)
        grad_B: Output gradient w.r.t. B (K, N) (MUTATED)

    Raises:
        ValueError: If dimensions are incompatible or invalid
    """
    M, K = A.shape
    M2, N = grad_C.shape

    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError(f"Invalid dimensions: A=({M}, {K}), grad_C=({M2}, {N})")

    if M != M2:
        raise ValueError(f"Dimension mismatch: A.shape[0]={M} != grad_C.shape[0]={M2}")

    if grad_B.shape != (K, N):
        raise ValueError(
            f"grad_B shape {grad_B.shape} doesn't match expected ({K}, {N})"
        )

    config = pipeline_cache.device.config
    TILE_SIZE = config.matmul_tile_size
    params = np.array([M, K, N], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        get_matmul_backward_b_kernel_from_config(config),
        params,
        [A, grad_C, grad_B],
        (N + TILE_SIZE - 1) // TILE_SIZE,
        (K + TILE_SIZE - 1) // TILE_SIZE,
        1,
    )


def run_layernorm_backward(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
    grad_gamma: GPUBuffer1D,
    grad_beta: GPUBuffer1D,
    accumulate: bool = False,
) -> None:
    """
    Backward pass for layer normalization (mutation)

    Uses two-stage reduction to avoid race conditions:
    - Stage 1: Each workgroup computes partial gamma/beta gradients
    - Stage 2: Reduction kernel sums partial gradients into final result

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input from forward pass (n_elements, size)
        gamma: Scale parameters from forward pass (size,)
        grad_output: Gradient of loss w.r.t. output (n_elements, size)
        grad_input: Output gradient w.r.t. input (MUTATED)
        grad_gamma: Output gradient w.r.t. gamma (MUTATED)
        grad_beta: Output gradient w.r.t. beta (MUTATED)
        accumulate: If False, zeros grad_gamma/grad_beta before accumulation.
                   If True, accumulates into existing values.

    Example:
        # Gradient accumulation (multiple mini-batches)
        run_layernorm_backward(cache, x1, gamma, grad_out1, grad_in1, grad_g, grad_b, accumulate=False)
        run_layernorm_backward(cache, x2, gamma, grad_out2, grad_in2, grad_g, grad_b, accumulate=True)
    """
    from gpu_buffer import pool_release_buffer, pool_take_buffer_2d

    n_elements, size = input_buf.shape
    config = pipeline_cache.device.config

    if n_elements == 0 or size == 0:
        raise ValueError(f"Invalid input_buf shape: {(n_elements, size)}")

    if gamma.shape != (size,):
        raise ValueError(
            f"gamma shape {gamma.shape} doesn't match input size {(size,)}"
        )

    if grad_output.shape != (n_elements, size):
        raise ValueError(
            f"grad_output shape {grad_output.shape} doesn't match input_buf shape {(n_elements, size)}"
        )

    if grad_input.shape != (n_elements, size):
        raise ValueError(
            f"grad_input shape {grad_input.shape} doesn't match input_buf shape {(n_elements, size)}"
        )

    if grad_gamma.shape != (size,):
        raise ValueError(
            f"grad_gamma shape {grad_gamma.shape} doesn't match expected {(size,)}"
        )

    if grad_beta.shape != (size,):
        raise ValueError(
            f"grad_beta shape {grad_beta.shape} doesn't match expected {(size,)}"
        )

    # Allocate temporary buffers for partial gradients
    buffer_pool = (
        pipeline_cache.device.buffer_pool
        if hasattr(pipeline_cache.device, "buffer_pool")
        else None
    )

    if buffer_pool is not None:
        partial_grad_gamma = pool_take_buffer_2d(buffer_pool, n_elements, size)
        partial_grad_beta = pool_take_buffer_2d(buffer_pool, n_elements, size)
    else:
        # Fallback: create temporary buffers directly
        from gpu_buffer import create_gpu_buffer_2d

        partial_grad_gamma = create_gpu_buffer_2d(
            pipeline_cache.device,
            n_elements,
            size,
            np.zeros((n_elements, size), dtype=np.float32),
        )
        partial_grad_beta = create_gpu_buffer_2d(
            pipeline_cache.device,
            n_elements,
            size,
            np.zeros((n_elements, size), dtype=np.float32),
        )

    # Stage 1: Compute partial gradients
    params = np.array([size, n_elements], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        get_layernorm_backward_kernel_from_config(config),
        params,
        [
            input_buf,
            gamma,
            grad_output,
            grad_input,
            partial_grad_gamma,
            partial_grad_beta,
        ],
        n_elements,
    )

    # Stage 2: Reduce partial gradients
    if accumulate:
        # Use atomic accumulation kernel (still needed for accumulation mode)
        reduction_kernel = get_layernorm_backward_reduce_accumulate_kernel_from_config(
            config
        )
    else:
        clear_buffer(grad_gamma)
        clear_buffer(grad_beta)
        reduction_kernel = get_layernorm_backward_reduce_kernel_from_config(config)

    params_reduce = np.array([size, n_elements], dtype=np.uint32)
    dispatch_simple_compute(
        pipeline_cache,
        reduction_kernel,
        params_reduce,
        [partial_grad_gamma, partial_grad_beta, grad_gamma, grad_beta],
        (size + 255) // 256,
    )

    # Release temporary buffers
    if buffer_pool is not None:
        pool_release_buffer(buffer_pool, partial_grad_gamma)
        pool_release_buffer(buffer_pool, partial_grad_beta)


def run_gelu_backward(
    pipeline_cache: PipelineCache,
    input_buf: GPUBuffer2D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
) -> None:
    """Backward pass for GELU activation (mutation).

    This function MUTATES grad_input by writing the computed gradients.
    Returns None to signal mutation.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        input_buf: Input from forward pass
        grad_output: Gradient of loss w.r.t. output
        grad_input: Output gradient w.r.t. input (MUTATED)

    Raises:
        ValueError: If shapes are incompatible
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
    config = pipeline_cache.device.config
    params = np.array([total_size], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        get_gelu_backward_kernel_from_config(config),
        params,
        [input_buf, grad_output, grad_input],
        (total_size + 255) // 256,
    )


def run_bias_backward(
    pipeline_cache: PipelineCache,
    grad_output: GPUBuffer2D,
    grad_bias: GPUBuffer1D,
    accumulate: bool = False,
) -> None:
    """
    Backward pass for bias - sum gradients over batch (mutation)

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        grad_output: Gradient of loss w.r.t. output (n_elements, dim)
        grad_bias: Output gradient w.r.t. bias (dim,) - MUTATED
        accumulate: If False, zeros grad_bias before accumulation.
                   If True, accumulates into existing grad_bias values.

    Example:
        # Gradient accumulation (multiple mini-batches)
        run_bias_backward(cache, grad_out1, grad_bias, accumulate=False)  # First batch
        run_bias_backward(cache, grad_out2, grad_bias, accumulate=True)   # Accumulate
        run_bias_backward(cache, grad_out3, grad_bias, accumulate=True)   # Accumulate
    """
    n_elements, dim = grad_output.shape

    if n_elements == 0 or dim == 0:
        raise ValueError(f"Invalid grad_output shape: {(n_elements, dim)}")

    if grad_bias.shape != (dim,):
        raise ValueError(
            f"grad_bias shape {grad_bias.shape} doesn't match expected {(dim,)}"
        )

    # Zero accumulation buffer only if not accumulating
    if not accumulate:
        clear_buffer(grad_bias)

    total_size = n_elements * dim
    config = pipeline_cache.device.config
    params = np.array([total_size, dim], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        get_bias_backward_kernel_from_config(config),
        params,
        [grad_output, grad_bias],
        (dim + 255) // 256,
    )


def run_attention_backward(
    pipeline_cache: PipelineCache,
    grad_output: GPUBuffer2D,
    Q: GPUBuffer2D,
    K: GPUBuffer2D,
    V: GPUBuffer2D,
    O: GPUBuffer2D,
    grad_Q: GPUBuffer2D,
    grad_K: GPUBuffer2D,
    grad_V: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
) -> None:
    """
    Attention backward pass - ATOMIC-FREE version (mutation)

    Each workgroup handles ONE query position completely, avoiding race conditions.
    """
    if batch_size <= 0 or seq_len <= 0 or n_heads <= 0 or head_dim <= 0:
        raise ValueError(
            f"Invalid dimensions: batch_size={batch_size}, seq_len={seq_len}, "
            f"n_heads={n_heads}, head_dim={head_dim}"
        )

    embedding_dim = n_heads * head_dim
    total_tokens = batch_size * seq_len

    validate_buffer_shape_2d(grad_output, (total_tokens, embedding_dim), "grad_output")
    validate_buffer_shape_2d(Q, (total_tokens, embedding_dim), "Q")
    validate_buffer_shape_2d(K, (total_tokens, embedding_dim), "K")
    validate_buffer_shape_2d(V, (total_tokens, embedding_dim), "V")
    validate_buffer_shape_2d(O, (total_tokens, embedding_dim), "O")
    validate_buffer_shape_2d(grad_Q, (total_tokens, embedding_dim), "grad_Q")
    validate_buffer_shape_2d(grad_K, (total_tokens, embedding_dim), "grad_K")
    validate_buffer_shape_2d(grad_V, (total_tokens, embedding_dim), "grad_V")

    config = pipeline_cache.device.config
    params = np.array([batch_size, seq_len, n_heads, head_dim], dtype=np.uint32)

    dispatch_simple_compute(
        pipeline_cache,
        get_attention_backward_kernel_from_config(config),
        params,
        [grad_output, Q, K, V, O, grad_Q, grad_K, grad_V],
        seq_len,
        n_heads,
        batch_size,
    )


def run_flash_attention_backward(
    pipeline_cache: PipelineCache,
    Q: GPUBuffer2D,
    K: GPUBuffer2D,
    V: GPUBuffer2D,
    O: GPUBuffer2D,
    L: GPUBuffer1D,
    M: GPUBuffer1D,
    grad_O: GPUBuffer2D,
    grad_Q: GPUBuffer2D,
    grad_K: GPUBuffer2D,
    grad_V: GPUBuffer2D,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
) -> None:
    """
    FlashAttention backward pass - ATOMIC-FREE version (mutation)

    Recomputes attention weights from saved statistics (L, M) to avoid
    materializing full attention matrix.
    """
    config = pipeline_cache.device.config

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
        get_flash_attention_backward_kernel_from_config(config),
        params,
        [Q, K, V, O, grad_O, L, M, grad_Q, grad_K, grad_V],
        num_q_blocks,
        n_heads,
        batch_size,
    )


# ============================================================================
# OPTIMIZER OPERATIONS
# ============================================================================


def run_adamw_update(
    pipeline_cache: PipelineCache,
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
    """
    Execute AdamW optimizer update.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        gradients: Gradient buffer [any_shape]
        weights: Weight buffer [any_shape]
        m: First moment buffer [any_shape]
        v: Second moment buffer [any_shape]
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        weight_decay: Weight decay coefficient (L2 penalty)
        eps: Small constant for numerical stability
        step: Current training step (for bias correction)

    Note:
        All buffers must have the same shape.
        Uses decoupled weight decay (AdamW variant).
    """
    from gpu_device import BindGroupEntry, get_or_create_pipeline
    from gpu_kernels import create_adamw_kernel
    from gpu_operations import (
        _create_bind_group,
        _create_uniform_buffer,
        _dispatch_compute,
    )

    total_size = weights.size

    # Create uniform buffer with all parameters including size
    opt_params = np.array(
        [lr, beta1, beta2, weight_decay, eps, float(step), total_size], dtype=np.float32
    )
    opt_params_buffer = _create_uniform_buffer(pipeline_cache, opt_params)

    # Generate kernel code
    kernel_code = create_adamw_kernel(workgroup_size=256)

    pipeline = get_or_create_pipeline(pipeline_cache, kernel_code)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, opt_params_buffer, 0, opt_params.nbytes),
            BindGroupEntry(1, gradients.buffer, 0, gradients.size * 4),
            BindGroupEntry(2, weights.buffer, 0, weights.size * 4),
            BindGroupEntry(3, m.buffer, 0, m.size * 4),
            BindGroupEntry(4, v.buffer, 0, v.size * 4),
        ],
    )

    _dispatch_compute(pipeline_cache, pipeline, bind_group, (total_size + 255) // 256)


def run_adamw_update_1d(
    pipeline_cache: PipelineCache,
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
    """
    Execute AdamW optimizer update for 1D buffers (biases, layer norms, etc).

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        gradients: Gradient buffer [size]
        weights: Weight buffer [size]
        m: First moment buffer [size]
        v: Second moment buffer [size]
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        weight_decay: Weight decay coefficient
        eps: Small constant for numerical stability
        step: Current training step
    """
    from gpu_device import BindGroupEntry, get_or_create_pipeline
    from gpu_kernels import create_adamw_kernel
    from gpu_operations import (
        _create_bind_group,
        _create_uniform_buffer,
        _dispatch_compute,
    )

    total_size = weights.size

    opt_params = np.array(
        [lr, beta1, beta2, weight_decay, eps, float(step), total_size], dtype=np.float32
    )
    opt_params_buffer = _create_uniform_buffer(pipeline_cache, opt_params)

    kernel_code = create_adamw_kernel(workgroup_size=256)

    pipeline = get_or_create_pipeline(pipeline_cache, kernel_code)

    bind_group = _create_bind_group(
        pipeline_cache,
        pipeline,
        [
            BindGroupEntry(0, opt_params_buffer, 0, opt_params.nbytes),
            BindGroupEntry(1, gradients.buffer, 0, gradients.size * 4),
            BindGroupEntry(2, weights.buffer, 0, weights.size * 4),
            BindGroupEntry(3, m.buffer, 0, m.size * 4),
            BindGroupEntry(4, v.buffer, 0, v.size * 4),
        ],
    )

    _dispatch_compute(pipeline_cache, pipeline, bind_group, (total_size + 255) // 256)


def run_gradient_clip(
    pipeline_cache: PipelineCache,
    gradients: GPUBuffer2D,
    max_norm: float,
    total_norm: float,
) -> None:
    """
    Clip gradients by global norm to prevent exploding gradients.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        gradients: Gradient buffer to clip in-place [any_shape]
        max_norm: Maximum allowed gradient norm
        total_norm: Pre-computed global norm of all gradients

    Note:
        Requires pre-computing total_norm (L2 norm of all gradients).
        If total_norm > max_norm, scales all gradients by max_norm/total_norm.
    """
    from gpu_kernels import create_gradient_clip_kernel

    size = gradients.shape[0] * gradients.shape[1]

    # Create uniform buffer with parameters
    params_data = bytearray(12)  # u32 + 2 f32
    params_data[0:4] = size.to_bytes(4, "little")
    params_data[4:8] = int.from_bytes(
        bytearray(max_norm.to_bytes(4, "little", signed=False)), "little"
    )
    params_data[8:12] = int.from_bytes(
        bytearray(total_norm.to_bytes(4, "little", signed=False)), "little"
    )

    # Generate kernel code
    kernel_code = create_gradient_clip_kernel(workgroup_size=256)

    # Dispatch computation
    workgroup_count = (size + 255) // 256
    dispatch_simple_compute(
        device=pipeline_cache.device,
        pipeline_cache=pipeline_cache,
        kernel_name="gradient_clip",
        kernel_code=kernel_code,
        buffers=[gradients.buffer],
        uniform_data=params_data,
        workgroup_count=(workgroup_count, 1, 1),
    )


def run_buffer_fill(
    pipeline_cache: PipelineCache,
    buffer: GPUBuffer2D,
    value: float,
) -> None:
    """
    Fill buffer with a constant value.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        buffer: Buffer to fill [any_shape]
        value: Value to fill buffer with

    Note:
        Useful for initializing buffers to zeros or specific values.
    """
    from gpu_kernels import create_buffer_fill_kernel

    size = buffer.shape[0] * buffer.shape[1]

    # Create uniform buffer with parameters
    params_data = bytearray(8)  # u32 + f32
    params_data[0:4] = size.to_bytes(4, "little")
    params_data[4:8] = int.from_bytes(
        bytearray(value.to_bytes(4, "little", signed=False)), "little"
    )

    # Generate kernel code
    kernel_code = create_buffer_fill_kernel(workgroup_size=256)

    # Dispatch computation
    workgroup_count = (size + 255) // 256
    dispatch_simple_compute(
        device=pipeline_cache.device,
        pipeline_cache=pipeline_cache,
        kernel_name="buffer_fill",
        kernel_code=kernel_code,
        buffers=[buffer.buffer],
        uniform_data=params_data,
        workgroup_count=(workgroup_count, 1, 1),
    )


def run_reduce_sum(
    pipeline_cache: PipelineCache,
    input_buffer: GPUBuffer2D,
    output_buffer: GPUBuffer1D,
) -> None:
    """
    Compute sum of all elements in buffer using parallel reduction.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        input_buffer: Input buffer [any_shape]
        output_buffer: Output buffer for partial sums [num_workgroups]

    Note:
        Returns partial sums per workgroup. For full reduction,
        may need second pass or CPU-side final sum.
    """
    from gpu_kernels import create_reduce_sum_kernel

    size = input_buffer.shape[0] * input_buffer.shape[1]
    workgroup_size = 256
    workgroup_count = (size + workgroup_size - 1) // workgroup_size

    # Create uniform buffer with parameters
    params_data = bytearray(4)  # u32
    params_data[0:4] = size.to_bytes(4, "little")

    # Generate kernel code
    kernel_code = create_reduce_sum_kernel(workgroup_size=workgroup_size)

    # Dispatch computation
    dispatch_simple_compute(
        device=pipeline_cache.device,
        pipeline_cache=pipeline_cache,
        kernel_name="reduce_sum",
        kernel_code=kernel_code,
        buffers=[input_buffer.buffer, output_buffer.buffer],
        uniform_data=params_data,
        workgroup_count=(workgroup_count, 1, 1),
    )


# ===============================


"""Command batching"""

from gpu_kernels import (
    BIAS_ADD_KERNEL,
    BIAS_BACKWARD_KERNEL,
    EMBEDDING_KERNEL,
    GELU_BACKWARD_KERNEL,
    GELU_KERNEL,
    LAYERNORM_BACKWARD_KERNEL,
    LAYERNORM_KERNEL,
    MATMUL_BACKWARD_A_KERNEL,
    MATMUL_BACKWARD_B_KERNEL,
    MULTIHEAD_ATTENTION_KERNEL,
    RESIDUAL_ADD_KERNEL,
    TILED_MATMUL_KERNEL,
)
from gpu_operations import (
    _add_compute_to_batch_internal,
    _validate_buffer_shapes,
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


def batch_add_softmax(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    logits: GPUBuffer2D,
    probs: GPUBuffer2D,
) -> None:
    """
    Add softmax operation to command batch.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        batch_state: Current batch state with encoder
        logits: Input logits [batch_size, vocab_size]
        probs: Output probabilities [batch_size, vocab_size]

    Note:
        Uses numerically stable softmax with max subtraction.
        For generation/sampling in batched forward passes.
    """
    from gpu_kernels import create_softmax_kernel
    from gpu_operations import (
        _add_compute_to_batch_internal,
        _create_and_retain_uniform_buffer_internal,
        _create_bind_group_for_batch_internal,
        validate_buffer_shape_2d,
    )

    # Validate shapes
    validate_buffer_shape_2d(logits)
    validate_buffer_shape_2d(probs)

    if logits.shape != probs.shape:
        raise ValueError(
            f"Softmax shape mismatch: logits {logits.shape} != probs {probs.shape}"
        )

    batch_size, vocab_size = logits.shape

    # Create uniform buffer with parameters
    params_data = bytearray(8)  # 2 u32s
    params_data[0:4] = batch_size.to_bytes(4, "little")
    params_data[4:8] = vocab_size.to_bytes(4, "little")

    # Generate kernel code
    kernel_code = create_softmax_kernel(workgroup_size=256)

    uniform_buffer = _create_and_retain_uniform_buffer_internal(
        batch_state=batch_state,
        data=params_data,
    )

    # Create bind group
    bind_group = _create_bind_group_for_batch_internal(
        pipeline_cache=pipeline_cache,
        batch_state=batch_state,
        kernel_name="softmax",
        kernel_code=kernel_code,
        buffers=[logits.buffer, probs.buffer],
        uniform_buffer=uniform_buffer,
    )

    # Add compute pass to batch
    _add_compute_to_batch_internal(
        pipeline_cache=pipeline_cache,
        batch_state=batch_state,
        kernel_name="softmax",
        kernel_code=kernel_code,
        bind_group=bind_group,
        workgroup_count=(batch_size, 1, 1),
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
    pipelinecache: PipelineCache,
    batch_state: BatchState,
    input_buf: GPUBuffer2D,
    gamma: GPUBuffer1D,
    grad_output: GPUBuffer2D,
    grad_input: GPUBuffer2D,
    grad_gamma: GPUBuffer1D,
    grad_beta: GPUBuffer1D,
    accumulate: bool = False,
) -> None:
    """
    Add layernorm backward to batch (mutation)

    This function MUTATES batch_state by adding operations.
    Also MUTATES grad_gamma and grad_beta (conditionally zeros based on accumulate flag).
    Returns None to signal mutation.

    NOTE: For batched operations with accumulate=True, temporary buffers are NOT supported.
    This is a limitation of the batch API - partial gradients must be computed and reduced
    in separate submissions. For gradient accumulation, prefer the non-batch API.

    Args:
        pipelinecache: Pipeline cache for kernel compilation
        batch_state: Batch state - MUTATED
        input_buf: Input from forward pass (n_elements, size)
        gamma: Scale parameters from forward pass (size,)
        grad_output: Gradient of loss w.r.t. output (n_elements, size)
        grad_input: Output gradient w.r.t. input (n_elements, size)
        grad_gamma: Output gradient w.r.t. gamma (size,)
        grad_beta: Output gradient w.r.t. beta (size,)
        accumulate: If False (default), zeros grad_gamma/grad_beta before operation.
                   If True, uses atomic accumulation (requires WGSL atomics support).

    Raises:
        ValueError: If buffer shapes don't match
        NotImplementedError: If accumulate=True (not supported in batch mode yet)
    """

    n_elements, size = input_buf.shape

    if n_elements == 0 or size == 0:
        raise ValueError(f"Invalid input shape: {(n_elements, size)}")

    validate_buffer_shape_1d(gamma, size, "gamma")
    validate_buffer_shape_2d(grad_output, (n_elements, size), "grad_output")
    validate_buffer_shape_2d(grad_input, (n_elements, size), "grad_input")
    validate_buffer_shape_1d(grad_gamma, size, "grad_gamma")
    validate_buffer_shape_1d(grad_beta, size, "grad_beta")

    if accumulate:
        # Batch mode doesn't support temporary buffer management required for accumulation
        # Users should use non-batch API for gradient accumulation
        raise NotImplementedError(
            "Gradient accumulation (accumulate=True) not supported in batch mode. "
            "Use run_layernorm_backward() instead for gradient accumulation."
        )

    # Zero accumulation buffers
    clear_buffer(grad_gamma)
    clear_buffer(grad_beta)

    params = np.array([size, n_elements], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipelinecache,
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
    pipelinecache: PipelineCache,
    batch_state: BatchState,
    grad_output: GPUBuffer2D,
    grad_bias: GPUBuffer1D,
    accumulate: bool = False,
) -> None:
    """
    Add bias backward to batch (mutation)

    This function MUTATES batch_state by adding an operation.
    Returns None to signal mutation.

    Args:
        pipelinecache: Pipeline cache for kernel compilation
        batch_state: Batch state - MUTATED
        grad_output: Gradient of loss w.r.t. output (n_elements, dim)
        grad_bias: Output gradient w.r.t. bias (dim,)
        accumulate: If False (default), zeros grad_bias before operation.
                   If True, accumulates into existing grad_bias values.

    Raises:
        ValueError: If buffer shapes don't match
    """
    n_elements, dim = grad_output.shape

    if n_elements == 0 or dim == 0:
        raise ValueError(f"Invalid grad_output shape: {(n_elements, dim)}")

    validate_buffer_shape_1d(grad_bias, dim, "grad_bias")

    # Zero accumulation buffer only if not accumulating
    if not accumulate:
        clear_buffer(grad_bias)

    total_size = n_elements * dim
    params = np.array([total_size, dim], dtype=np.uint32)

    _add_compute_to_batch_internal(
        pipelinecache,
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
    """
    Add AdamW optimizer update to command batch.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        batch_state: Current batch state with encoder
        gradients: Gradient buffer
        weights: Weight buffer
        m: First moment buffer
        v: Second moment buffer
        lr: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        weight_decay: Weight decay coefficient
        eps: Numerical stability constant
        step: Current training step
    """
    import struct

    from gpu_kernels import create_adamw_kernel
    from gpu_operations import (
        _add_compute_to_batch_internal,
        _create_and_retain_uniform_buffer_internal,
        _create_bind_group_for_batch_internal,
    )

    total_size = weights.size

    # Create uniform buffer (7 floats: lr, beta1, beta2, weight_decay, eps, step, size)
    params_data = bytearray(28)
    params_data[0:4] = struct.pack("<f", lr)
    params_data[4:8] = struct.pack("<f", beta1)
    params_data[8:12] = struct.pack("<f", beta2)
    params_data[12:16] = struct.pack("<f", weight_decay)
    params_data[16:20] = struct.pack("<f", eps)
    params_data[20:24] = struct.pack("<f", float(step))
    params_data[24:28] = struct.pack("<f", float(total_size))

    kernel_code = create_adamw_kernel(workgroup_size=256)

    uniform_buffer = _create_and_retain_uniform_buffer_internal(
        batch_state=batch_state,
        data=params_data,
    )

    bind_group = _create_bind_group_for_batch_internal(
        pipeline_cache=pipeline_cache,
        batch_state=batch_state,
        kernel_name="adamw",
        kernel_code=kernel_code,
        buffers=[
            gradients.buffer,
            weights.buffer,
            m.buffer,
            v.buffer,
        ],
        uniform_buffer=uniform_buffer,
    )

    workgroup_count = (total_size + 255) // 256
    _add_compute_to_batch_internal(
        pipeline_cache=pipeline_cache,
        batch_state=batch_state,
        kernel_name="adamw",
        kernel_code=kernel_code,
        bind_group=bind_group,
        workgroup_count=(workgroup_count, 1, 1),
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
    """
    Add AdamW optimizer update for 1D buffers to command batch.

    Args:
        pipeline_cache: Pipeline cache
        batch_state: Current batch state
        gradients: Gradient buffer (1D)
        weights: Weight buffer (1D)
        m: First moment buffer (1D)
        v: Second moment buffer (1D)
        lr: Learning rate
        beta1: First moment decay
        beta2: Second moment decay
        weight_decay: Weight decay coefficient
        eps: Stability constant
        step: Training step
    """
    import struct

    from gpu_kernels import create_adamw_kernel
    from gpu_operations import (
        _add_compute_to_batch_internal,
        _create_and_retain_uniform_buffer_internal,
        _create_bind_group_for_batch_internal,
    )

    total_size = weights.size

    params_data = bytearray(28)
    params_data[0:4] = struct.pack("<f", lr)
    params_data[4:8] = struct.pack("<f", beta1)
    params_data[8:12] = struct.pack("<f", beta2)
    params_data[12:16] = struct.pack("<f", weight_decay)
    params_data[16:20] = struct.pack("<f", eps)
    params_data[20:24] = struct.pack("<f", float(step))
    params_data[24:28] = struct.pack("<f", float(total_size))

    kernel_code = create_adamw_kernel(workgroup_size=256)

    uniform_buffer = _create_and_retain_uniform_buffer_internal(
        batch_state=batch_state,
        data=params_data,
    )

    bind_group = _create_bind_group_for_batch_internal(
        pipeline_cache=pipeline_cache,
        batch_state=batch_state,
        kernel_name="adamw_1d",
        kernel_code=kernel_code,
        buffers=[
            gradients.buffer,
            weights.buffer,
            m.buffer,
            v.buffer,
        ],
        uniform_buffer=uniform_buffer,
    )

    workgroup_count = (total_size + 255) // 256
    _add_compute_to_batch_internal(
        pipeline_cache=pipeline_cache,
        batch_state=batch_state,
        kernel_name="adamw_1d",
        kernel_code=kernel_code,
        bind_group=bind_group,
        workgroup_count=(workgroup_count, 1, 1),
    )


def batch_add_gradient_clip(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    gradients: GPUBuffer2D,
    max_norm: float,
    total_norm: float,
) -> None:
    """
    Add gradient clipping operation to command batch.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        batch_state: Current batch state with encoder
        gradients: Gradient buffer to clip in-place
        max_norm: Maximum allowed gradient norm
        total_norm: Pre-computed global norm of all gradients
    """
    from gpu_kernels import create_gradient_clip_kernel
    from gpu_operations import (
        _add_compute_to_batch_internal,
        _create_and_retain_uniform_buffer_internal,
        _create_bind_group_for_batch_internal,
    )

    size = gradients.shape[0] * gradients.shape[1]

    # Create uniform buffer with parameters
    import struct

    params_data = bytearray(12)
    params_data[0:4] = size.to_bytes(4, "little")
    params_data[4:8] = struct.pack("<f", max_norm)
    params_data[8:12] = struct.pack("<f", total_norm)

    kernel_code = create_gradient_clip_kernel(workgroup_size=256)

    uniform_buffer = _create_and_retain_uniform_buffer_internal(
        batch_state=batch_state,
        data=params_data,
    )

    bind_group = _create_bind_group_for_batch_internal(
        pipeline_cache=pipeline_cache,
        batch_state=batch_state,
        kernel_name="gradient_clip",
        kernel_code=kernel_code,
        buffers=[gradients.buffer],
        uniform_buffer=uniform_buffer,
    )

    workgroup_count = (size + 255) // 256
    _add_compute_to_batch_internal(
        pipeline_cache=pipeline_cache,
        batch_state=batch_state,
        kernel_name="gradient_clip",
        kernel_code=kernel_code,
        bind_group=bind_group,
        workgroup_count=(workgroup_count, 1, 1),
    )


def batch_add_buffer_fill(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    buffer: GPUBuffer2D,
    value: float,
) -> None:
    """
    Add buffer fill operation to command batch.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        batch_state: Current batch state with encoder
        buffer: Buffer to fill
        value: Value to fill with
    """
    from gpu_kernels import create_buffer_fill_kernel
    from gpu_operations import (
        _add_compute_to_batch_internal,
        _create_and_retain_uniform_buffer_internal,
        _create_bind_group_for_batch_internal,
    )

    size = buffer.shape[0] * buffer.shape[1]

    # Create uniform buffer with parameters
    import struct

    params_data = bytearray(8)
    params_data[0:4] = size.to_bytes(4, "little")
    params_data[4:8] = struct.pack("<f", value)

    kernel_code = create_buffer_fill_kernel(workgroup_size=256)

    uniform_buffer = _create_and_retain_uniform_buffer_internal(
        batch_state=batch_state,
        data=params_data,
    )

    bind_group = _create_bind_group_for_batch_internal(
        pipeline_cache=pipeline_cache,
        batch_state=batch_state,
        kernel_name="buffer_fill",
        kernel_code=kernel_code,
        buffers=[buffer.buffer],
        uniform_buffer=uniform_buffer,
    )

    workgroup_count = (size + 255) // 256
    _add_compute_to_batch_internal(
        pipeline_cache=pipeline_cache,
        batch_state=batch_state,
        kernel_name="buffer_fill",
        kernel_code=kernel_code,
        bind_group=bind_group,
        workgroup_count=(workgroup_count, 1, 1),
    )


def batch_add_reduce_sum(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buffer: GPUBuffer2D,
    output_buffer: GPUBuffer1D,
) -> None:
    """
    Add reduction sum operation to command batch.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        batch_state: Current batch state with encoder
        input_buffer: Input buffer to reduce
        output_buffer: Output buffer for partial sums
    """
    from gpu_kernels import create_reduce_sum_kernel
    from gpu_operations import (
        _add_compute_to_batch_internal,
        _create_and_retain_uniform_buffer_internal,
        _create_bind_group_for_batch_internal,
    )

    size = input_buffer.shape[0] * input_buffer.shape[1]
    workgroup_size = 256
    workgroup_count = (size + workgroup_size - 1) // workgroup_size

    params_data = bytearray(4)
    params_data[0:4] = size.to_bytes(4, "little")

    kernel_code = create_reduce_sum_kernel(workgroup_size=workgroup_size)

    uniform_buffer = _create_and_retain_uniform_buffer_internal(
        batch_state=batch_state,
        data=params_data,
    )

    bind_group = _create_bind_group_for_batch_internal(
        pipeline_cache=pipeline_cache,
        batch_state=batch_state,
        kernel_name="reduce_sum",
        kernel_code=kernel_code,
        buffers=[input_buffer.buffer, output_buffer.buffer],
        uniform_buffer=uniform_buffer,
    )

    _add_compute_to_batch_internal(
        pipeline_cache=pipeline_cache,
        batch_state=batch_state,
        kernel_name="reduce_sum",
        kernel_code=kernel_code,
        bind_group=bind_group,
        workgroup_count=(workgroup_count, 1, 1),
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
