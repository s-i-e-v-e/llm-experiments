from typing import List, Union

import numpy as np

from .gpu_buffer import gpu_buffer_1d_create, gpu_buffer_1d_read
from .gpu_kernels import (
    get_adamw_kernel,
    get_buffer_fill_kernel,
    get_gradient_clip_kernel,
    get_gradient_norm_kernel,
    get_gradient_norm_reduce_kernel,
    get_reduce_sum_kernel,
)
from .gpu_ops import add_compute_to_batch, create_command_batch, submit_batch
from .gpu_types import (
    BatchState,
    GPUBuffer1D,
    GPUBuffer2D,
    GPUConfig,
    GPUDevice,
    PipelineCache,
)

# ============================================================================
# OPTIMIZER OPERATIONS
# ============================================================================


def __adamw_update(
    device: GPUDevice,
    config: GPUConfig,
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    gradients: Union[GPUBuffer1D, GPUBuffer2D],
    weights: Union[GPUBuffer1D, GPUBuffer2D],
    m: Union[GPUBuffer1D, GPUBuffer2D],
    v: Union[GPUBuffer1D, GPUBuffer2D],
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    step: int,
) -> None:
    """
    Execute AdamW optimizer update for 2D buffers.

    Uses decoupled weight decay (AdamW variant).

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        batch_state: Current batch state with encoder
        gradients: Gradient buffer
        weights: Weight buffer
        m: First moment buffer
        v: Second moment buffer
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        weight_decay: Weight decay coefficient (L2 penalty)
        eps: Small constant for numerical stability
        step: Current training step (for bias correction)
    """

    total_size = weights.size
    params = np.array(
        [lr, beta1, beta2, weight_decay, eps, float(step), float(total_size)],
        dtype=np.float32,
    )

    add_compute_to_batch(
        device,
        config,
        pipeline_cache,
        batch_state,
        get_adamw_kernel(config),
        params,
        [gradients, weights, m, v],
        (total_size + 255) // 256,
    )


def adamw_update_2d(
    device: GPUDevice,
    config: GPUConfig,
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
    __adamw_update(
        device,
        config,
        pipeline_cache,
        batch_state,
        gradients,
        weights,
        m,
        v,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        step,
    )


def adamw_update_1d(
    device: GPUDevice,
    config: GPUConfig,
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
    __adamw_update(
        device,
        config,
        pipeline_cache,
        batch_state,
        gradients,
        weights,
        m,
        v,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        step,
    )


def gradient_clip(
    device: GPUDevice,
    config: GPUConfig,
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    gradients: GPUBuffer2D,
    max_norm: float,
    total_norm: float,
) -> None:
    """
    Clip gradients by global norm to prevent exploding gradients.

    Note:
        Requires pre-computing total_norm (L2 norm of all gradients).
        If total_norm > max_norm, scales all gradients by max_norm/total_norm.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        batch_state: Current batch state with encoder
        gradients: Gradient buffer to clip in-place
        max_norm: Maximum allowed gradient norm
        total_norm: Pre-computed global norm of all gradients
    """

    total_size = gradients.shape[0] * gradients.shape[1]

    params = np.array(
        [float(total_size), max_norm, total_norm],
        dtype=np.float32,
    )

    add_compute_to_batch(
        device,
        config,
        pipeline_cache,
        batch_state,
        get_gradient_clip_kernel(config),
        params,
        [gradients],
        (total_size + 255) // 256,
    )


def buffer_fill(
    device: GPUDevice,
    config: GPUConfig,
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    buffer: GPUBuffer2D,
    value: float,
) -> None:
    """
    Fill buffer with a constant value.

    Useful for initializing buffers to zeros or specific values.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        batch_state: Current batch state with encoder
        buffer: Buffer to fill
        value: Value to fill with
    """

    total_size = buffer.shape[0] * buffer.shape[1]

    params = np.array(
        [float(total_size), value],
        dtype=np.float32,
    )

    add_compute_to_batch(
        device,
        config,
        pipeline_cache,
        batch_state,
        get_buffer_fill_kernel(config),
        params,
        [buffer],
        (total_size + 255) // 256,
    )


def reduce_sum(
    device: GPUDevice,
    config: GPUConfig,
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    input_buffer: GPUBuffer2D,
    output_buffer: GPUBuffer1D,
) -> None:
    """
    Compute sum of all elements in buffer using parallel reduction.

    Note:
        Computes partial sums per workgroup. For full reduction,
        may need second pass or CPU-side final sum.

    Args:
        pipeline_cache: Pipeline cache for compute pipelines
        batch_state: Current batch state with encoder
        input_buffer: Input buffer to reduce
        output_buffer: Output buffer for partial sums [num_workgroups]
    """

    total_size = input_buffer.shape[0] * input_buffer.shape[1]

    params = np.array(
        [float(total_size)],
        dtype=np.float32,
    )

    add_compute_to_batch(
        device,
        config,
        pipeline_cache,
        batch_state,
        get_reduce_sum_kernel(config),
        params,
        [input_buffer, output_buffer],
        (total_size + 255) // 256,
    )


def __compute_gradient_norm(
    device: GPUDevice,
    config: GPUConfig,
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    gradients: List[GPUBuffer2D],
    reduction_workspace: GPUBuffer1D,
) -> GPUBuffer1D:
    """Compute global L2 norm using pre-allocated workspace across all gradient buffers.

    Two-stage process:
    1. Each buffer computes partial norms per workgroup
    2. Reduce all partials to single global norm

    Args:
        device: GPU device
        config: GPU configuration
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state
        gradients: List of all gradient buffers (GPUBuffer2D)

    Returns:
        Buffer containing single scalar with global gradient norm
    """

    # Track offset into reduction workspace
    current_offset = 0
    total_partials = 0

    # Phase 1: Per-gradient partial reductions
    for grad_buf in gradients:
        size = grad_buf.shape[0] * grad_buf.shape[1]
        num_workgroups = (size + 255) // 256

        # Write partial sums to offset in pre-allocated workspace
        params = np.array([size, current_offset], dtype=np.uint32)

        add_compute_to_batch(
            device,
            config,
            pipeline_cache,
            batch_state,
            get_gradient_norm_kernel(config),
            params,
            [grad_buf, reduction_workspace],
            num_workgroups,
        )

        current_offset += num_workgroups
        total_partials += num_workgroups

    # Phase 2: Final reduction over all partials
    global_norm_buf = gpu_buffer_1d_create(device, 1)
    reduce_params = np.array([total_partials], dtype=np.uint32)

    add_compute_to_batch(
        device,
        config,
        pipeline_cache,
        batch_state,
        get_gradient_norm_reduce_kernel(config),
        reduce_params,
        [reduction_workspace, global_norm_buf],
        1,
    )

    return global_norm_buf


def gradient_clip_with_norm(
    device: GPUDevice,
    config: GPUConfig,
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    gradients: List[GPUBuffer2D],
    max_norm: float,
    reduction_workspace: GPUBuffer1D,
) -> None:
    """Clip gradients by computing and using global norm.

    Args:
        device: GPU device
        config: GPU configuration
        pipeline_cache: Pipeline cache for kernel compilation
        batch_state: Batch state
        buffer_pool: Buffer pool for temporary allocations
        gradients: List of all gradient buffers to clip
        max_norm: Maximum allowed gradient norm
    """
    total_norm_buf = __compute_gradient_norm(
        device, config, pipeline_cache, batch_state, gradients, reduction_workspace
    )

    submit_batch(device, batch_state)

    norm_array = np.array([], dtype=np.float32)
    gpu_buffer_1d_read(device, total_norm_buf, norm_array)
    total_norm = float(norm_array[0])

    batch_state = create_command_batch(device, config)

    for grad_buf in gradients:
        size = grad_buf.shape[0] * grad_buf.shape[1]
        params = np.array([size, max_norm, total_norm], dtype=np.float32)

        add_compute_to_batch(
            device,
            config,
            pipeline_cache,
            batch_state,
            get_gradient_clip_kernel(config),
            params,
            [grad_buf],
            (size + 255) // 256,
        )
    submit_batch(device, batch_state)
