from typing import Union

import numpy as np
from gpu_kernels import (
    get_adamw_kernel,
    get_buffer_fill_kernel,
    get_gradient_clip_kernel,
    get_reduce_sum_kernel,
)
from gpu_types import (
    BatchState,
    GPUBuffer1D,
    GPUBuffer2D,
    PipelineCache,
)

from .gpu_ops import (
    _add_compute_to_batch_internal,
)

# ============================================================================
# OPTIMIZER OPERATIONS
# ============================================================================


def INTERNAL__adamw_update(
    pipeline_cache: PipelineCache,
    batch_state: BatchState,
    gradients: Union[GPUBuffer1D | GPUBuffer2D],
    weights: Union[GPUBuffer1D | GPUBuffer2D],
    m: Union[GPUBuffer1D | GPUBuffer2D],
    v: Union[GPUBuffer1D | GPUBuffer2D],
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
    config = pipeline_cache.device.config

    total_size = weights.size
    params = np.array(
        [lr, beta1, beta2, weight_decay, eps, float(step), float(total_size)],
        dtype=np.float32,
    )

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_adamw_kernel(config),
        params,
        [
            gradients.buffer,
            weights.buffer,
            m.buffer,
            v.buffer,
        ],
        (total_size + 255) // 256,
    )


def adamw_update_2d(
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
    INTERNAL__adamw_update(
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
    INTERNAL__adamw_update(
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
    config = pipeline_cache.device.config

    total_size = gradients.shape[0] * gradients.shape[1]

    params = np.array(
        [float(total_size), max_norm, total_norm],
        dtype=np.float32,
    )

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_gradient_clip_kernel(config),
        params,
        [gradients.buffer],
        (total_size + 255) // 256,
    )


def buffer_fill(
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
    config = pipeline_cache.device.config

    total_size = buffer.shape[0] * buffer.shape[1]

    params = np.array(
        [float(total_size), value],
        dtype=np.float32,
    )

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_buffer_fill_kernel(config),
        params,
        [buffer.buffer],
        (total_size + 255) // 256,
    )


def reduce_sum(
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
        batch_state: Current batch state with encoder (MUTATED)
        input_buffer: Input buffer to reduce
        output_buffer: Output buffer for partial sums [num_workgroups] (MUTATED)
    """
    config = pipeline_cache.device.config

    total_size = input_buffer.shape[0] * input_buffer.shape[1]

    params = np.array(
        [float(total_size)],
        dtype=np.float32,
    )

    _add_compute_to_batch_internal(
        pipeline_cache,
        batch_state,
        get_reduce_sum_kernel(config),
        params,
        [input_buffer.buffer, output_buffer.buffer],
        (total_size + 255) // 256,
    )
