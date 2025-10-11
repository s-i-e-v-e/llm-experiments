"""Optimizer operations"""

import numpy as np
from gpu_kernels_opt import ADAMW_OPTIMIZER_KERNEL
from gpu_ops import dispatch_simple_compute
from gpu_types import GPUBuffer1D, GPUBuffer2D, PipelineCache

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
    Apply AdamW optimizer update to weights.

    This function MUTATES weights, m (momentum), and v (variance).
    Uses decoupled weight decay as described in Loshchilov & Hutter 2019.

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        gradients: Gradient buffer (same shape as weights)
        weights: Parameter buffer (MUTATED)
        m: First moment (momentum) buffer (MUTATED)
        v: Second moment (variance) buffer (MUTATED)
        lr: Learning rate
        beta1: Exponential decay rate for first moment (typically 0.9)
        beta2: Exponential decay rate for second moment (typically 0.999)
        weight_decay: Weight decay coefficient (typically 0.01)
        eps: Small constant for numerical stability (typically 1e-8)
        step: Current training step (1-indexed, for bias correction)
    """
    assert gradients.shape == weights.shape, (
        f"Shape mismatch: {gradients.shape} != {weights.shape}"
    )
    assert weights.shape == m.shape == v.shape, "All buffers must have same shape"
    assert step >= 1, "Step must be >= 1"

    size = weights.size

    # Pack optimizer hyperparameters
    params = np.array(
        [lr, beta1, beta2, weight_decay, eps, float(step), size],
        dtype=np.float32,
    )

    dispatch_simple_compute(
        pipeline_cache,
        ADAMW_OPTIMIZER_KERNEL,
        params,
        [gradients, weights, m, v],
        (size + 255) // 256,
    )


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
    Apply AdamW optimizer update to 1D parameters (biases, layer norm params).

    This function MUTATES weights, m (momentum), and v (variance).

    Args:
        pipeline_cache: Pipeline cache for kernel compilation
        gradients: Gradient buffer (same shape as weights)
        weights: Parameter buffer (MUTATED)
        m: First moment (momentum) buffer (MUTATED)
        v: Second moment (variance) buffer (MUTATED)
        lr: Learning rate
        beta1: Exponential decay rate for first moment (typically 0.9)
        beta2: Exponential decay rate for second moment (typically 0.999)
        weight_decay: Weight decay coefficient (typically 0.01)
        eps: Small constant for numerical stability (typically 1e-8)
        step: Current training step (1-indexed, for bias correction)
    """
    assert gradients.size == weights.size, (
        f"Size mismatch: {gradients.size} != {weights.size}"
    )
    assert weights.size == m.size == v.size, "All buffers must have same size"
    assert step >= 1, "Step must be >= 1"

    size = weights.size

    # Pack optimizer hyperparameters
    params = np.array(
        [lr, beta1, beta2, weight_decay, eps, float(step), size],
        dtype=np.float32,
    )

    dispatch_simple_compute(
        pipeline_cache,
        ADAMW_OPTIMIZER_KERNEL,
        params,
        [gradients, weights, m, v],
        (size + 255) // 256,
    )
