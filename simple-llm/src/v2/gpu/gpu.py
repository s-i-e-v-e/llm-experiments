"""
GPU (WGPU) Backend for LLM Training and Inference
"""

# Core types
# Buffer operations
from .gpu_buffer import (
    clear_buffer,
    create_buffer_pool,
    create_gpu_buffer_1d,
    create_gpu_buffer_2d,
    create_gpu_buffer_3d,
    gpu_to_numpy,
)

# Device management
from .gpu_device import (
    create_device,
    create_perf_monitor,
    create_pipeline_cache,
    get_perf_stats,
    query_device_limits,
    record_kernel_time,
)

# Operations (the main API for wgpu_backend.py)
from .gpu_operations import (
    # Optimizer
    adamw_step,
    attention,
    attention_backward,
    bias_add,
    bias_backward,
    # Infrastructure
    create_batch_state,
    cross_entropy_loss,
    embedding,
    gelu,
    gelu_backward,
    layernorm,
    layernorm_backward,
    layernorm_backward_reduce,
    # Forward ops
    matmul,
    # Backward ops
    matmul_backward_a,
    matmul_backward_b,
    residual_add,
    submit_batch,
)
from .gpu_types import (
    BatchState,
    Device,
    GPUBuffer1D,
    GPUBuffer2D,
    GPUBuffer3D,
    GPUConfig,
    GPULayerParams,
    GPUModelParams,
    GPUOptimizerState,
    PipelineCache,
    WorkspaceBuffers,
    WorkspaceManager,
)

# Workspace management
from .gpu_workspace import create_workspace_manager, get_workspace, release_workspace

__all__ = [
    # Types
    "Device",
    "GPUConfig",
    "GPUBuffer1D",
    "GPUBuffer2D",
    "GPUBuffer3D",
    "GPUModelParams",
    "GPULayerParams",
    "GPUOptimizerState",
    "BatchState",
    "PipelineCache",
    "WorkspaceBuffers",
    "WorkspaceManager",
    # Device
    "create_device",
    "query_device_limits",
    "create_pipeline_cache",
    # Buffers
    "create_gpu_buffer_1d",
    "create_gpu_buffer_2d",
    "create_gpu_buffer_3d",
    "gpu_to_numpy",
    "clear_buffer",
    "create_buffer_pool",
    # Workspace
    "create_workspace_manager",
    "get_workspace",
    "release_workspace",
    # Operations
    "create_batch_state",
    "submit_batch",
    "matmul",
    "embedding",
    "attention",
    "layernorm",
    "gelu",
    "bias_add",
    "residual_add",
    "cross_entropy_loss",
    "matmul_backward_a",
    "matmul_backward_b",
    "layernorm_backward",
    "layernorm_backward_reduce",
    "gelu_backward",
    "bias_backward",
    "attention_backward",
    # Optimizer
    "adamw_step",
    # Profiling
    "create_perf_monitor",
    "record_kernel_time",
    "get_perf_stats",
]
