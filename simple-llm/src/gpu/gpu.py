"""
GPU (WGPU) Backend for LLM Training and Inference
"""

from .gpu_buffer import (
    gpu_buffer_1d_create,
    gpu_buffer_1d_read,
    gpu_buffer_1d_write,
    gpu_buffer_2d_create,
    gpu_buffer_2d_read,
    gpu_buffer_2d_write,
)
from .gpu_device import (
    device_config_auto_detect,
    device_config_create,
    device_config_shared_memory_usage_estimate,
    device_config_validate,
    device_create,
    device_limits_query,
    perf_monitor_create,
    perf_monitor_kernel_time_record,
    perf_monitor_reset,
    perf_monitor_stats_get,
    perf_monitor_submission_record,
    pipeline_cache_create,
    pipeline_get_or_create,
    pipeline_tuned_create,
    select_optimal_tile_size,
)
from .gpu_ops import create_command_batch
from .gpu_pass_backward import (
    attention_backward,
    bias_backward,
    flash_attention_backward,
    gelu_backward,
    layernorm_backward,
    matmul_backward_a,
    matmul_backward_b,
)
from .gpu_pass_forward import (
    attention,
    bias_add,
    cross_entropy_loss,
    embedding,
    extract_last_tokens,
    flash_attention,
    gelu,
    layernorm,
    matmul,
    residual_add,
    softmax,
)
from .gpu_pass_optimizer import (
    adamw_update_1d,
    adamw_update_2d,
    buffer_fill,
    gradient_clip,
    gradient_clip_with_norm,
    reduce_sum,
)
from .gpu_types import (
    BatchState,
    GPUBuffer1D,
    GPUBuffer2D,
    GPUConfig,
    GPUDevice,
    GPULayerGradients,
    GPULayerParams,
    GPUModelGradients,
    GPUModelParams,
    GPUOptimizerState,
    PipelineCache,
)

__all__ = [
    # Types
    "GPUDevice",
    "GPUModelGradients",
    "GPUModelParams",
    "GPULayerGradients",
    "GPULayerParams",
    "GPUConfig",
    "GPUBuffer1D",
    "GPUBuffer2D",
    "GPUOptimizerState",
    "BatchState",
    "PipelineCache",
    # Device
    "device_create",
    "device_limits_query",
    "pipeline_cache_create",
    "pipeline_get_or_create",
    "pipeline_tuned_create",
    "select_optimal_tile_size",
    "device_config_create",
    "device_config_validate",
    "device_config_auto_detect",
    "device_config_shared_memory_usage_estimate",
    # Profiling
    "perf_monitor_reset",
    "perf_monitor_stats_get",
    "perf_monitor_submission_record",
    "perf_monitor_kernel_time_record",
    "perf_monitor_create",
    # Buffers
    "gpu_buffer_1d_create",
    "gpu_buffer_2d_create",
    "gpu_buffer_1d_write",
    "gpu_buffer_2d_write",
    "gpu_buffer_1d_read",
    "gpu_buffer_2d_read",
    # Optimizer
    "adamw_update_1d",
    "adamw_update_2d",
    "gradient_clip",
    "buffer_fill",
    "reduce_sum",
    "gradient_clip_with_norm",
    # Forward
    "matmul",
    "embedding",
    "attention",
    "layernorm",
    "gelu",
    "softmax",
    "bias_add",
    "residual_add",
    "extract_last_tokens",
    "cross_entropy_loss",
    "flash_attention",
    # Backward
    "matmul_backward_a",
    "matmul_backward_b",
    "layernorm_backward",
    "gelu_backward",
    "bias_backward",
    "attention_backward",
    "flash_attention_backward",
    # Ops
    "create_command_batch",
]
