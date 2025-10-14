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
from .gpu_kv_cache import (
    attention_with_kv_cache,
    kv_cache_create,
    kv_cache_reset,
    kv_cache_update,
)
from .gpu_ops import batch_add, batch_begin, batch_commit
from .gpu_pass_backward import (
    bias_backward,
    dropout_backward,
    embedding_backward,
    flash_attention_backward,
    gelu_backward,
    layernorm_backward,
    matmul_backward_a,
    matmul_backward_b,
)
from .gpu_pass_forward import (
    bias_add,
    cross_entropy_loss,
    dropout,
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
    gradient_clip_1d,
    gradient_clip_2d,
    gradient_clip_with_norm,
    reduce_sum,
    transpose,
)
from .gpu_types import (
    GPUBuffer1D,
    GPUBuffer2D,
    GPUContext,
    GPULayerGradients,
    GPULayerParams,
    GPUModelGradients,
    GPUModelParams,
    GPUOptimizerState,
    KVCacheConfig,
)

__all__ = [
    # Types
    "GPUModelGradients",
    "GPUModelParams",
    "GPULayerGradients",
    "GPULayerParams",
    "GPUBuffer1D",
    "GPUBuffer2D",
    "GPUOptimizerState",
    "GPUContext",
    "KVCacheConfig",
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
    "gradient_clip_1d",
    "gradient_clip_2d",
    "buffer_fill",
    "reduce_sum",
    "gradient_clip_with_norm",
    "transpose",
    # Forward
    "matmul",
    "embedding",
    "layernorm",
    "gelu",
    "softmax",
    "bias_add",
    "residual_add",
    "extract_last_tokens",
    "cross_entropy_loss",
    "flash_attention",
    "dropout",
    # Backward
    "matmul_backward_a",
    "matmul_backward_b",
    "layernorm_backward",
    "gelu_backward",
    "bias_backward",
    "embedding_backward",
    "flash_attention_backward",
    "dropout_backward",
    # Ops
    "batch_begin",
    "batch_add",
    "batch_commit",
    # KV-cache
    "kv_cache_create",
    "kv_cache_reset",
    "kv_cache_update",
    "attention_with_kv_cache",
]
