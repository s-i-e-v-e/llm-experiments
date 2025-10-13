"""Device management and pipeline caching"""

from typing import Dict, Optional

import wgpu

from .gpu_types import (
    GPUConfig,
    GPUContext,
    KernelTimeStats,
    PerfMonitor,
    PerfStats,
    PipelineCache,
)

__all__ = ["wgpu"]

# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================


def device_create() -> wgpu.GPUDevice:
    """
    Create a new WGPU device

    Attempts to initialize WGPU with high-performance adapter.
    Falls back to default adapter if high-performance is unavailable.

    Args:
        config: Optional GPU configuration. If None, auto-detects from device.

    Returns:
        Device state if successful, None if WGPU unavailable or initialization fails

    Raises:
        None - exceptions are caught and logged
    """

    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        wgpu_device = adapter.request_device_sync()

        print("WGPU device initialized")

        return wgpu_device

    except Exception as e:
        raise RuntimeError(f"WGPU initialization failed: {e}")


def pipeline_cache_create(device: wgpu.GPUDevice) -> PipelineCache:
    """Create a new pipeline cache.

    Returns:
        New empty pipeline cache for caching compiled shaders
    """
    return PipelineCache()


def device_limits_query(device: wgpu.GPUDevice) -> Dict[str, int]:
    """Query device capabilities for kernel optimization.

    Returns default limits if device query fails, ensuring code
    always has valid limits to work with.

    Args:
        device: GPU device state

    Returns:
        Dictionary of device limits with keys:
        - max_compute_workgroup_size_x
        - max_compute_workgroup_size_y
        - max_compute_workgroup_size_z
        - max_compute_invocations_per_workgroup
        - max_compute_workgroup_storage_size
    """
    limits = {
        "max_compute_workgroup_size_x": 256,
        "max_compute_workgroup_size_y": 256,
        "max_compute_workgroup_size_z": 64,
        "max_compute_invocations_per_workgroup": 256,
        "max_compute_workgroup_storage_size": 16384,
    }

    # Try to query actual device limits if available
    try:
        adapter_info = device.adapter.request_adapter_info()
        if hasattr(adapter_info, "limits"):
            limits.update(adapter_info.limits)
    except Exception:
        pass  # Use defaults

    return limits


def select_optimal_tile_size(
    device_limits: Dict[str, int], M: int, N: int, K: int
) -> int:
    """Select optimal tile size based on matrix dimensions and device.

    Selects tile size to maximize performance while respecting device
    shared memory constraints. Larger tiles are preferred for large
    matrices when memory allows.

    Args:
        device_limits: Device capability limits from query_device_limits
        M: Matrix dimension M
        N: Matrix dimension N
        K: Matrix dimension K

    Returns:
        Optimal tile size (8, 16, or 32)
    """
    max_shared_mem = device_limits.get("max_compute_workgroup_storage_size", 16384)

    # Each tile needs 2 * tile_size^2 * 4 bytes (two tiles, float32)
    # Plus some overhead for other shared memory
    max_tile_from_memory = int(((max_shared_mem * 0.8) / 8) ** 0.5)

    # Common tile sizes: 8, 16, 32
    candidate_sizes = [8, 16, 32]

    # Filter by memory constraints
    valid_sizes = [s for s in candidate_sizes if s <= max_tile_from_memory]
    if not valid_sizes:
        return 8

    # Prefer 16 for most cases, 32 for large matrices
    if max(M, N, K) >= 2048 and 32 in valid_sizes:
        return 32
    elif 16 in valid_sizes:
        return 16
    else:
        return valid_sizes[-1]


def pipeline_tuned_create(
    ctx: GPUContext,
    kernel_code: str,
    tune_params: Optional[Dict[str, int]] = None,
) -> wgpu.GPUComputePipeline:
    """Create pipeline with device-specific tuning.

    Args:
        kernel_code: WGSL kernel source with optional {param} placeholders
        tune_params: Optional dictionary of parameter substitutions

    Returns:
        Compiled compute pipeline
    """
    # NOTE: limits must be used in the tuning
    limits = device_limits_query(ctx.device)

    tuned_code = kernel_code
    if tune_params:
        for key, value in tune_params.items():
            tuned_code = tuned_code.replace(f"{{{key}}}", str(value))

    return pipeline_get_or_create(ctx, tuned_code)


def pipeline_get_or_create(
    ctx: GPUContext, shader_code: str
) -> wgpu.GPUComputePipeline:
    """Cache compute pipelines to avoid recompilation.

    Uses SHA256 hash of shader code to avoid collisions.
    Previously used Python hash() which can collide for different shaders.

    Args:
        shader_code: WGSL shader source code

    Returns:
        Cached or newly compiled compute pipeline
    """
    import hashlib

    shader_hash = hashlib.sha256(shader_code.encode("utf-8")).hexdigest()

    if shader_hash not in ctx.pipeline_cache.pipelines:
        shader_module = ctx.device.create_shader_module(code=shader_code)
        pipeline = ctx.device.create_compute_pipeline(
            layout="auto",
            compute={
                "module": shader_module,
                "entry_point": "main",
            },
        )
        ctx.pipeline_cache.pipelines[shader_hash] = pipeline

    return ctx.pipeline_cache.pipelines[shader_hash]


# ============================================================================
# CONFIGURATION
# ============================================================================

"""GPU configuration and auto-tuning"""


def device_config_auto_detect(
    adapter: wgpu.GPUAdapter, device: wgpu.GPUDevice
) -> GPUConfig:
    """
    Auto-detect GPU capabilities and return optimized configuration.

    Queries GPU device limits and adapter properties to determine optimal
    kernel parameters. Falls back to conservative defaults if detection fails.

    Args:
        adapter: WGPU adapter (from wgpu.gpu.request_adapter_sync())
        device: WGPU device (from adapter.request_device_sync())

    Returns:
        GPUConfig optimized for the detected GPU

    Example:
        >>> import wgpu
        >>> adapter = wgpu.gpu.request_adapter_sync()
        >>> device = adapter.request_device_sync()
        >>> config = auto_detect_config(adapter, device)
    """
    limits = device.limits

    # ========================================================================
    # Detect optimal workgroup size
    # ========================================================================
    max_workgroup_size_x = limits["max_compute_workgroup_size_x"]

    if max_workgroup_size_x >= 512:
        default_wg = 256  # Conservative: avoid register spilling
    elif max_workgroup_size_x >= 256:
        default_wg = 256
    elif max_workgroup_size_x >= 128:
        default_wg = 128
    else:
        default_wg = 64  # Low-end GPU

    # ========================================================================
    # Detect optimal matmul tile size
    # ========================================================================
    # Larger tiles = better cache utilization but require more shared memory
    max_workgroup_storage_size = getattr(
        limits, "max_compute_workgroup_storage_size", 16384
    )

    # Matmul needs 2 tiles (A and B) of size tile_size^2 * 4 bytes
    if (
        max_workgroup_storage_size >= 32 * 32 * 2 * 4
        and max_workgroup_size_x >= 32 * 32
    ):
        matmul_tile = 32  # High-end GPU
    elif (
        max_workgroup_storage_size >= 16 * 16 * 2 * 4
        and max_workgroup_size_x >= 16 * 16
    ):
        matmul_tile = 16  # Mid-range GPU
    else:
        matmul_tile = 8  # Low-end/mobile GPU

    # ========================================================================
    # Detect head dimension support for FlashAttention
    # ========================================================================
    # FlashAttention needs significant workgroup memory for Q, K, V tiles
    # Conservative estimate: Qi (Br*head_dim) + Kj (Bc*head_dim) + Vj (Bc*head_dim) +
    #                        Sij (Br*Bc) + Pij (Br*Bc) + stats

    flash_br = 32
    flash_bc = 32

    # For head_dim=256: (32*256 + 32*256 + 32*256 + 32*32 + 32*32) * 4 bytes = ~106KB
    # For head_dim=128: (32*128 + 32*128 + 32*128 + 32*32 + 32*32) * 4 bytes = ~54KB
    # For head_dim=64:  (32*64 + 32*64 + 32*64 + 32*32 + 32*32) * 4 bytes = ~30KB

    if max_workgroup_storage_size >= 106496:  # ~104KB
        head_dim_max = 256
    elif max_workgroup_storage_size >= 53248:  # ~52KB
        head_dim_max = 128
    elif max_workgroup_storage_size >= 28672:  # ~28KB
        head_dim_max = 64
    else:
        head_dim_max = 64  # Conservative fallback

    # ========================================================================
    # Detect optimal FlashAttention block sizes
    # ========================================================================
    # Larger blocks = better but need more memory
    if max_workgroup_storage_size >= 65536:  # 64KB
        flash_bc = 64
        flash_br = 64
    elif max_workgroup_storage_size >= 32768:  # 32KB
        flash_bc = 32
        flash_br = 32
    else:
        flash_bc = 16
        flash_br = 16

    # ========================================================================
    # Memory limits
    # ========================================================================
    # Query total device memory if available
    # Note: WGPU spec doesn't expose total memory, so we use heuristics
    max_buffer_size = getattr(limits, "max_buffer_size", 2**30)  # Default: 1GB

    if max_buffer_size >= 8 * 2**30:  # >= 8GB
        buffer_pool_mb = 2048  # High-end GPU
    elif max_buffer_size >= 4 * 2**30:  # >= 4GB
        buffer_pool_mb = 1024  # Mid-range GPU
    elif max_buffer_size >= 2 * 2**30:  # >= 2GB
        buffer_pool_mb = 512
    else:
        buffer_pool_mb = 256  # Low-end GPU

    # ========================================================================
    # Build configuration
    # ========================================================================
    return GPUConfig(
        # Kernel parameters
        matmul_tile_size=matmul_tile,
        flash_attn_bc=flash_bc,
        flash_attn_br=flash_br,
        flash_attn_max_head_dim=head_dim_max,
        # Workgroup sizes
        default_workgroup_size=default_wg,
        layernorm_workgroup_size=default_wg,
        attention_workgroup_size=default_wg,
        # Memory limits
        buffer_pool_max_mb=buffer_pool_mb,
        buffer_pool_max_buffer_mb=buffer_pool_mb,
        workspace_lru_keep_count=2,  # Keep default
        staging_buffer_threshold_kb=256,  # Keep default
        staging_buffer_max_entries=8,  # Keep default
        # Compute limits (keep defaults)
        max_workgroups_per_dim=65535,
        max_batch_operations=1000,
        # Numerical stability (keep defaults)
        layernorm_epsilon=1e-5,
        optimizer_epsilon=1e-8,
    )


def device_config_create(device: wgpu.GPUDevice) -> GPUConfig:
    """
    Create GPU configuration tuned for specific device.

    Auto-tunes parameters based on device name if provided.
    Falls back to default config if device not recognized.

    **Note**: This function uses heuristics. For accurate detection,
    use auto_detect_config() with actual WGPU adapter/device objects.

    Args:
        device_name: GPU device name (e.g., "NVIDIA RTX 4090", "Apple M2")
                    None = use defaults

    Returns:
        GPUConfig tuned for the specified device
    """
    default_config = GPUConfig()

    device_name = device.adapter_info["device"]
    if device_name is None:
        return default_config

    # Normalize device name for matching
    device_lower = device_name.lower()

    # NVIDIA devices
    if "nvidia" in device_lower or "geforce" in device_lower or "rtx" in device_lower:
        return GPUConfig(
            matmul_tile_size=16,
            flash_attn_bc=32,
            flash_attn_br=32,
            flash_attn_max_head_dim=128,  # Modern NVIDIA supports head_dim=128
            default_workgroup_size=256,
            layernorm_workgroup_size=256,
            attention_workgroup_size=256,
            buffer_pool_max_mb=1024,  # NVIDIA typically has more VRAM
            buffer_pool_max_buffer_mb=1024,
        )

    # AMD devices
    elif "amd" in device_lower or "radeon" in device_lower:
        return GPUConfig(
            matmul_tile_size=16,
            flash_attn_bc=32,
            flash_attn_br=32,
            flash_attn_max_head_dim=128,
            default_workgroup_size=256,
            layernorm_workgroup_size=256,
            attention_workgroup_size=256,
            buffer_pool_max_mb=512,
            buffer_pool_max_buffer_mb=512,
        )

    # Intel devices
    elif "intel" in device_lower:
        return GPUConfig(
            matmul_tile_size=8,  # Intel integrated GPUs have less shared memory
            flash_attn_bc=16,
            flash_attn_br=16,
            flash_attn_max_head_dim=64,  # Conservative for integrated GPUs
            default_workgroup_size=128,
            layernorm_workgroup_size=128,
            attention_workgroup_size=128,
            buffer_pool_max_mb=256,
            buffer_pool_max_buffer_mb=256,
        )

    # Apple Silicon
    elif (
        "apple" in device_lower
        or "m1" in device_lower
        or "m2" in device_lower
        or "m3" in device_lower
    ):
        return GPUConfig(
            matmul_tile_size=16,
            flash_attn_bc=32,
            flash_attn_br=32,
            flash_attn_max_head_dim=128,
            default_workgroup_size=512,  # Apple GPUs have high thread count
            layernorm_workgroup_size=512,
            attention_workgroup_size=512,
            buffer_pool_max_mb=512,
            buffer_pool_max_buffer_mb=512,
        )

    else:
        return default_config


def device_config_validate(config: GPUConfig) -> None:
    """
    Validate GPU configuration for correctness.

    Raises ValueError if configuration has invalid parameters.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If any parameter is invalid
    """
    # Tile sizes must be positive powers of 2
    if (
        config.matmul_tile_size <= 0
        or (config.matmul_tile_size & (config.matmul_tile_size - 1)) != 0
    ):
        raise ValueError(
            f"matmul_tile_size must be power of 2, got {config.matmul_tile_size}"
        )

    if config.matmul_tile_size > 32:
        raise ValueError(
            f"matmul_tile_size too large: {config.matmul_tile_size}. "
            "Maximum is 32 due to shared memory limits."
        )

    # FlashAttention parameters
    if config.flash_attn_bc <= 0 or config.flash_attn_br <= 0:
        raise ValueError(
            f"FlashAttention block sizes must be positive: "
            f"bc={config.flash_attn_bc}, br={config.flash_attn_br}"
        )

    if config.flash_attn_max_head_dim not in [64, 128, 256]:
        raise ValueError(
            f"flash_attn_max_head_dim must be 64, 128, or 256 for kernel compatibility, "
            f"got {config.flash_attn_max_head_dim}"
        )

    # Workgroup sizes
    if config.default_workgroup_size <= 0:
        raise ValueError(
            f"default_workgroup_size must be positive, got {config.default_workgroup_size}"
        )

    if config.default_workgroup_size > 1024:
        raise ValueError(
            f"default_workgroup_size too large: {config.default_workgroup_size}. "
            "WebGPU limit is 1024."
        )

    if config.layernorm_workgroup_size <= 0 or config.layernorm_workgroup_size > 1024:
        raise ValueError(
            f"layernorm_workgroup_size must be in (0, 1024], got {config.layernorm_workgroup_size}"
        )

    if config.attention_workgroup_size <= 0 or config.attention_workgroup_size > 1024:
        raise ValueError(
            f"attention_workgroup_size must be in (0, 1024], got {config.attention_workgroup_size}"
        )

    # Memory limits
    if config.buffer_pool_max_mb < 0:
        raise ValueError(
            f"buffer_pool_max_mb must be non-negative, got {config.buffer_pool_max_mb}"
        )

    if config.buffer_pool_max_buffer_mb < 0:
        raise ValueError(
            f"buffer_pool_max_buffer_mb must be non-negative, got {config.buffer_pool_max_buffer_mb}"
        )

    if config.workspace_lru_keep_count < 0:
        raise ValueError(
            f"workspace_lru_keep_count must be non-negative, got {config.workspace_lru_keep_count}"
        )

    # Transfer thresholds
    if config.staging_buffer_threshold_kb < 0:
        raise ValueError(
            f"staging_buffer_threshold_kb must be non-negative, got {config.staging_buffer_threshold_kb}"
        )

    if config.staging_buffer_max_entries < 0:
        raise ValueError(
            f"staging_buffer_max_entries must be non-negative, got {config.staging_buffer_max_entries}"
        )

    # Numerical parameters
    if config.layernorm_epsilon <= 0:
        raise ValueError(
            f"layernorm_epsilon must be positive, got {config.layernorm_epsilon}"
        )

    if config.optimizer_epsilon <= 0:
        raise ValueError(
            f"optimizer_epsilon must be positive, got {config.optimizer_epsilon}"
        )


def device_config_shared_memory_usage_estimate(config: GPUConfig) -> Dict[str, int]:
    """
    Estimate shared memory usage for different kernels.

    Helps validate configuration against GPU shared memory limits.
    Typical limits: 16KB (integrated), 32KB (mid-range), 48-96KB (high-end)

    Args:
        config: GPU configuration

    Returns:
        Dictionary with shared memory estimates in bytes for each kernel type
    """
    return {
        "matmul": config.matmul_tile_size
        * config.matmul_tile_size
        * 2
        * 4,  # 2 tiles, fp32
        "flashattention": (
            config.flash_attn_br * config.flash_attn_max_head_dim * 4  # Qi
            + config.flash_attn_bc * config.flash_attn_max_head_dim * 2 * 4  # Kj, Vj
            + config.flash_attn_br * config.flash_attn_bc * 3 * 4  # Sij, Pij, dSij
            + config.flash_attn_br * 4  # row statistics
        ),
        "layernorm": config.default_workgroup_size * 4,  # Shared reduction buffer
        "attention": config.attention_workgroup_size * 4,  # Shared reduction buffer
    }


# ============================================================================
# PROFILING
# ============================================================================


def perf_monitor_create() -> PerfMonitor:
    """Create performance monitor state"""
    return PerfMonitor()


def perf_monitor_kernel_time_record(
    monitor: PerfMonitor, kernel_name: str, duration_ms: float
) -> None:
    """Record kernel execution time"""
    if kernel_name not in monitor.kernel_times:
        monitor.kernel_times[kernel_name] = []
    monitor.kernel_times[kernel_name].append(duration_ms)


def perf_monitor_submission_record(monitor: PerfMonitor) -> None:
    """Increment submission counter"""
    monitor.submission_count += 1


def perf_monitor_stats_get(monitor: PerfMonitor) -> PerfStats:
    """Get performance statistics"""
    kernel_stats = {}
    for kernel_name, times in monitor.kernel_times.items():
        kernel_stats[kernel_name] = KernelTimeStats(
            count=len(times),
            total_ms=sum(times),
            avg_ms=sum(times) / len(times) if times else 0,
            min_ms=min(times) if times else 0,
            max_ms=max(times) if times else 0,
        )
    return PerfStats(
        total_submissions=monitor.submission_count, kernel_times=kernel_stats
    )


def perf_monitor_reset(monitor: PerfMonitor) -> None:
    """Reset all counters"""
    monitor.kernel_times.clear()
    monitor.memory_usage.clear()
    monitor.submission_count = 0
