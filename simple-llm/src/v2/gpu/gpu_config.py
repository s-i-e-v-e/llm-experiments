"""GPU configuration and auto-tuning"""

from typing import Optional


def create_default_config() -> GPUConfig:
    """
    Create default GPU configuration with conservative settings.

    These settings work on most GPUs but may not be optimal for all hardware.
    For automatic optimization, use auto_detect_config() instead.

    Returns:
        GPUConfig with default parameters
    """
    return GPUConfig()


def auto_detect_config(adapter: WGPUAdapter, device: WGPUDevice) -> GPUConfig:
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
    max_workgroup_size_x = limits.max_compute_workgroup_size_x

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


def create_config_for_device(device_name: Optional[str] = None) -> GPUConfig:
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
    if device_name is None:
        return create_default_config()

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

    # Unknown device - use defaults
    else:
        return create_default_config()


def validate_config(config: GPUConfig) -> None:
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


def estimate_shared_memory_usage(config: GPUConfig) -> dict:
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
