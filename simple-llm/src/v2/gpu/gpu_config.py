"""GPU configuration and auto-tuning"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUConfig:
    """
    Centralized GPU configuration for kernel parameters and memory limits

    This dataclass is immutable - do not modify fields after creation.
    All parameters can be tuned for different GPU architectures.
    """

    # ========================================================================
    # KERNEL TILE SIZES
    # ========================================================================

    matmul_tile_size: int = 16
    """Tile size for matrix multiplication kernels (16x16 default)

    Optimal values:
    - Small GPUs (integrated): 8
    - Mid-range GPUs: 16
    - High-end GPUs: 32

    Constraints:
    - Must be power of 2
    - Shared memory usage = tile_size * tile_size * 2 * 4 bytes
    - 16x16 = 2KB shared memory per tile
    """

    # ========================================================================
    # FLASHATTENTION PARAMETERS
    # ========================================================================

    flash_attn_bc: int = 32
    """FlashAttention block size for KV (columns)

    Controls memory tiling for keys and values.
    Larger values = more shared memory usage but fewer kernel iterations.

    Constraints:
    - bc * head_dim * 4 bytes must fit in shared memory
    - bc=32, head_dim=64 → 8KB shared memory
    """

    flash_attn_br: int = 32
    """FlashAttention block size for Q (rows)

    Controls memory tiling for queries.
    Larger values = more shared memory usage but fewer kernel iterations.

    Constraints:
    - br * head_dim * 4 bytes must fit in shared memory
    - br=32, head_dim=64 → 8KB shared memory
    """

    flash_attn_max_head_dim: int = 64
    """Maximum head dimension for FlashAttention

    Hardcoded in WGSL due to static shared memory allocation.
    Larger values require kernel recompilation.

    WARNING: Changing this requires regenerating kernels!
    """

    # ========================================================================
    # WORKGROUP SIZES
    # ========================================================================

    default_workgroup_size: int = 256
    """Default workgroup size for 1D kernels

    Used for: LayerNorm, GELU, Bias, element-wise ops

    Optimal values:
    - NVIDIA: 256 or 512
    - AMD: 256
    - Intel: 128 or 256
    - Apple Silicon: 256 or 512
    """

    attention_workgroup_size: int = 256
    """Workgroup size for attention kernels (non-Flash)

    Each workgroup processes one query position.
    """

    flash_attn_workgroup_size: int = 32
    """Workgroup size for FlashAttention kernels

    Must match tile dimensions for efficient cooperation.
    Typically set to Bc or Br.
    """

    matmul_workgroup_dim: int = 16
    """Workgroup dimension for 2D matmul kernels

    Creates workgroups of size (dim x dim).
    Must match tile size for efficient tiling.
    """

    # ========================================================================
    # MEMORY LIMITS
    # ========================================================================

    buffer_pool_max_mb: int = 512
    """Maximum total memory for buffer pool (MB)

    0 = unlimited
    Recommended: 25-50% of total VRAM
    """

    buffer_pool_max_buffer_mb: int = 128
    """Maximum size for individual pooled buffer (MB)

    Prevents pool from caching very large buffers.
    """

    staging_buffer_max_entries: int = 8
    """Maximum number of different-sized staging buffers to cache

    Staging buffers are used for CPU-GPU transfers.
    Higher values = more memory but less allocation overhead.
    """

    workspace_lru_keep_count: int = 2
    """Number of workspaces to keep in LRU cache

    Workspaces are cached by (batch_size, seq_len).
    Higher values = more memory but less reallocation.
    """

    # ========================================================================
    # TRANSFER THRESHOLDS
    # ========================================================================

    staging_buffer_threshold_kb: int = 256
    """Threshold for using staging buffers vs direct transfer (KB)

    Buffers larger than this use staging buffer pool.
    Buffers smaller use direct queue.write_buffer.
    """

    # ========================================================================
    # COMPUTE LIMITS
    # ========================================================================

    max_workgroups_per_dim: int = 65535
    """Maximum workgroups per dimension (WGSL limit)

    This is a WebGPU spec limit and should not be changed.
    Used for validation and automatic tiling.
    """

    max_batch_operations: int = 1000
    """Maximum operations per batch submission

    Prevents unbounded command buffer growth.
    Batches are automatically submitted when this limit is reached.
    """

    # ========================================================================
    # NUMERICAL STABILITY
    # ========================================================================

    layernorm_epsilon: float = 1e-5
    """Epsilon for LayerNorm numerical stability

    Added to variance before taking square root.
    """

    optimizer_epsilon: float = 1e-8
    """Epsilon for AdamW optimizer numerical stability

    Added to denominator in adaptive learning rate computation.
    """


def create_default_config() -> GPUConfig:
    """
    Create default GPU configuration

    Returns balanced configuration suitable for most modern GPUs.
    For custom tuning, modify the returned config or create new GPUConfig directly.

    Returns:
        New GPUConfig with default values
    """
    return GPUConfig()


def create_config_for_device(device_name: Optional[str] = None) -> GPUConfig:
    """
    Create GPU configuration tuned for specific device

    Auto-tunes parameters based on device name if provided.
    Falls back to default config if device not recognized.

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
            default_workgroup_size=256,
            buffer_pool_max_mb=1024,  # NVIDIA typically has more VRAM
        )

    # AMD devices
    elif "amd" in device_lower or "radeon" in device_lower:
        return GPUConfig(
            matmul_tile_size=16,
            flash_attn_bc=32,
            flash_attn_br=32,
            default_workgroup_size=256,
            buffer_pool_max_mb=512,
        )

    # Intel devices
    elif "intel" in device_lower:
        return GPUConfig(
            matmul_tile_size=8,  # Intel integrated GPUs have less shared memory
            flash_attn_bc=16,
            flash_attn_br=16,
            default_workgroup_size=128,
            buffer_pool_max_mb=256,
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
            default_workgroup_size=512,  # Apple GPUs have high thread count
            buffer_pool_max_mb=512,
        )

    # Unknown device - use defaults
    else:
        return create_default_config()


def validate_config(config: GPUConfig) -> None:
    """
    Validate GPU configuration for correctness

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
            f"matmul_tile_size too large: {config.matmul_tile_size}. Maximum is 32 due to shared memory limits."
        )

    # FlashAttention parameters
    if config.flash_attn_bc <= 0 or config.flash_attn_br <= 0:
        raise ValueError(
            f"FlashAttention block sizes must be positive: bc={config.flash_attn_bc}, br={config.flash_attn_br}"
        )

    if config.flash_attn_max_head_dim not in [64, 128, 256]:
        raise ValueError(
            f"flash_attn_max_head_dim must be 64, 128, or 256 for kernel compatibility, got {config.flash_attn_max_head_dim}"
        )

    # Workgroup sizes
    if config.default_workgroup_size <= 0:
        raise ValueError(
            f"default_workgroup_size must be positive, got {config.default_workgroup_size}"
        )

    if config.default_workgroup_size > 1024:
        raise ValueError(
            f"default_workgroup_size too large: {config.default_workgroup_size}. WebGPU limit is 1024."
        )

    # Memory limits
    if config.buffer_pool_max_mb < 0:
        raise ValueError(
            f"buffer_pool_max_mb must be non-negative, got {config.buffer_pool_max_mb}"
        )

    if config.workspace_lru_keep_count < 0:
        raise ValueError(
            f"workspace_lru_keep_count must be non-negative, got {config.workspace_lru_keep_count}"
        )

    # Thresholds
    if config.staging_buffer_threshold_kb < 0:
        raise ValueError(
            f"staging_buffer_threshold_kb must be non-negative, got {config.staging_buffer_threshold_kb}"
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
    Estimate shared memory usage for different kernels

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
        "flash_attention": (
            config.flash_attn_br * config.flash_attn_max_head_dim * 4  # Qi
            + config.flash_attn_bc * config.flash_attn_max_head_dim * 2 * 4  # Kj, Vj
            + config.flash_attn_br * config.flash_attn_bc * 3 * 4  # Sij, Pij, dSij
            + config.flash_attn_br * 4  # row statistics
        ),
        "layernorm": config.default_workgroup_size * 4,  # Shared reduction buffer
        "attention": config.attention_workgroup_size * 4,  # Shared reduction buffer
    }
