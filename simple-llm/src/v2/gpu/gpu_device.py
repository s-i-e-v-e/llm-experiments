"""Device management and pipeline caching"""

from typing import Dict, List, Optional

from gpu_types import (
    BindGroupEntry,
    Device,
    PipelineCache,
    WGPUComputePipeline,
)

try:
    import wgpu

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None


# ============================================================================
# BIND GROUP HELPERS
# ============================================================================


def create_bind_group_entries(entries: List[BindGroupEntry]) -> List[Dict]:
    """
    Convert typed BindGroupEntry list to wgpu bind group entry format.

    Args:
        entries: List of BindGroupEntry specifications

    Returns:
        List of dictionaries in wgpu bind group format
    """
    return [
        {
            "binding": entry.binding,
            "resource": {
                "buffer": entry.buffer,
                "offset": entry.offset,
                "size": entry.size,
            },
        }
        for entry in entries
    ]


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================


def create_device() -> Optional[Device]:
    """Create a new WGPU device"""
    if not WGPU_AVAILABLE:
        return None

    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        wgpu_device = adapter.request_device_sync()
        print("WGPU device initialized")
        return Device(wgpu_device=wgpu_device, adapter=adapter)
    except Exception as e:
        print(f"WGPU initialization failed: {e}")
        return None


def create_pipeline_cache(device: Device) -> PipelineCache:
    """Create a new pipeline cache for the given device"""
    return PipelineCache(device=device)


def query_device_limits(device: Device) -> Dict[str, int]:
    """Query device capabilities for kernel optimization"""
    limits = {
        "max_compute_workgroup_size_x": 256,
        "max_compute_workgroup_size_y": 256,
        "max_compute_workgroup_size_z": 64,
        "max_compute_invocations_per_workgroup": 256,
        "max_compute_workgroup_storage_size": 16384,
    }

    # Try to query actual device limits if available
    try:
        adapter_info = device.wgpu_device.adapter.request_adapter_info()
        if hasattr(adapter_info, "limits"):
            limits.update(adapter_info.limits)
    except:
        pass  # Use defaults

    return limits


def select_optimal_tile_size(
    device_limits: Dict[str, int], M: int, N: int, K: int
) -> int:
    """Select optimal tile size based on matrix dimensions and device"""
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


def create_tuned_pipeline(
    pipeline_cache: PipelineCache,
    kernel_code: str,
    tune_params: Optional[Dict[str, int]] = None,
) -> WGPUComputePipeline:
    """Create pipeline with device-specific tuning"""
    limits = query_device_limits(pipeline_cache.device)

    # Apply tuning if provided
    if tune_params:
        for key, value in tune_params.items():
            kernel_code = kernel_code.replace(f"{{{key}}}", str(value))

    return get_or_create_pipeline(pipeline_cache, kernel_code)


def get_or_create_pipeline(
    pipeline_cache: PipelineCache, shader_code: str
) -> WGPUComputePipeline:
    """
    Cache compute pipelines to avoid recompilation.

    Uses SHA256 hash of shader code to avoid collisions.
    Previously used Python hash() which can collide for different shaders.
    """
    import hashlib

    device = pipeline_cache.device

    # FIXED: Use SHA256 instead of hash() to avoid collisions
    shader_hash = hashlib.sha256(shader_code.encode("utf-8")).hexdigest()
    cache_key = (id(device.wgpu_device), shader_hash)

    if cache_key not in pipeline_cache.pipelines:
        shader_module = device.wgpu_device.create_shader_module(code=shader_code)
        pipeline = device.wgpu_device.create_compute_pipeline(
            layout="auto",
            compute={
                "module": shader_module,
                "entry_point": "main",
            },
        )
        pipeline_cache.pipelines[cache_key] = pipeline

    return pipeline_cache.pipelines[cache_key]
