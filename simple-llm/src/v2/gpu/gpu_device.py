"""Device management and pipeline caching"""

try:
    import wgpu

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None

# Module-level state
_device = None
_pipeline_cache = {}


def get_device():
    """Get or create the default WGPU device"""
    global _device
    if not WGPU_AVAILABLE:
        return None

    if _device is None:
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            _device = adapter.request_device_sync()
            print("✅ WGPU device initialized")
        except Exception as e:
            print(f"⚠️ WGPU initialization failed: {e}")
            _device = None
    return _device


def query_device_limits(device):
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
        adapter_info = device.adapter.request_adapter_info()
        if hasattr(adapter_info, "limits"):
            limits.update(adapter_info.limits)
    except:
        pass  # Use defaults

    return limits


def select_optimal_tile_size(M, N, K, device_limits):
    """Select optimal tile size based on matrix dimensions and device"""
    max_shared_mem = device_limits.get("max_compute_workgroup_storage_size", 16384)

    # Each tile needs 2 * tile_size^2 * 4 bytes (two tiles, float32)
    # Plus some overhead for other shared memory
    max_tile_from_memory = int((max_shared_mem * 0.8 / 8) ** 0.5)

    # Common tile sizes: 8, 16, 32
    candidate_sizes = [8, 16, 32]

    # Filter by memory constraints
    valid_sizes = [s for s in candidate_sizes if s <= max_tile_from_memory]

    if not valid_sizes:
        return 8

    # Prefer 16 for most cases, 32 for large matrices
    if max(M, N, K) > 2048 and 32 in valid_sizes:
        return 32
    elif 16 in valid_sizes:
        return 16
    else:
        return valid_sizes[-1]


def create_tuned_pipeline(kernel_code, device, tune_params=None):
    """Create pipeline with device-specific tuning"""
    limits = query_device_limits(device)

    # Apply tuning if provided
    if tune_params:
        for key, value in tune_params.items():
            kernel_code = kernel_code.replace(f"${key}$", str(value))

    return get_or_create_pipeline(kernel_code, device)


def get_or_create_pipeline(shader_code, device=None):
    """Cache compute pipelines"""
    device = device or get_device()
    cache_key = (id(device), hash(shader_code))

    if cache_key not in _pipeline_cache:
        shader_module = device.create_shader_module(code=shader_code)
        pipeline = device.create_compute_pipeline(
            layout="auto", compute={"module": shader_module, "entry_point": "main"}
        )
        _pipeline_cache[cache_key] = pipeline

    return _pipeline_cache[cache_key]
