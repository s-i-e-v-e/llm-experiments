import dataclasses
from typing import Tuple

import numpy as np

try:
    import wgpu

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None
# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

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

    return _get_or_create_pipeline(kernel_code, device)


class GPUPerformanceMonitor:
    """Monitor GPU performance metrics"""

    def __init__(self):
        self.kernel_times = {}
        self.memory_usage = {}
        self.submission_count = 0

    def record_kernel_time(self, kernel_name, duration_ms):
        """Record kernel execution time"""
        if kernel_name not in self.kernel_times:
            self.kernel_times[kernel_name] = []
        self.kernel_times[kernel_name].append(duration_ms)

    def record_submission(self):
        """Increment submission counter"""
        self.submission_count += 1

    def get_stats(self):
        """Get performance statistics"""
        stats = {"total_submissions": self.submission_count, "kernel_times": {}}

        for kernel_name, times in self.kernel_times.items():
            stats["kernel_times"][kernel_name] = {
                "count": len(times),
                "total_ms": sum(times),
                "avg_ms": sum(times) / len(times) if times else 0,
                "min_ms": min(times) if times else 0,
                "max_ms": max(times) if times else 0,
            }

        return stats

    def reset(self):
        """Reset all counters"""
        self.kernel_times.clear()
        self.memory_usage.clear()
        self.submission_count = 0


# Global monitor instance
_perf_monitor = GPUPerformanceMonitor()


def get_performance_monitor():
    """Get global performance monitor"""
    return _perf_monitor


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclasses.dataclass
class GPUBuffer:
    buffer: object
    shape: Tuple[int, ...]
    size: int
    device: object


@dataclasses.dataclass
class GPULayerParams:
    attn_wq: GPUBuffer
    attn_wk: GPUBuffer
    attn_wv: GPUBuffer
    attn_wo: GPUBuffer
    ff_w1: GPUBuffer
    ff_b1: GPUBuffer
    ff_w2: GPUBuffer
    ff_b2: GPUBuffer
    ln_gamma1: GPUBuffer
    ln_beta1: GPUBuffer
    ln_gamma2: GPUBuffer
    ln_beta2: GPUBuffer


@dataclasses.dataclass
class GPUModelParams:
    embedding: GPUBuffer
    pos_encoding: GPUBuffer
    layers: list  # List of GPULayerParams


@dataclasses.dataclass
class GPUOptimizerState:
    m_embedding: GPUBuffer
    v_embedding: GPUBuffer
    m_layers: list  # List of GPULayerParams (momentum)
    v_layers: list  # List of GPULayerParams (variance)
    step: int


# ============================================================================
# BUFFER MANAGEMENT
# ============================================================================
def create_buffer_pool(device, max_buffer_size_mb=512):
    """Create a memory pool for reusable GPU buffers"""
    return BufferPool(device, max_buffer_size_mb)


class BufferPool:
    """Memory pool for reusing GPU buffers across training steps"""

    def __init__(self, device, max_size_mb=512):
        self.device = device
        self.max_size = max_size_mb * 1024 * 1024 // 4  # Convert to float32 count
        self.pools = {}  # size -> list of buffers
        self.in_use = set()  # Track buffers currently in use

    def get_buffer(self, shape):
        """Get a buffer from pool or create new"""
        size = int(np.prod(shape))

        if size in self.pools and len(self.pools[size]) > 0:
            buffer_info = self.pools[size].pop()
            self.in_use.add(id(buffer_info["buffer"]))
            return GPUBuffer(
                buffer=buffer_info["buffer"], shape=shape, size=size, device=self.device
            )

        # Create new buffer
        buffer_size = size * 4
        buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )

        gpu_buffer = GPUBuffer(
            buffer=buffer, shape=shape, size=size, device=self.device
        )
        self.in_use.add(id(buffer))
        return gpu_buffer

    def release_buffer(self, gpu_buffer):
        """Return buffer to pool for reuse"""
        buffer_id = id(gpu_buffer.buffer)
        if buffer_id in self.in_use:
            self.in_use.remove(buffer_id)

            size = gpu_buffer.size
            if size not in self.pools:
                self.pools[size] = []

            self.pools[size].append({"buffer": gpu_buffer.buffer, "size": size})


class StagingBufferPool:
    """Manages persistent staging buffers to reduce allocation overhead"""

    def __init__(self, device, initial_size_mb=64):
        self.device = device
        self.staging_buffers = {}  # size -> buffer
        self.max_size = initial_size_mb * 1024 * 1024

    def get_staging_buffer(self, size_bytes):
        """Get or create staging buffer for CPU->GPU or GPU->CPU transfers"""
        # Round up to next power of 2 for better pooling
        rounded_size = 2 ** (size_bytes - 1).bit_length()
        rounded_size = min(rounded_size, self.max_size)

        if rounded_size not in self.staging_buffers:
            self.staging_buffers[rounded_size] = self.device.create_buffer(
                size=rounded_size,
                usage=wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST
                | wgpu.BufferUsage.MAP_READ
                | wgpu.BufferUsage.MAP_WRITE,
            )

        return self.staging_buffers[rounded_size]

    def upload_data(self, gpu_buffer, data_np):
        """Upload data to GPU using persistent staging buffer"""
        size_bytes = data_np.nbytes

        # For small transfers, use direct write
        if size_bytes < 256 * 1024:  # 256KB threshold
            self.device.queue.write_buffer(
                gpu_buffer.buffer, 0, np.ascontiguousarray(data_np, dtype=np.float32)
            )
            return

        # For large transfers, use staging buffer
        staging = self.get_staging_buffer(size_bytes)
        staging.map_sync(wgpu.MapMode.WRITE)
        staging.write_mapped(np.ascontiguousarray(data_np, dtype=np.float32))
        staging.unmap()

        encoder = self.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(staging, 0, gpu_buffer.buffer, 0, size_bytes)
        self.device.queue.submit([encoder.finish()])

    def download_data(self, gpu_buffer, shape):
        """Download data from GPU using persistent staging buffer"""
        size_bytes = gpu_buffer.size * 4
        staging = self.get_staging_buffer(size_bytes)

        encoder = self.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(gpu_buffer.buffer, 0, staging, 0, size_bytes)
        self.device.queue.submit([encoder.finish()])

        staging.map_sync(wgpu.MapMode.READ)
        data = np.frombuffer(
            staging.read_mapped(), dtype=np.float32, count=gpu_buffer.size
        ).copy()
        staging.unmap()

        return data.reshape(shape)


# Global pool instance
_staging_pool = None


def get_staging_pool():
    """Get or create global staging buffer pool"""
    global _staging_pool
    if _staging_pool is None:
        device = get_device()
        if device is not None:
            _staging_pool = StagingBufferPool(device)
    return _staging_pool


def gpu_to_numpy_optimized(gpu_buffer):
    """Optimized GPU to numpy with staging buffer pool"""
    pool = get_staging_pool()
    if pool is not None:
        return pool.download_data(gpu_buffer, gpu_buffer.shape)
    else:
        # Fallback to original implementation
        return gpu_to_numpy(gpu_buffer)


def create_gpu_buffer(shape, data=None, device=None):
    """Create GPU buffer"""
    device = device or get_device()
    size = int(np.prod(shape))
    buffer_size = size * 4  # 4 bytes per float32

    if data is not None:
        data_np = np.ascontiguousarray(data, dtype=np.float32).flatten()
        buffer = device.create_buffer_with_data(
            data=data_np,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )
    else:
        buffer = device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )

    return GPUBuffer(buffer=buffer, shape=shape, size=size, device=device)


def gpu_to_numpy(gpu_buffer):
    """Read GPU buffer back to CPU"""
    buffer_size = gpu_buffer.size * 4
    read_buffer = gpu_buffer.device.create_buffer(
        size=buffer_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )

    encoder = gpu_buffer.device.create_command_encoder()
    encoder.copy_buffer_to_buffer(gpu_buffer.buffer, 0, read_buffer, 0, buffer_size)
    gpu_buffer.device.queue.submit([encoder.finish()])

    read_buffer.map_sync(wgpu.MapMode.READ)
    data = np.frombuffer(read_buffer.read_mapped(), dtype=np.float32).copy()
    read_buffer.unmap()

    return data.reshape(gpu_buffer.shape)


def clear_buffer(gpu_buffer: GPUBuffer):
    """Zero-initialize a GPU buffer"""
    device = gpu_buffer.device
    zero_data = np.zeros(gpu_buffer.size, dtype=np.float32)
    device.queue.write_buffer(gpu_buffer.buffer, 0, zero_data)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def positional_encoding(seq_len: int, dim: int) -> np.ndarray:
    """Generate sinusoidal positional encoding"""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(dim)[None, :]
    angle_rates = 1 / (10000 ** (2 * (i // 2) / dim))
    angle_rads = pos * angle_rates

    pos_encoding = np.zeros((seq_len, dim), dtype=np.float32)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return pos_encoding


def _get_or_create_pipeline(shader_code: str, device=None):
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


# ============================================================================
# MODEL CREATION
# ============================================================================


def create_gpu_layer_params(embedding_dim: int, device=None) -> GPULayerParams:
    """Initialize GPU layer parameters"""
    device = device or get_device()
    dim = embedding_dim

    return GPULayerParams(
        attn_wq=create_gpu_buffer(
            (dim, dim), np.random.normal(0, 0.02, (dim, dim)).astype(np.float32), device
        ),
        attn_wk=create_gpu_buffer(
            (dim, dim), np.random.normal(0, 0.02, (dim, dim)).astype(np.float32), device
        ),
        attn_wv=create_gpu_buffer(
            (dim, dim), np.random.normal(0, 0.02, (dim, dim)).astype(np.float32), device
        ),
        attn_wo=create_gpu_buffer(
            (dim, dim), np.random.normal(0, 0.02, (dim, dim)).astype(np.float32), device
        ),
        ff_w1=create_gpu_buffer(
            (dim, 4 * dim),
            np.random.normal(0, 0.02, (dim, 4 * dim)).astype(np.float32),
            device,
        ),
        ff_b1=create_gpu_buffer(
            (4 * dim,), np.zeros(4 * dim, dtype=np.float32), device
        ),
        ff_w2=create_gpu_buffer(
            (4 * dim, dim),
            np.random.normal(0, 0.02, (4 * dim, dim)).astype(np.float32),
            device,
        ),
        ff_b2=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
        ln_gamma1=create_gpu_buffer((dim,), np.ones(dim, dtype=np.float32), device),
        ln_beta1=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
        ln_gamma2=create_gpu_buffer((dim,), np.ones(dim, dtype=np.float32), device),
        ln_beta2=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
    )


def create_gpu_model_params(
    vocab_size: int, embedding_dim: int, context_size: int, n_layers: int, device=None
) -> GPUModelParams:
    """Initialize complete GPU model"""
    device = device or get_device()

    embedding_data = np.random.normal(0, 0.02, (vocab_size, embedding_dim)).astype(
        np.float32
    )
    embedding = create_gpu_buffer((vocab_size, embedding_dim), embedding_data, device)

    pos_encoding_data = positional_encoding(context_size, embedding_dim)
    pos_encoding = create_gpu_buffer(
        (context_size, embedding_dim), pos_encoding_data, device
    )

    layers = [create_gpu_layer_params(embedding_dim, device) for _ in range(n_layers)]

    return GPUModelParams(embedding=embedding, pos_encoding=pos_encoding, layers=layers)


def create_optimizer_state(model_params: GPUModelParams) -> GPUOptimizerState:
    """Initialize optimizer state (zero moments)"""
    device = model_params.embedding.device

    m_embedding = create_gpu_buffer(
        model_params.embedding.shape,
        np.zeros(model_params.embedding.shape, dtype=np.float32),
        device,
    )
    v_embedding = create_gpu_buffer(
        model_params.embedding.shape,
        np.zeros(model_params.embedding.shape, dtype=np.float32),
        device,
    )

    m_layers = []
    v_layers = []
    for layer in model_params.layers:
        dim = layer.attn_wq.shape[0]

        m_layer = create_gpu_layer_params(dim, device)
        v_layer = create_gpu_layer_params(dim, device)

        # Zero initialize all buffers
        for attr in [
            "attn_wq",
            "attn_wk",
            "attn_wv",
            "attn_wo",
            "ff_w1",
            "ff_b1",
            "ff_w2",
            "ff_b2",
            "ln_gamma1",
            "ln_beta1",
            "ln_gamma2",
            "ln_beta2",
        ]:
            buf = getattr(m_layer, attr)
            device.queue.write_buffer(
                buf.buffer, 0, np.zeros(buf.size, dtype=np.float32)
            )
            buf = getattr(v_layer, attr)
            device.queue.write_buffer(
                buf.buffer, 0, np.zeros(buf.size, dtype=np.float32)
            )

        m_layers.append(m_layer)
        v_layers.append(v_layer)

    return GPUOptimizerState(
        m_embedding=m_embedding,
        v_embedding=v_embedding,
        m_layers=m_layers,
        v_layers=v_layers,
        step=0,
    )


# ============================================================================
# SERIALIZATION
# ============================================================================


def gpu_layer_to_dict(layer: GPULayerParams) -> dict:
    """Convert GPU layer to dict"""
    return {
        "attn_wq": gpu_to_numpy(layer.attn_wq),
        "attn_wk": gpu_to_numpy(layer.attn_wk),
        "attn_wv": gpu_to_numpy(layer.attn_wv),
        "attn_wo": gpu_to_numpy(layer.attn_wo),
        "ff_w1": gpu_to_numpy(layer.ff_w1),
        "ff_b1": gpu_to_numpy(layer.ff_b1),
        "ff_w2": gpu_to_numpy(layer.ff_w2),
        "ff_b2": gpu_to_numpy(layer.ff_b2),
        "ln_gamma1": gpu_to_numpy(layer.ln_gamma1),
        "ln_beta1": gpu_to_numpy(layer.ln_beta1),
        "ln_gamma2": gpu_to_numpy(layer.ln_gamma2),
        "ln_beta2": gpu_to_numpy(layer.ln_beta2),
    }


def dict_to_gpu_layer(data: dict, embedding_dim: int, device=None) -> GPULayerParams:
    """Create GPU layer from dict"""
    device = device or get_device()
    dim = embedding_dim

    return GPULayerParams(
        attn_wq=create_gpu_buffer((dim, dim), data["attn_wq"], device),
        attn_wk=create_gpu_buffer((dim, dim), data["attn_wk"], device),
        attn_wv=create_gpu_buffer((dim, dim), data["attn_wv"], device),
        attn_wo=create_gpu_buffer((dim, dim), data["attn_wo"], device),
        ff_w1=create_gpu_buffer((dim, 4 * dim), data["ff_w1"], device),
        ff_b1=create_gpu_buffer((4 * dim,), data["ff_b1"], device),
        ff_w2=create_gpu_buffer((4 * dim, dim), data["ff_w2"], device),
        ff_b2=create_gpu_buffer((dim,), data["ff_b2"], device),
        ln_gamma1=create_gpu_buffer((dim,), data["ln_gamma1"], device),
        ln_beta1=create_gpu_buffer((dim,), data["ln_beta1"], device),
        ln_gamma2=create_gpu_buffer((dim,), data["ln_gamma2"], device),
        ln_beta2=create_gpu_buffer((dim,), data["ln_beta2"], device),
    )
