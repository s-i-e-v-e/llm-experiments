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
