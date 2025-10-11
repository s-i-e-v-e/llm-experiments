"""Model parameter initialization"""

from typing import Dict

import numpy as np
from gpu_buffer import create_gpu_buffer
from gpu_types import Device, GPULayerParams, GPUModelParams, GPUOptimizerState

# ============================================================================
# POSITIONAL ENCODING
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


# ============================================================================
# LAYER CREATION
# ============================================================================


def create_gpu_layer_params(device: Device, embedding_dim: int) -> GPULayerParams:
    """Initialize GPU layer parameters"""
    dim = embedding_dim

    return GPULayerParams(
        attn_wq=create_gpu_buffer(
            device, (dim, dim), np.random.normal(0, 0.02, (dim, dim)).astype(np.float32)
        ),
        attn_wk=create_gpu_buffer(
            device, (dim, dim), np.random.normal(0, 0.02, (dim, dim)).astype(np.float32)
        ),
        attn_wv=create_gpu_buffer(
            device, (dim, dim), np.random.normal(0, 0.02, (dim, dim)).astype(np.float32)
        ),
        attn_wo=create_gpu_buffer(
            device, (dim, dim), np.random.normal(0, 0.02, (dim, dim)).astype(np.float32)
        ),
        ff_w1=create_gpu_buffer(
            device,
            (dim, 4 * dim),
            np.random.normal(0, 0.02, (dim, 4 * dim)).astype(np.float32),
        ),
        ff_b1=create_gpu_buffer(
            device, (4 * dim,), np.zeros(4 * dim, dtype=np.float32)
        ),
        ff_w2=create_gpu_buffer(
            device,
            (4 * dim, dim),
            np.random.normal(0, 0.02, (4 * dim, dim)).astype(np.float32),
        ),
        ff_b2=create_gpu_buffer(device, (dim,), np.zeros(dim, dtype=np.float32)),
        ln_gamma1=create_gpu_buffer(device, (dim,), np.ones(dim, dtype=np.float32)),
        ln_beta1=create_gpu_buffer(device, (dim,), np.zeros(dim, dtype=np.float32)),
        ln_gamma2=create_gpu_buffer(device, (dim,), np.ones(dim, dtype=np.float32)),
        ln_beta2=create_gpu_buffer(device, (dim,), np.zeros(dim, dtype=np.float32)),
    )


def create_gpu_model_params(
    device: Device,
    vocab_size: int,
    embedding_dim: int,
    context_size: int,
    n_layers: int,
) -> GPUModelParams:
    """Initialize complete GPU model"""
    embedding_data = np.random.normal(0, 0.02, (vocab_size, embedding_dim)).astype(
        np.float32
    )
    embedding = create_gpu_buffer(device, (vocab_size, embedding_dim), embedding_data)

    pos_encoding_data = positional_encoding(context_size, embedding_dim)
    pos_encoding = create_gpu_buffer(
        device, (context_size, embedding_dim), pos_encoding_data
    )

    layers = [create_gpu_layer_params(device, embedding_dim) for _ in range(n_layers)]

    return GPUModelParams(embedding=embedding, pos_encoding=pos_encoding, layers=layers)


# ============================================================================
# OPTIMIZER STATE
# ============================================================================


def create_optimizer_state(model_params: GPUModelParams) -> GPUOptimizerState:
    """Initialize optimizer state (zero moments)"""
    device = model_params.embedding.device

    m_embedding = create_gpu_buffer(
        device,
        model_params.embedding.shape,
        np.zeros(model_params.embedding.shape, dtype=np.float32),
    )
    v_embedding = create_gpu_buffer(
        device,
        model_params.embedding.shape,
        np.zeros(model_params.embedding.shape, dtype=np.float32),
    )

    m_layers = []
    v_layers = []
    for layer in model_params.layers:
        dim = layer.attn_wq.shape[0]

        m_layer = create_gpu_layer_params(device, dim)
        v_layer = create_gpu_layer_params(device, dim)

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
            device.wgpu_device.queue.write_buffer(
                buf.buffer, 0, np.zeros(buf.size, dtype=np.float32)
            )
            buf = getattr(v_layer, attr)
            device.wgpu_device.queue.write_buffer(
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


def gpu_layer_to_dict(layer: GPULayerParams) -> Dict[str, np.ndarray]:
    """Convert GPU layer to dict"""
    from gpu_buffer import gpu_to_numpy

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


def dict_to_gpu_layer(
    device: Device, data: Dict[str, np.ndarray], embedding_dim: int
) -> GPULayerParams:
    """Create GPU layer from dict"""
    dim = embedding_dim

    return GPULayerParams(
        attn_wq=create_gpu_buffer(device, (dim, dim), data["attn_wq"]),
        attn_wk=create_gpu_buffer(device, (dim, dim), data["attn_wk"]),
        attn_wv=create_gpu_buffer(device, (dim, dim), data["attn_wv"]),
        attn_wo=create_gpu_buffer(device, (dim, dim), data["attn_wo"]),
        ff_w1=create_gpu_buffer(device, (dim, 4 * dim), data["ff_w1"]),
        ff_b1=create_gpu_buffer(device, (4 * dim,), data["ff_b1"]),
        ff_w2=create_gpu_buffer(device, (4 * dim, dim), data["ff_w2"]),
        ff_b2=create_gpu_buffer(device, (dim,), data["ff_b2"]),
        ln_gamma1=create_gpu_buffer(device, (dim,), data["ln_gamma1"]),
        ln_beta1=create_gpu_buffer(device, (dim,), data["ln_beta1"]),
        ln_gamma2=create_gpu_buffer(device, (dim,), data["ln_gamma2"]),
        ln_beta2=create_gpu_buffer(device, (dim,), data["ln_beta2"]),
    )
