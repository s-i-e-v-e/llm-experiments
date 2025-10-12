"""Model parameter initialization"""

from typing import Dict

import numpy as np

from .gpu_buffer import clear_buffer, create_gpu_buffer_1d, create_gpu_buffer_2d
from .gpu_types import Device, GPULayerParams, GPUModelParams, GPUOptimizerState

# ============================================================================
# POSITIONAL ENCODING
# ============================================================================


def positional_encoding(seq_len: int, dim: int) -> np.ndarray:
    """Generate sinusoidal positional encoding.

    Uses sine for even indices and cosine for odd indices,
    following the Transformer paper (Vaswani et al. 2017).

    Args:
        seq_len: Maximum sequence length
        dim: Embedding dimension

    Returns:
        New numpy array of shape (seq_len, dim) with positional encodings

    Raises:
        ValueError: If seq_len or dim are <= 0
    """
    if seq_len <= 0 or dim <= 0:
        raise ValueError(f"Invalid seq_len={seq_len} or dim={dim}")

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


def create_gpu_layer_params(device: GPUDevice, embedding_dim: int) -> GPULayerParams:
    """Initialize GPU layer parameters with random weights.

    Creates attention and feedforward weights with normal initialization
    (std=0.02), and layer norm parameters (gamma=1, beta=0).

    Args:
        device: GPU device state
        embedding_dim: Model embedding dimension

    Returns:
        New GPULayerParams with initialized weights on GPU

    Raises:
        ValueError: If embedding_dim <= 0
    """
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

    dim = embedding_dim

    return GPULayerParams(
        attn_wq=create_gpu_buffer_2d(
            device, dim, dim, np.random.normal(0, 0.02, (dim, dim)).astype(np.float32)
        ),
        attn_wk=create_gpu_buffer_2d(
            device, dim, dim, np.random.normal(0, 0.02, (dim, dim)).astype(np.float32)
        ),
        attn_wv=create_gpu_buffer_2d(
            device, dim, dim, np.random.normal(0, 0.02, (dim, dim)).astype(np.float32)
        ),
        attn_wo=create_gpu_buffer_2d(
            device, dim, dim, np.random.normal(0, 0.02, (dim, dim)).astype(np.float32)
        ),
        ff_w1=create_gpu_buffer_2d(
            device,
            dim,
            4 * dim,
            np.random.normal(0, 0.02, (dim, 4 * dim)).astype(np.float32),
        ),
        ff_b1=create_gpu_buffer_1d(
            device, 4 * dim, np.zeros(4 * dim, dtype=np.float32)
        ),
        ff_w2=create_gpu_buffer_2d(
            device,
            4 * dim,
            dim,
            np.random.normal(0, 0.02, (4 * dim, dim)).astype(np.float32),
        ),
        ff_b2=create_gpu_buffer_1d(device, dim, np.zeros(dim, dtype=np.float32)),
        ln_gamma1=create_gpu_buffer_1d(device, dim, np.ones(dim, dtype=np.float32)),
        ln_beta1=create_gpu_buffer_1d(device, dim, np.zeros(dim, dtype=np.float32)),
        ln_gamma2=create_gpu_buffer_1d(device, dim, np.ones(dim, dtype=np.float32)),
        ln_beta2=create_gpu_buffer_1d(device, dim, np.zeros(dim, dtype=np.float32)),
    )


def create_gpu_model_params(
    device: GPUDevice,
    vocab_size: int,
    embedding_dim: int,
    context_size: int,
    n_layers: int,
) -> GPUModelParams:
    """Initialize complete GPU model with random weights.

    Args:
        device: GPU device state
        vocab_size: Vocabulary size
        embedding_dim: Model embedding dimension
        context_size: Maximum context length
        n_layers: Number of transformer layers

    Returns:
        New GPUModelParams with all weights initialized on GPU

    Raises:
        ValueError: If any dimensions are <= 0
    """
    if vocab_size <= 0 or embedding_dim <= 0 or context_size <= 0 or n_layers <= 0:
        raise ValueError(
            f"All dimensions must be positive: vocab_size={vocab_size}, "
            f"embedding_dim={embedding_dim}, context_size={context_size}, n_layers={n_layers}"
        )

    # Embedding table
    embedding_data = np.random.normal(0, 0.02, (vocab_size, embedding_dim)).astype(
        np.float32
    )
    embedding = create_gpu_buffer_2d(device, vocab_size, embedding_dim, embedding_data)

    # Positional encoding
    pos_encoding_data = positional_encoding(context_size, embedding_dim)
    pos_encoding = create_gpu_buffer_2d(
        device, context_size, embedding_dim, pos_encoding_data
    )

    # Layers
    layers = [create_gpu_layer_params(device, embedding_dim) for _ in range(n_layers)]

    return GPUModelParams(embedding=embedding, pos_encoding=pos_encoding, layers=layers)


# ============================================================================
# OPTIMIZER STATE
# ============================================================================


def create_optimizer_state(model_params: GPUModelParams) -> GPUOptimizerState:
    """Initialize optimizer state with zero moments.

    Creates momentum (m) and variance (v) buffers for AdamW,
    all initialized to zero.

    Args:
        model_params: Model parameters

    Returns:
        New GPUOptimizerState with zero-initialized moments
    """
    device = model_params.embedding.device

    # Embedding momentum and variance
    m_embedding = create_gpu_buffer_2d(
        device,
        *model_params.embedding.shape,
        np.zeros(model_params.embedding.shape, dtype=np.float32),
    )
    v_embedding = create_gpu_buffer_2d(
        device,
        *model_params.embedding.shape,
        np.zeros(model_params.embedding.shape, dtype=np.float32),
    )

    # Layer moments
    m_layers = []
    v_layers = []

    for layer in model_params.layers:
        dim = layer.attn_wq.shape[0]

        # Create layer params for momentum
        m_layer = create_gpu_layer_params(device, dim)
        v_layer = create_gpu_layer_params(device, dim)

        # Zero initialize all buffers using clear_buffer utility
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
            clear_buffer(device, getattr(m_layer, attr))
            clear_buffer(device, getattr(v_layer, attr))

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
    """Convert GPU layer to dictionary for serialization.

    Args:
        layer: GPU layer parameters

    Returns:
        Dictionary mapping parameter names to numpy arrays
    """
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
    """Create GPU layer from dictionary.

    Args:
        device: GPU device state
        data: Dictionary mapping parameter names to numpy arrays
        embedding_dim: Model embedding dimension

    Returns:
        New GPULayerParams with weights uploaded to GPU

    Raises:
        KeyError: If required keys missing from data
        ValueError: If embedding_dim doesn't match data
    """
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

    dim = embedding_dim

    return GPULayerParams(
        attn_wq=create_gpu_buffer_2d(device, dim, dim, data["attn_wq"]),
        attn_wk=create_gpu_buffer_2d(device, dim, dim, data["attn_wk"]),
        attn_wv=create_gpu_buffer_2d(device, dim, dim, data["attn_wv"]),
        attn_wo=create_gpu_buffer_2d(device, dim, dim, data["attn_wo"]),
        ff_w1=create_gpu_buffer_2d(device, dim, 4 * dim, data["ff_w1"]),
        ff_b1=create_gpu_buffer_1d(device, 4 * dim, data["ff_b1"]),
        ff_w2=create_gpu_buffer_2d(device, 4 * dim, dim, data["ff_w2"]),
        ff_b2=create_gpu_buffer_1d(device, dim, data["ff_b2"]),
        ln_gamma1=create_gpu_buffer_1d(device, dim, data["ln_gamma1"]),
        ln_beta1=create_gpu_buffer_1d(device, dim, data["ln_beta1"]),
        ln_gamma2=create_gpu_buffer_1d(device, dim, data["ln_gamma2"]),
        ln_beta2=create_gpu_buffer_1d(device, dim, data["ln_beta2"]),
    )
