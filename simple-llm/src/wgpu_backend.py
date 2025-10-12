"""
WGPU backend for transformer training and inference.

Refactored to work with the new GPU module API that uses:
- Type-safe buffers (GPUBuffer1D, GPUBuffer2D)
- Batch-based operations via BatchState
- Buffer pools and workspace management
- Separate forward/backward/optimizer pass modules
"""

import dataclasses
import json
import time
from typing import Optional, Tuple, Union

import numpy as np

from common.util import deserialize, get_model_file_names, serialize
from common_backend import TransformerModelParams
from gpu import gpu
from hyper import HyperParams


# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclasses.dataclass
class BackendState:
    """GPU backend infrastructure"""

    device: gpu.GPUDevice
    config: gpu.GPUConfig
    pipeline_cache: gpu.PipelineCache
    buffer_pool: gpu.BufferPool
    workspace_manager: gpu.WorkspaceManager


@dataclasses.dataclass
class TransformerModel:
    """Complete transformer model with GPU state"""

    tm_params: TransformerModelParams
    params: gpu.GPUModelParams
    opt_state: gpu.GPUOptimizerState
    learning_rate: float
    total_steps: int
    backend: BackendState


@dataclasses.dataclass
class ForwardCache:
    """Cache activations from forward pass for backward pass"""

    input_ids: gpu.GPUBuffer1D
    embedded: gpu.GPUBuffer2D
    layer_inputs: list
    attn_outputs: list
    mlp_hidden: list
    mlp_outputs: list
    qkv_projections: list
    attention_weights: list
    final_layernorm: gpu.GPUBuffer2D
    logits: gpu.GPUBuffer2D
    grad_logits: gpu.GPUBuffer2D = None


# ============================================================================
# BACKEND INITIALIZATION
# ============================================================================


def __create_backend_state() -> BackendState:
    """Initialize GPU backend infrastructure"""
    device = gpu.device_create()
    config = gpu.device_config_create(device)

    # Override Flash Attention parameters to reduce workgroup memory usage
    # Original defaults likely exceed 64KB limit
    # Memory usage = (Bc + Br) * head_dim * 4 bytes per workgroup
    # For head_dim=64: need Bc + Br < 256 to stay under 64KB
    config.flash_attn_bc = 32  # Block size for K/V (columns)
    config.flash_attn_br = 32  # Block size for Q (rows)
    config.flash_attn_max_head_dim = 64  # Maximum head dimension

    pipeline_cache = gpu.pipeline_cache_create(device)
    buffer_pool = gpu.pool_create(config)
    workspace_manager = gpu.workspace_manager_create(buffer_pool)

    return BackendState(
        device=device,
        config=config,
        pipeline_cache=pipeline_cache,
        buffer_pool=buffer_pool,
        workspace_manager=workspace_manager,
    )


def __create_sinusoidal_position_encoding(
    backend: BackendState, context_size: int, embedding_dim: int
) -> gpu.GPUBuffer2D:
    """
    Generate sinusoidal positional encoding (fixed, not trainable).
    This is called ONCE during model initialization.
    """
    position = np.arange(context_size)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim)
    )

    pos_encoding = np.zeros((context_size, embedding_dim))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    xs = pos_encoding.astype(np.float32)
    return gpu.create_gpu_buffer_2d(backend.device, context_size, embedding_dim, xs)


def __create_model_params_on_gpu(
    backend: BackendState,
    vocab_size: int,
    embedding_dim: int,
    n_heads: int,
    n_layers: int,
    context_size: int,
) -> gpu.GPUModelParams:
    """Create model parameters with random initialization"""
    device = backend.device

    embedding_table = gpu.create_gpu_buffer_2d(
        device,
        vocab_size,
        embedding_dim,
        np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.02,
    )

    layers = []
    for _ in range(n_layers):
        layer = gpu.GPULayerParams(
            attn_wq=gpu.create_gpu_buffer_2d(
                device,
                embedding_dim,
                embedding_dim,
                np.random.randn(embedding_dim, embedding_dim).astype(np.float32) * 0.02,
            ),
            attn_wk=gpu.create_gpu_buffer_2d(
                device,
                embedding_dim,
                embedding_dim,
                np.random.randn(embedding_dim, embedding_dim).astype(np.float32) * 0.02,
            ),
            attn_wv=gpu.create_gpu_buffer_2d(
                device,
                embedding_dim,
                embedding_dim,
                np.random.randn(embedding_dim, embedding_dim).astype(np.float32) * 0.02,
            ),
            attn_wo=gpu.create_gpu_buffer_2d(
                device,
                embedding_dim,
                embedding_dim,
                np.random.randn(embedding_dim, embedding_dim).astype(np.float32) * 0.02,
            ),
            ff_w1=gpu.create_gpu_buffer_2d(
                device,
                embedding_dim,
                4 * embedding_dim,
                np.random.randn(embedding_dim, 4 * embedding_dim).astype(np.float32)
                * 0.02,
            ),
            ff_w2=gpu.create_gpu_buffer_2d(
                device,
                4 * embedding_dim,
                embedding_dim,
                np.random.randn(4 * embedding_dim, embedding_dim).astype(np.float32)
                * 0.02,
            ),
            ff_b1=gpu.create_gpu_buffer_1d(
                device, 4 * embedding_dim, np.zeros(4 * embedding_dim, dtype=np.float32)
            ),
            ff_b2=gpu.create_gpu_buffer_1d(
                device, embedding_dim, np.zeros(embedding_dim, dtype=np.float32)
            ),
            ln_gamma1=gpu.create_gpu_buffer_1d(
                device, embedding_dim, np.ones(embedding_dim, dtype=np.float32)
            ),
            ln_beta1=gpu.create_gpu_buffer_1d(
                device, embedding_dim, np.zeros(embedding_dim, dtype=np.float32)
            ),
            ln_gamma2=gpu.create_gpu_buffer_1d(
                device, embedding_dim, np.ones(embedding_dim, dtype=np.float32)
            ),
            ln_beta2=gpu.create_gpu_buffer_1d(
                device, embedding_dim, np.zeros(embedding_dim, dtype=np.float32)
            ),
        )
        layers.append(layer)

    final_ln_gamma = gpu.create_gpu_buffer_1d(
        device, embedding_dim, np.ones(embedding_dim, dtype=np.float32)
    )
    final_ln_beta = gpu.create_gpu_buffer_1d(
        device, embedding_dim, np.zeros(embedding_dim, dtype=np.float32)
    )

    # Create output projection with TRANSPOSED dimensions for matmul
    # We want: [batch*seq, embedding_dim] @ [embedding_dim, vocab_size] = [batch*seq, vocab_size]
    output_projection = gpu.create_gpu_buffer_2d(
        device,
        embedding_dim,  # rows
        vocab_size,  # cols
        np.random.randn(embedding_dim, vocab_size).astype(np.float32) * 0.02,
    )

    return gpu.GPUModelParams(
        embedding=embedding_table,
        layers=layers,
        pos_encoding=__create_sinusoidal_position_encoding(
            backend, context_size, embedding_dim
        ),
        final_ln_gamma=final_ln_gamma,
        final_ln_beta=final_ln_beta,
        output_projection=output_projection,
    )


def __create_optimizer_state(
    backend: BackendState, params: gpu.GPUModelParams
) -> gpu.GPUOptimizerState:
    """Create AdamW optimizer state"""
    device = backend.device

    def zero_2d(buf: gpu.GPUBuffer2D):
        rows, cols = buf.shape
        return gpu.create_gpu_buffer_2d(
            device, rows, cols, np.zeros((rows, cols), dtype=np.float32)
        )

    def zero_1d(buf: gpu.GPUBuffer1D):
        size = buf.shape[0]
        return gpu.create_gpu_buffer_1d(
            device, size, np.zeros(buf.size, dtype=np.float32)
        )

    # Create momentum and velocity for embedding
    m_embedding = zero_2d(params.embedding)
    v_embedding = zero_2d(params.embedding)

    # Create momentum and velocity for each layer
    # Use the same GPULayerParams structure for momentum and velocity
    m_layers = []
    v_layers = []
    for layer in params.layers:
        # Momentum layer params
        m_layer = gpu.GPULayerParams(
            attn_wq=zero_2d(layer.attn_wq),
            attn_wk=zero_2d(layer.attn_wk),
            attn_wv=zero_2d(layer.attn_wv),
            attn_wo=zero_2d(layer.attn_wo),
            ff_w1=zero_2d(layer.ff_w1),
            ff_b1=zero_1d(layer.ff_b1),
            ff_w2=zero_2d(layer.ff_w2),
            ff_b2=zero_1d(layer.ff_b2),
            ln_gamma1=zero_1d(layer.ln_gamma1),
            ln_beta1=zero_1d(layer.ln_beta1),
            ln_gamma2=zero_1d(layer.ln_gamma2),
            ln_beta2=zero_1d(layer.ln_beta2),
        )
        m_layers.append(m_layer)

        # Velocity layer params
        v_layer = gpu.GPULayerParams(
            attn_wq=zero_2d(layer.attn_wq),
            attn_wk=zero_2d(layer.attn_wk),
            attn_wv=zero_2d(layer.attn_wv),
            attn_wo=zero_2d(layer.attn_wo),
            ff_w1=zero_2d(layer.ff_w1),
            ff_b1=zero_1d(layer.ff_b1),
            ff_w2=zero_2d(layer.ff_w2),
            ff_b2=zero_1d(layer.ff_b2),
            ln_gamma1=zero_1d(layer.ln_gamma1),
            ln_beta1=zero_1d(layer.ln_beta1),
            ln_gamma2=zero_1d(layer.ln_gamma2),
            ln_beta2=zero_1d(layer.ln_beta2),
        )
        v_layers.append(v_layer)

    m_output_projection = zero_2d(params.output_projection)
    v_output_projection = zero_2d(params.output_projection)

    return gpu.GPUOptimizerState(
        m_embedding=m_embedding,
        v_embedding=v_embedding,
        m_layers=m_layers,
        v_layers=v_layers,
        m_output_projection=m_output_projection,
        v_output_projection=v_output_projection,
        step=0,
    )


# ============================================================================
# BUFFER TRANSFER HELPERS
# ============================================================================


def __upload_array_to_gpu_2d(
    backend: BackendState, array: np.ndarray
) -> gpu.GPUBuffer2D:
    """Upload 2D numpy array to GPU"""
    rows, cols = array.shape
    return gpu.create_gpu_buffer_2d(
        backend.device, rows, cols, array.astype(np.float32)
    )


def __upload_array_to_gpu_1d(
    backend: BackendState, array: np.ndarray
) -> gpu.GPUBuffer1D:
    """Upload 1D numpy array to GPU"""
    size = array.shape[0]
    return gpu.create_gpu_buffer_1d(backend.device, size, array.astype(np.float32))


def __download_buffer_from_gpu_2d(
    backend: BackendState, buffer: gpu.GPUBuffer2D
) -> np.ndarray:
    """Download 2D GPU buffer to numpy"""
    rows, cols = buffer.shape
    staging = gpu.buffer_pool_acquire_staging(backend.buffer_pool, rows * cols * 4)
    batch = gpu.create_command_batch(backend.device)
    gpu.copy_buffer_to_staging(batch, buffer, staging)
    gpu.submit_batch(backend.device, batch)
    data = gpu.read_staging_buffer(staging, rows * cols)
    gpu.buffer_pool_release(backend.buffer_pool, staging)
    return data.reshape(rows, cols)


def _download_buffer_from_gpu_1d(
    backend: BackendState, buffer: gpu.GPUBuffer1D
) -> np.ndarray:
    """Download 1D GPU buffer to numpy"""
    size = buffer.shape[0]
    staging = gpu.buffer_pool_acquire_staging(backend.buffer_pool, size * 4)
    batch = gpu.create_command_batch(backend.device)
    gpu.copy_buffer_to_staging(batch, buffer, staging)
    gpu.submit_batch(backend.device, batch)
    data = gpu.read_staging_buffer(staging, size)
    gpu.buffer_pool_release(backend.buffer_pool, staging)
    return data


# ============================================================================
# PUBLIC API
# ============================================================================


def initialize_model(
    hp: HyperParams,
    epochs: int,
    total_steps: int,
) -> TransformerModel:
    """Create new model with random initialization"""
    print(hp)
    backend = __create_backend_state()
    params = __create_model_params_on_gpu(
        backend,
        hp.vocab_size,
        hp.embedding_dim,
        hp.n_heads,
        hp.n_layers,
        hp.context_size,
    )
    opt_state = __create_optimizer_state(backend, params)

    tm_params = TransformerModelParams(
        vocab_size=hp.vocab_size,
        embedding_dim=hp.embedding_dim,
        context_size=hp.context_size,
        n_heads=hp.n_heads,
        n_layers=hp.n_layers,
        epochs=[],
    )

    return TransformerModel(
        tm_params=tm_params,
        params=params,
        opt_state=opt_state,
        learning_rate=hp.learning_rate,
        total_steps=0,
        backend=backend,
    )


def load_model(hp: HyperParams, total_steps: int, model_path: str) -> TransformerModel:
    """Load model from checkpoint"""
    backend = __create_backend_state()
    model_weights_file, model_opt_file, model_config_file = get_model_file_names(
        model_path
    )

    with open(model_config_file, "r") as f:
        x = json.load(f)
        tm_params = TransformerModelParams(**x)

    q = dataclasses.asdict(tm_params)
    q["learning_rate"] = hp.learning_rate
    q["batch_size"] = hp.batch_size
    epochs = q.pop("epochs")
    hp0 = HyperParams(**q)
    dummy_model = initialize_model(hp0, epochs, total_steps)

    weights_data = deserialize(dummy_model, model_weights_file)

    # Upload weights (simplified - full implementation would handle all layers)
    embedding = __upload_array_to_gpu_2d(backend, weights_data["embedding"])
    layers = []
    for i in range(hp.n_layers):
        ld = weights_data["layers"][i]
        layers.append(
            gpu.GPULayerParams(
                attn_wq=__upload_array_to_gpu_2d(backend, ld["attn_wq"]),
                attn_wk=__upload_array_to_gpu_2d(backend, ld["attn_wk"]),
                attn_wv=__upload_array_to_gpu_2d(backend, ld["attn_wv"]),
                attn_wo=__upload_array_to_gpu_2d(backend, ld["attn_wo"]),
                ff_w1=__upload_array_to_gpu_2d(backend, ld["ff_w1"]),
                ff_w2=__upload_array_to_gpu_2d(backend, ld["ff_w2"]),
                ff_b1=__upload_array_to_gpu_1d(backend, ld["ff_b1"]),
                ff_b2=__upload_array_to_gpu_1d(backend, ld["ff_b2"]),
                ln_gamma1=__upload_array_to_gpu_1d(backend, ld["ln_gamma1"]),
                ln_beta1=__upload_array_to_gpu_1d(backend, ld["ln_beta1"]),
                ln_gamma2=__upload_array_to_gpu_1d(backend, ld["ln_gamma2"]),
                ln_beta2=__upload_array_to_gpu_1d(backend, ld["ln_beta2"]),
            )
        )

    params = gpu.GPUModelParams(
        embedding=embedding,
        layers=layers,
        pos_encoding=__create_sinusoidal_position_encoding(
            backend, hp.context_size, hp.embedding_dim
        ),
        final_ln_gamma=__upload_array_to_gpu_1d(
            backend, weights_data["final_ln_gamma"]
        ),
        final_ln_beta=__upload_array_to_gpu_1d(backend, weights_data["final_ln_beta"]),
        output_projection=__upload_array_to_gpu_2d(
            backend, weights_data["output_projection"]
        ),
    )

    opt_state = __create_optimizer_state(backend, params)

    return TransformerModel(
        tm_params=tm_params,
        params=params,
        opt_state=opt_state,
        learning_rate=hp.learning_rate,
        total_steps=total_steps,
        backend=backend,
    )


def save_model(model: TransformerModel, base_path: str) -> None:
    """Save model to checkpoint"""
    weights_data = {
        "embedding": __download_buffer_from_gpu_2d(
            model.backend, model.params.embedding
        ),
        "layers": [],
        "final_ln_gamma": _download_buffer_from_gpu_1d(
            model.backend, model.params.final_ln_gamma
        ),
        "final_ln_beta": _download_buffer_from_gpu_1d(
            model.backend, model.params.final_ln_beta
        ),
        "output_projection": __download_buffer_from_gpu_2d(
            model.backend, model.params.output_projection
        ),
    }

    for layer in model.params.layers:
        weights_data["layers"].append(
            {
                # FIXED: Use correct field names with attn_ and ff_ prefixes
                "attn_wq": __download_buffer_from_gpu_2d(model.backend, layer.attn_wq),
                "attn_wk": __download_buffer_from_gpu_2d(model.backend, layer.attn_wk),
                "attn_wv": __download_buffer_from_gpu_2d(model.backend, layer.attn_wv),
                "attn_wo": __download_buffer_from_gpu_2d(model.backend, layer.attn_wo),
                "ff_w1": __download_buffer_from_gpu_2d(model.backend, layer.ff_w1),
                "ff_w2": __download_buffer_from_gpu_2d(model.backend, layer.ff_w2),
                "ff_b1": _download_buffer_from_gpu_1d(model.backend, layer.ff_b1),
                "ff_b2": _download_buffer_from_gpu_1d(model.backend, layer.ff_b2),
                "ln_gamma1": _download_buffer_from_gpu_1d(
                    model.backend, layer.ln_gamma1
                ),
                "ln_beta1": _download_buffer_from_gpu_1d(model.backend, layer.ln_beta1),
                "ln_gamma2": _download_buffer_from_gpu_1d(
                    model.backend, layer.ln_gamma2
                ),
                "ln_beta2": _download_buffer_from_gpu_1d(model.backend, layer.ln_beta2),
            }
        )

    metadata = dataclasses.asdict(model.tm_params)
    metadata.update(
        {"learning_rate": model.learning_rate, "total_steps": model.total_steps}
    )

    model_weights_file, model_opt_file, model_config_file = get_model_file_names(
        base_path
    )
    with open(model_config_file, "w") as f:
        json.dump(metadata, f, indent=2)
    serialize(weights_data, model_weights_file)


def forward(
    params: gpu.GPUModelParams,
    input_tokens: np.ndarray,
    n_layers: int,
    n_heads: int,
    head_dim: int,
) -> np.ndarray:
    """
    Forward pass for inference (stateless).

    Args:
        params: Model parameters (GPUModelParams)
        input_tokens: Input token IDs [batch, seq_len] as int32
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        head_dim: Dimension per attention head

    Returns:
        logits: [batch, seq_len, vocab_size] as float32
    """
    # Create temporary backend for this inference call (stateless)
    backend = __create_backend_state()

    # Run forward pass
    logits = __forward_transformer(
        backend=backend,
        params=params,
        input_tokens=input_tokens,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
    )

    return logits


def sample_token(
    logits: np.ndarray, temperature: float = 1.0, top_k: Optional[int] = None
) -> int:
    """Sample next token from logits"""
    logits = logits / temperature
    if top_k:
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = -float("inf")
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


def train_epoch(
    model: TransformerModel,
    token_data: list[int],
    batch_size: int,
    context_size: int,
    epoch_num: int,
) -> Tuple[TransformerModel, dict[str, Union[float, int]]]:
    """Train for one epoch and return updated model + metrics"""
    backend = model.backend

    # Extract model dimensions
    embedding_dim = model.tm_params.embedding_dim
    vocab_size = model.tm_params.vocab_size
    n_layers = model.tm_params.n_layers
    n_heads = model.tm_params.n_heads
    head_dim = embedding_dim // n_heads

    # Convert list to numpy array
    token_array = np.array(token_data, dtype=np.int32)

    # Calculate number of batches
    tokens_per_batch = batch_size * (context_size + 1)
    n_batches = len(token_array) // tokens_per_batch

    if n_batches == 0:
        print(
            f"Warning: Not enough tokens for even one batch. Need {tokens_per_batch}, have {len(token_array)}"
        )
        return model, {"steps": 0, "avg_loss": 0.0, "smooth_loss": 0.0}

    # Training state
    total_loss = 0.0
    smooth_loss = 0.0
    steps_completed = 0
    start_time = time.time()

    # Process batches
    for batch_idx in range(n_batches):
        # Extract batch from flat token list
        batch_start = batch_idx * tokens_per_batch
        batch_end = batch_start + tokens_per_batch
        batch_tokens = token_array[batch_start:batch_end]

        # Reshape to [batch_size, context_size + 1]
        batch_data = batch_tokens.reshape(batch_size, context_size + 1)

        # Split into inputs and targets, then FLATTEN to 1D
        input_ids = (
            batch_data[:, :-1].astype(np.int32).flatten()
        )  # [batch*context_size]
        target_ids = (
            batch_data[:, 1:].astype(np.int32).flatten()
        )  # [batch*context_size]

        # Upload to GPU as 1D buffers
        input_buffer = __upload_array_to_gpu_1d(backend, input_ids)
        target_buffer = __upload_array_to_gpu_1d(backend, target_ids)

        # Get or create workspace
        workspace = gpu.workspace_get_or_create(
            backend.device,
            backend.workspace_manager,
            model.params,
            batch_size,
            context_size,
        )

        # Create command batch for entire training step
        batch = gpu.create_command_batch(backend.device, backend.config)

        # Forward pass with cache
        loss_buffer, cache = __forward_pass_with_cache(
            backend=backend,
            batch=batch,
            params=model.params,
            input_ids=input_buffer,
            target_ids=target_buffer,
            workspace=workspace,
            batch_size=batch_size,
            seq_len=context_size,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
        )

        grads = __backward_pass(
            backend=backend,
            batch=batch,
            params=model.params,
            cache=cache,
            workspace=workspace,
            batch_size=batch_size,
            seq_len=context_size,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
        )

        __optimizer_step(
            backend=backend,
            batch=batch,
            params=model.params,
            grads=grads,
            opt_state=model.opt_state,
            learning_rate=model.learning_rate,
            step=model.total_steps,
        )

        # Submit batch and wait
        gpu.submit_batch(backend.device, batch)

        # Download loss values (now an array of per-token losses)
        loss_values = _download_buffer_from_gpu_1d(backend, loss_buffer)

        # Compute mean loss across all tokens
        loss_value = float(np.mean(loss_values))

        # Update metrics
        total_loss += float(loss_value)
        if smooth_loss == 0.0:
            smooth_loss = float(loss_value)
        else:
            smooth_loss = smooth_loss * 0.999 + float(loss_value) * 0.001

        steps_completed += 1
        model.total_steps += 1

        # Progress reporting
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - start_time
            tokens_per_sec = (
                (batch_size * context_size * steps_completed) / elapsed
                if elapsed > 0
                else 0
            )
            print(
                f"Epoch {epoch_num + 1}, Step {batch_idx + 1}/{n_batches}, "
                f"Loss: {loss_value:.4f}, Smooth: {smooth_loss:.4f}, "
                f"TPS: {tokens_per_sec:.0f}",
                end="\r",
            )

    print()  # New line after progress

    # Calculate final metrics
    avg_loss = total_loss / steps_completed if steps_completed > 0 else 0.0

    metrics = {
        "steps": steps_completed,
        "avg_loss": avg_loss,
        "smooth_loss": smooth_loss,
    }

    return model, metrics


# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def __forward_transformer(
    backend: BackendState,
    params: gpu.GPUModelParams,
    input_tokens: np.ndarray,
    n_layers: int,
    n_heads: int,
    head_dim: int,
) -> np.ndarray:
    """
    Forward pass for inference (no cache, no loss).

    Args:
        backend: GPU backend state
        params: Model parameters
        input_tokens: Input token IDs [batch, seq_len] as int32
        n_layers: Number of transformer layers to use
        n_heads: Number of attention heads
        head_dim: Dimension per attention head

    Returns:
        logits: [batch, seq_len, vocab_size] as float32 numpy array
    """
    device = backend.device
    config = backend.config
    pipeline_cache = backend.pipeline_cache
    buffer_pool = backend.buffer_pool

    # Get dimensions
    batch_size, seq_len = input_tokens.shape
    embedding_dim = head_dim * n_heads
    vocab_size = params.output_projection.shape[1]  # [embedding_dim, vocab_size]

    # Flatten input to 1D for embedding lookup
    input_flat = input_tokens.flatten().astype(np.int32)

    # Upload input to GPU as 1D buffer
    input_ids = gpu.create_gpu_buffer_1d(device, batch_size * seq_len, input_flat)

    # Create command batch for entire forward pass
    batch = gpu.create_command_batch(device, config)

    # ========================================================================
    # 1. EMBEDDING + POSITIONAL ENCODING
    # ========================================================================
    embedded = gpu.pool_take_buffer_2d(
        device, buffer_pool, batch_size * seq_len, embedding_dim
    )

    gpu.embedding(
        device,
        config,
        pipeline_cache,
        batch,
        params.embedding,
        params.pos_encoding,
        input_ids,
        embedded,
        batch_size,
        seq_len,
    )

    layer_input = embedded

    # ========================================================================
    # 2. TRANSFORMER LAYERS
    # ========================================================================
    for layer_idx in range(n_layers):
        layer = params.layers[layer_idx]

        # Pre-attention LayerNorm
        ln1_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.layernorm(
            device,
            config,
            pipeline_cache,
            batch,
            layer_input,
            layer.ln_gamma1,
            layer.ln_beta1,
            ln1_out,
        )

        # Q, K, V projections
        q_proj = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        k_proj = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        v_proj = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )

        gpu.matmul(
            device, config, pipeline_cache, batch, ln1_out, layer.attn_wq, q_proj
        )
        gpu.matmul(
            device, config, pipeline_cache, batch, ln1_out, layer.attn_wk, k_proj
        )
        gpu.matmul(
            device, config, pipeline_cache, batch, ln1_out, layer.attn_wv, v_proj
        )

        # Flash Attention
        attn_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        L = gpu.pool_take_buffer_1d(device, buffer_pool, batch_size * seq_len * n_heads)
        M = gpu.pool_take_buffer_1d(device, buffer_pool, batch_size * seq_len * n_heads)

        gpu.flash_attention(
            device,
            config,
            pipeline_cache,
            batch,
            q_proj,
            k_proj,
            v_proj,
            attn_out,
            L,
            M,
            batch_size,
            seq_len,
            n_heads,
            head_dim,
        )

        # Attention output projection
        attn_proj = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.matmul(
            device, config, pipeline_cache, batch, attn_out, layer.attn_wo, attn_proj
        )

        # Residual connection 1
        post_attn = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.residual_add(
            device, config, pipeline_cache, batch, layer_input, attn_proj, post_attn
        )

        # Pre-MLP LayerNorm
        ln2_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.layernorm(
            device,
            config,
            pipeline_cache,
            batch,
            post_attn,
            layer.ln_gamma2,
            layer.ln_beta2,
            ln2_out,
        )

        # MLP: First linear + bias + GELU
        mlp_w1_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, 4 * embedding_dim
        )
        gpu.matmul(
            device, config, pipeline_cache, batch, ln2_out, layer.ff_w1, mlp_w1_out
        )

        mlp_with_bias = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, 4 * embedding_dim
        )
        gpu.bias_add(
            device,
            config,
            pipeline_cache,
            batch,
            mlp_w1_out,
            layer.ff_b1,
            mlp_with_bias,
        )

        mlp_activated = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, 4 * embedding_dim
        )
        gpu.gelu(device, config, pipeline_cache, batch, mlp_with_bias, mlp_activated)

        # MLP: Second linear + bias
        mlp_w2_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.matmul(
            device,
            config,
            pipeline_cache,
            batch,
            mlp_activated,
            layer.ff_w2,
            mlp_w2_out,
        )

        mlp_final = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.bias_add(
            device, config, pipeline_cache, batch, mlp_w2_out, layer.ff_b2, mlp_final
        )

        # Residual connection 2
        layer_output = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.residual_add(
            device, config, pipeline_cache, batch, post_attn, mlp_final, layer_output
        )

        # Output becomes input for next layer
        layer_input = layer_output

    # ========================================================================
    # 3. FINAL LAYER NORM
    # ========================================================================
    final_ln = gpu.pool_take_buffer_2d(
        device, buffer_pool, batch_size * seq_len, embedding_dim
    )
    gpu.layernorm(
        device,
        config,
        pipeline_cache,
        batch,
        layer_input,
        params.final_ln_gamma,
        params.final_ln_beta,
        final_ln,
    )

    # ========================================================================
    # 4. OUTPUT PROJECTION
    # ========================================================================
    logits = gpu.pool_take_buffer_2d(
        device, buffer_pool, batch_size * seq_len, vocab_size
    )
    gpu.matmul(
        device,
        config,
        pipeline_cache,
        batch,
        final_ln,
        params.output_projection,
        logits,
    )

    # ========================================================================
    # 5. SUBMIT AND DOWNLOAD
    # ========================================================================
    gpu.submit_batch(device, batch)

    # Download logits from GPU
    logits_np = __download_buffer_from_gpu_2d(backend, logits)

    # Reshape from [batch*seq, vocab_size] to [batch, seq, vocab_size]
    logits_reshaped = logits_np.reshape(batch_size, seq_len, vocab_size)

    return logits_reshaped


def __forward_pass_with_cache(
    backend: BackendState,
    batch: gpu.BatchState,
    params: gpu.GPUModelParams,
    input_ids: gpu.GPUBuffer1D,  # CHANGED: Now 1D
    target_ids: gpu.GPUBuffer1D,  # CHANGED: Now 1D
    workspace: gpu.WorkspaceBuffers,
    batch_size: int,  # ADDED: Need explicit batch_size
    seq_len: int,  # ADDED: Need explicit seq_len
    n_layers: int,
    n_heads: int,
    head_dim: int,
    vocab_size: int,
) -> Tuple[gpu.GPUBuffer1D, ForwardCache]:
    """
    Forward pass that saves activations for backward pass.
    input_ids and target_ids are FLAT 1D buffers.
    """
    device = backend.device
    config = backend.config
    pipeline_cache = backend.pipeline_cache
    buffer_pool = backend.buffer_pool

    embedding_dim = head_dim * n_heads

    # Initialize forward cache
    cache = ForwardCache(
        input_ids=input_ids,
        embedded=None,
        layer_inputs=[],
        attn_outputs=[],
        mlp_hidden=[],
        mlp_outputs=[],
        qkv_projections=[],
        attention_weights=[],
        final_layernorm=None,
        logits=None,
    )

    # ========================================================================
    # 1. EMBEDDING + POSITIONAL ENCODING
    # ========================================================================
    embedded = gpu.pool_take_buffer_2d(
        device, buffer_pool, batch_size * seq_len, embedding_dim
    )

    # Embedding lookup (input_ids is 1D flat buffer)
    gpu.embedding(
        device,
        config,
        pipeline_cache,
        batch,
        params.embedding,  # GPUBuffer2D [vocab_size, embedding_dim]
        params.pos_encoding,  # GPUBuffer2D [context_size, embedding_dim]
        input_ids,  # GPUBuffer1D [batch_size * seq_len]
        embedded,  # GPUBuffer2D [batch_size * seq_len, embedding_dim]
        batch_size,
        seq_len,
    )

    cache.embedded = embedded
    layer_input = embedded
    cache.layer_inputs.append(layer_input)

    # ========================================================================
    # 2. TRANSFORMER LAYERS
    # ========================================================================
    for layer_idx, layer in enumerate(params.layers[:n_layers]):
        # Pre-attention LayerNorm
        ln1_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.layernorm(
            device,
            config,
            pipeline_cache,
            batch,
            layer_input,
            layer.ln_gamma1,
            layer.ln_beta1,
            ln1_out,
        )

        # Q, K, V projections
        q_proj = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        k_proj = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        v_proj = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )

        gpu.matmul(
            device, config, pipeline_cache, batch, ln1_out, layer.attn_wq, q_proj
        )
        gpu.matmul(
            device, config, pipeline_cache, batch, ln1_out, layer.attn_wk, k_proj
        )
        gpu.matmul(
            device, config, pipeline_cache, batch, ln1_out, layer.attn_wv, v_proj
        )

        cache.qkv_projections.append((q_proj, k_proj, v_proj))

        # Flash Attention
        attn_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        L = gpu.pool_take_buffer_1d(device, buffer_pool, batch_size * seq_len * n_heads)
        M = gpu.pool_take_buffer_1d(device, buffer_pool, batch_size * seq_len * n_heads)

        gpu.flash_attention(
            device,
            config,
            pipeline_cache,
            batch,
            q_proj,
            k_proj,
            v_proj,
            attn_out,
            L,
            M,
            batch_size,
            seq_len,
            n_heads,
            head_dim,
        )

        cache.attn_outputs.append(attn_out)
        cache.attention_weights.append((L, M))

        # Attention output projection
        attn_proj = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.matmul(
            device, config, pipeline_cache, batch, attn_out, layer.attn_wo, attn_proj
        )

        # Residual connection 1
        post_attn = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.residual_add(
            device, config, pipeline_cache, batch, layer_input, attn_proj, post_attn
        )

        # Pre-MLP LayerNorm
        ln2_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.layernorm(
            device,
            config,
            pipeline_cache,
            batch,
            post_attn,
            layer.ln_gamma2,
            layer.ln_beta2,
            ln2_out,
        )

        # --------------------------------------------------------------------
        # MLP: First linear
        # --------------------------------------------------------------------
        # Replace entire MLP section with this VERY EXPLICIT version:

        # MLP Layer 1: [batch*seq, emb_dim] @ [emb_dim, 4*emb_dim] -> [batch*seq, 4*emb_dim]
        mlp_w1_out = gpu.pool_take_buffer_2d(
            device,
            buffer_pool,
            batch_size * seq_len,  # rows
            4 * embedding_dim,  # cols
        )
        gpu.matmul(
            device, config, pipeline_cache, batch, ln2_out, layer.ff_w1, mlp_w1_out
        )

        # Add bias: [batch*seq, 4*emb_dim] + [4*emb_dim] -> [batch*seq, 4*emb_dim]
        mlp_with_bias = gpu.pool_take_buffer_2d(
            device,
            buffer_pool,
            batch_size * seq_len,  # rows
            4 * embedding_dim,  # cols
        )
        gpu.bias_add(
            device,
            config,
            pipeline_cache,
            batch,
            mlp_w1_out,
            layer.ff_b1,
            mlp_with_bias,
        )

        # GELU: [batch*seq, 4*emb_dim] -> [batch*seq, 4*emb_dim]
        mlp_activated = gpu.pool_take_buffer_2d(
            device,
            buffer_pool,
            batch_size * seq_len,  # rows
            4 * embedding_dim,  # cols
        )
        gpu.gelu(device, config, pipeline_cache, batch, mlp_with_bias, mlp_activated)

        cache.mlp_hidden.append(mlp_activated)

        # MLP Layer 2: [batch*seq, 4*emb_dim] @ [4*emb_dim, emb_dim] -> [batch*seq, emb_dim]
        mlp_w2_out = gpu.pool_take_buffer_2d(
            device,
            buffer_pool,
            batch_size * seq_len,  # rows
            embedding_dim,  # cols
        )
        gpu.matmul(
            device,
            config,
            pipeline_cache,
            batch,
            mlp_activated,
            layer.ff_w2,
            mlp_w2_out,
        )

        # Add bias: [batch*seq, emb_dim] + [emb_dim] -> [batch*seq, emb_dim]
        mlp_final = gpu.pool_take_buffer_2d(
            device,
            buffer_pool,
            batch_size * seq_len,  # rows
            embedding_dim,  # cols
        )
        gpu.bias_add(
            device, config, pipeline_cache, batch, mlp_w2_out, layer.ff_b2, mlp_final
        )

        cache.mlp_outputs.append(mlp_final)

        # Residual connection 2
        layer_output = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.residual_add(
            device, config, pipeline_cache, batch, post_attn, mlp_final, layer_output
        )

        layer_input = layer_output
        cache.layer_inputs.append(layer_input)

    # ========================================================================
    # 3. FINAL LAYER NORM
    # ========================================================================
    final_ln = gpu.pool_take_buffer_2d(
        device, buffer_pool, batch_size * seq_len, embedding_dim
    )
    gpu.layernorm(
        device,
        config,
        pipeline_cache,
        batch,
        layer_input,
        params.final_ln_gamma,
        params.final_ln_beta,
        final_ln,
    )
    cache.final_layernorm = final_ln

    # ========================================================================
    # 4. OUTPUT PROJECTION
    # ========================================================================
    logits = gpu.pool_take_buffer_2d(
        device, buffer_pool, batch_size * seq_len, vocab_size
    )

    gpu.matmul(
        device,
        config,
        pipeline_cache,
        batch,
        final_ln,
        params.output_projection,
        logits,
    )
    cache.logits = logits

    # ========================================================================
    # 5. CROSS-ENTROPY LOSS
    # ========================================================================

    # Loss buffer needs one value per token, not just one scalar
    loss_buffer = gpu.pool_take_buffer_1d(
        device, buffer_pool, batch_size * seq_len
    )  # Changed from 1
    grad_logits = gpu.pool_take_buffer_2d(
        device, buffer_pool, batch_size * seq_len, vocab_size
    )

    gpu.cross_entropy_loss(
        device,
        config,
        pipeline_cache,
        batch,
        logits,
        target_ids,
        loss_buffer,
        grad_logits,
    )

    cache.grad_logits = grad_logits

    return loss_buffer, cache


def __backward_pass(
    backend: BackendState,
    batch: gpu.BatchState,
    params: gpu.GPUModelParams,
    cache: ForwardCache,
    workspace: gpu.WorkspaceBuffers,
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
) -> gpu.GPUModelGradients:
    """
    Complete backward pass through transformer.
    Computes gradients for all parameters using cached activations.

    Gradients are accumulated directly into params buffers (in-place).
    This assumes params have been initialized with zero gradients before this call.

    Args:
        backend: GPU backend state
        batch: Command batch to add operations to
        params: Model parameters (will store gradients in-place)
        cache: Cached activations from forward pass
        workspace: Workspace buffers for intermediate computations
        batch_size: Batch size
        seq_len: Sequence length
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        head_dim: Dimension per attention head

    Returns:
        GPUModelGradients: Complete gradients for all model parameters
    """
    device = backend.device
    config = backend.config
    pipeline_cache = backend.pipeline_cache
    buffer_pool = backend.buffer_pool

    embedding_dim = head_dim * n_heads
    vocab_size = params.output_projection.shape[1]

    # ========================================================================
    # ALLOCATE ALL GRADIENT BUFFERS
    # ========================================================================
    grad_embedding = gpu.pool_take_buffer_2d(
        device, buffer_pool, params.embedding.shape[0], params.embedding.shape[1]
    )

    grad_layers = []
    for _ in range(n_layers):
        grad_layer = gpu.GPULayerParams(
            attn_wq=gpu.pool_take_buffer_2d(
                device, buffer_pool, embedding_dim, embedding_dim
            ),
            attn_wk=gpu.pool_take_buffer_2d(
                device, buffer_pool, embedding_dim, embedding_dim
            ),
            attn_wv=gpu.pool_take_buffer_2d(
                device, buffer_pool, embedding_dim, embedding_dim
            ),
            attn_wo=gpu.pool_take_buffer_2d(
                device, buffer_pool, embedding_dim, embedding_dim
            ),
            ff_w1=gpu.pool_take_buffer_2d(
                device, buffer_pool, embedding_dim, 4 * embedding_dim
            ),
            ff_w2=gpu.pool_take_buffer_2d(
                device, buffer_pool, 4 * embedding_dim, embedding_dim
            ),
            ff_b1=gpu.pool_take_buffer_1d(device, buffer_pool, 4 * embedding_dim),
            ff_b2=gpu.pool_take_buffer_1d(device, buffer_pool, embedding_dim),
            ln_gamma1=gpu.pool_take_buffer_1d(device, buffer_pool, embedding_dim),
            ln_beta1=gpu.pool_take_buffer_1d(device, buffer_pool, embedding_dim),
            ln_gamma2=gpu.pool_take_buffer_1d(device, buffer_pool, embedding_dim),
            ln_beta2=gpu.pool_take_buffer_1d(device, buffer_pool, embedding_dim),
        )
        grad_layers.append(grad_layer)

    grad_final_ln_gamma = gpu.pool_take_buffer_1d(device, buffer_pool, embedding_dim)
    grad_final_ln_beta = gpu.pool_take_buffer_1d(device, buffer_pool, embedding_dim)
    grad_output_projection = gpu.pool_take_buffer_2d(
        device, buffer_pool, embedding_dim, vocab_size
    )

    # ========================================================================
    # BACKWARD: OUTPUT PROJECTION
    # logits = final_ln @ output_projection
    # ========================================================================
    grad_logits = cache.grad_logits  # From cross_entropy loss

    gpu.matmul_backward_b(
        device,
        config,
        pipeline_cache,
        batch,
        cache.final_layernorm,  # A: [batch*seq, embedding_dim]
        grad_logits,  # grad_C: [batch*seq, vocab_size]
        grad_output_projection,  # grad_B: [embedding_dim, vocab_size]
    )

    grad_final_ln = gpu.pool_take_buffer_2d(
        device, buffer_pool, batch_size * seq_len, embedding_dim
    )
    gpu.matmul_backward_a(
        device,
        config,
        pipeline_cache,
        batch,
        params.output_projection,  # B: [embedding_dim, vocab_size]
        grad_logits,  # grad_C: [batch*seq, vocab_size]
        grad_final_ln,  # grad_A: [batch*seq, embedding_dim]
    )

    # ========================================================================
    # BACKWARD: FINAL LAYER NORM
    # final_ln = layernorm(layer_inputs[-1], final_ln_gamma, final_ln_beta)
    # ========================================================================
    grad_before_final_ln = gpu.pool_take_buffer_2d(
        device, buffer_pool, batch_size * seq_len, embedding_dim
    )

    gpu.layernorm_backward(
        device,
        config,
        pipeline_cache,
        batch,
        cache.layer_inputs[-1],  # input to layernorm
        params.final_ln_gamma,  # gamma parameter
        grad_final_ln,  # gradient from above
        grad_before_final_ln,  # gradient w.r.t. input
        grad_final_ln_gamma,  # gradient w.r.t. gamma
        grad_final_ln_beta,  # gradient w.r.t. beta
    )

    grad_layer_output = grad_before_final_ln

    # ========================================================================
    # BACKWARD: TRANSFORMER LAYERS (in reverse order)
    # ========================================================================
    for layer_idx in range(n_layers - 1, -1, -1):
        layer = params.layers[layer_idx]
        grad_layer = grad_layers[layer_idx]

        # Retrieve cached forward activations
        layer_input = cache.layer_inputs[layer_idx]
        q_proj, k_proj, v_proj = cache.qkv_projections[layer_idx]
        attn_out = cache.attn_outputs[layer_idx]
        mlp_hidden = cache.mlp_hidden[layer_idx]
        mlp_out = cache.mlp_outputs[layer_idx]
        L, M = cache.attention_weights[layer_idx]

        # ====================================================================
        # RESIDUAL CONNECTION 2: layer_output = post_attn + mlp_out
        # Gradient splits to both branches
        # ====================================================================
        grad_post_attn = grad_layer_output
        grad_mlp_out = grad_layer_output

        # ====================================================================
        # MLP BACKWARD
        # ====================================================================

        # Backward: bias_add (mlp_out = mlp_w2_out + ff_b2)
        grad_mlp_w2_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.bias_add_backward(
            device,
            config,
            pipeline_cache,
            batch,
            grad_mlp_out,  # gradient from above
            grad_mlp_w2_out,  # gradient w.r.t. linear output
            grad_layer.ff_b2,  # gradient w.r.t. bias
        )

        # Backward: matmul (mlp_w2_out = mlp_hidden @ ff_w2)
        grad_mlp_hidden_from_w2 = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, 4 * embedding_dim
        )
        gpu.matmul_backward_b(
            device,
            config,
            pipeline_cache,
            batch,
            mlp_hidden,  # A: [batch*seq, 4*embedding_dim]
            grad_mlp_w2_out,  # grad_C
            grad_layer.ff_w2,  # grad_B: [4*embedding_dim, embedding_dim]
        )
        gpu.matmul_backward_a(
            device,
            config,
            pipeline_cache,
            batch,
            layer.ff_w2,  # B
            grad_mlp_w2_out,  # grad_C
            grad_mlp_hidden_from_w2,  # grad_A
        )

        # Backward: GELU
        # Note: mlp_hidden is the output AFTER GELU in cache
        # We need the input BEFORE GELU for the backward pass
        # Cache stores post-GELU, so we need to recompute pre-GELU
        # WORKAROUND: Recompute mlp_with_bias
        mlp_w1_out_recompute = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, 4 * embedding_dim
        )
        ln2_out_recompute = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )

        # Recompute ln2_out
        gpu.layernorm(
            device,
            config,
            pipeline_cache,
            batch,
            layer_input,
            layer.ln_gamma2,
            layer.ln_beta2,
            ln2_out_recompute,
        )

        # Recompute mlp_w1_out
        gpu.matmul(
            device,
            config,
            pipeline_cache,
            batch,
            ln2_out_recompute,
            layer.ff_w1,
            mlp_w1_out_recompute,
        )

        # Recompute mlp_with_bias
        mlp_with_bias_recompute = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, 4 * embedding_dim
        )
        gpu.bias_add(
            device,
            config,
            pipeline_cache,
            batch,
            mlp_w1_out_recompute,
            layer.ff_b1,
            mlp_with_bias_recompute,
        )

        # Now backward through GELU
        grad_mlp_with_bias = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, 4 * embedding_dim
        )
        gpu.gelu_backward(
            device,
            config,
            pipeline_cache,
            batch,
            mlp_with_bias_recompute,  # input to GELU
            grad_mlp_hidden_from_w2,  # gradient from above
            grad_mlp_with_bias,  # gradient w.r.t. input
        )

        # Backward: bias_add (mlp_with_bias = mlp_w1_out + ff_b1)
        grad_mlp_w1_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, 4 * embedding_dim
        )
        gpu.bias_add_backward(
            device,
            config,
            pipeline_cache,
            batch,
            grad_mlp_with_bias,
            grad_mlp_w1_out,
            grad_layer.ff_b1,
        )

        # Backward: matmul (mlp_w1_out = ln2_out @ ff_w1)
        grad_ln2_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.matmul_backward_b(
            device,
            config,
            pipeline_cache,
            batch,
            ln2_out_recompute,  # A
            grad_mlp_w1_out,  # grad_C
            grad_layer.ff_w1,  # grad_B
        )
        gpu.matmul_backward_a(
            device,
            config,
            pipeline_cache,
            batch,
            layer.ff_w1,  # B
            grad_mlp_w1_out,  # grad_C
            grad_ln2_out,  # grad_A
        )

        # Backward: layernorm2 (ln2_out = layernorm(post_attn, ...))
        grad_post_attn_from_ln2 = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )

        # Need to recompute post_attn since it's not cached
        # post_attn = layer_input + attn_proj
        # Recompute attn_proj
        attn_proj_recompute = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.matmul(
            device,
            config,
            pipeline_cache,
            batch,
            attn_out,
            layer.attn_wo,
            attn_proj_recompute,
        )

        post_attn_recompute = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.residual_add(
            device,
            config,
            pipeline_cache,
            batch,
            layer_input,
            attn_proj_recompute,
            post_attn_recompute,
        )

        gpu.layernorm_backward(
            device,
            config,
            pipeline_cache,
            batch,
            post_attn_recompute,  # input
            layer.ln_gamma2,  # gamma
            grad_ln2_out,  # gradient from above
            grad_post_attn_from_ln2,  # gradient w.r.t. input
            grad_layer.ln_gamma2,  # gradient w.r.t. gamma
            grad_layer.ln_beta2,  # gradient w.r.t. beta
        )

        # Accumulate gradient from residual connection
        gpu.residual_add(
            device,
            config,
            pipeline_cache,
            batch,
            grad_post_attn,
            grad_post_attn_from_ln2,
            grad_post_attn,  # accumulate in-place
        )

        # ====================================================================
        # RESIDUAL CONNECTION 1: post_attn = layer_input + attn_proj
        # ====================================================================
        grad_layer_input_from_residual1 = grad_post_attn
        grad_attn_proj = grad_post_attn

        # ====================================================================
        # ATTENTION OUTPUT PROJECTION: attn_proj = attn_out @ attn_wo
        # ====================================================================
        grad_attn_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.matmul_backward_b(
            device,
            config,
            pipeline_cache,
            batch,
            attn_out,  # A
            grad_attn_proj,  # grad_C
            grad_layer.attn_wo,  # grad_B
        )
        gpu.matmul_backward_a(
            device,
            config,
            pipeline_cache,
            batch,
            layer.attn_wo,  # B
            grad_attn_proj,  # grad_C
            grad_attn_out,  # grad_A
        )

        # ====================================================================
        # FLASH ATTENTION BACKWARD
        # ====================================================================
        grad_q = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        grad_k = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        grad_v = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )

        gpu.flash_attention_backward(
            device,
            config,
            pipeline_cache,
            batch,
            q_proj,
            k_proj,
            v_proj,  # forward inputs
            grad_attn_out,  # gradient from above
            L,
            M,  # saved statistics from forward
            grad_q,
            grad_k,
            grad_v,  # gradients w.r.t. Q, K, V
            batch_size,
            seq_len,
            n_heads,
            head_dim,
        )

        # ====================================================================
        # QKV PROJECTIONS BACKWARD
        # q = ln1_out @ wq, k = ln1_out @ wk, v = ln1_out @ wv
        # ====================================================================

        # Need ln1_out - recompute it
        ln1_out_recompute = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.layernorm(
            device,
            config,
            pipeline_cache,
            batch,
            layer_input,
            layer.ln_gamma1,
            layer.ln_beta1,
            ln1_out_recompute,
        )

        # Gradient w.r.t. wq, wk, wv
        gpu.matmul_backward_b(
            device,
            config,
            pipeline_cache,
            batch,
            ln1_out_recompute,
            grad_q,
            grad_layer.attn_wq,
        )
        gpu.matmul_backward_b(
            device,
            config,
            pipeline_cache,
            batch,
            ln1_out_recompute,
            grad_k,
            grad_layer.attn_wk,
        )
        gpu.matmul_backward_b(
            device,
            config,
            pipeline_cache,
            batch,
            ln1_out_recompute,
            grad_v,
            grad_layer.attn_wv,
        )

        # Gradient w.r.t. ln1_out (accumulate from Q, K, V)
        grad_ln1_out_from_q = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        grad_ln1_out_from_k = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        grad_ln1_out_from_v = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )

        gpu.matmul_backward_a(
            device,
            config,
            pipeline_cache,
            batch,
            layer.attn_wq,
            grad_q,
            grad_ln1_out_from_q,
        )
        gpu.matmul_backward_a(
            device,
            config,
            pipeline_cache,
            batch,
            layer.attn_wk,
            grad_k,
            grad_ln1_out_from_k,
        )
        gpu.matmul_backward_a(
            device,
            config,
            pipeline_cache,
            batch,
            layer.attn_wv,
            grad_v,
            grad_ln1_out_from_v,
        )

        # Sum gradients from all three projections
        grad_ln1_out = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.residual_add(
            device,
            config,
            pipeline_cache,
            batch,
            grad_ln1_out_from_q,
            grad_ln1_out_from_k,
            grad_ln1_out,
        )
        gpu.residual_add(
            device,
            config,
            pipeline_cache,
            batch,
            grad_ln1_out,
            grad_ln1_out_from_v,
            grad_ln1_out,
        )

        # ====================================================================
        # LAYERNORM 1 BACKWARD: ln1_out = layernorm(layer_input, ...)
        # ====================================================================
        grad_layer_input_from_ln1 = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.layernorm_backward(
            device,
            config,
            pipeline_cache,
            batch,
            layer_input,  # input
            layer.ln_gamma1,  # gamma
            grad_ln1_out,  # gradient from above
            grad_layer_input_from_ln1,  # gradient w.r.t. input
            grad_layer.ln_gamma1,  # gradient w.r.t. gamma
            grad_layer.ln_beta1,  # gradient w.r.t. beta
        )

        # ====================================================================
        # ACCUMULATE GRADIENTS FOR LAYER INPUT
        # Comes from both residual connections
        # ====================================================================
        grad_layer_output = gpu.pool_take_buffer_2d(
            device, buffer_pool, batch_size * seq_len, embedding_dim
        )
        gpu.residual_add(
            device,
            config,
            pipeline_cache,
            batch,
            grad_layer_input_from_residual1,
            grad_layer_input_from_ln1,
            grad_layer_output,
        )

    # ========================================================================
    # BACKWARD: EMBEDDING
    # ========================================================================
    gpu.embedding_backward(
        device,
        config,
        pipeline_cache,
        batch,
        cache.input_ids,  # token indices
        grad_layer_output,  # gradient from first layer
        grad_embedding,  # gradient w.r.t. embedding table
        batch_size,
        seq_len,
    )

    # ========================================================================
    # RETURN COMPLETE GRADIENTS
    # ========================================================================
    return gpu.GPUModelParams(
        embedding=grad_embedding,
        layers=grad_layers,
        pos_encoding=None,  # Positional encoding is fixed (not trained)
        final_ln_gamma=grad_final_ln_gamma,
        final_ln_beta=grad_final_ln_beta,
        output_projection=grad_output_projection,
    )


def __optimizer_step(
    backend: BackendState,
    batch: gpu.BatchState,
    params: gpu.GPUModelParams,
    grads: gpu.GPUModelGradients,
    opt_state: gpu.GPUOptimizerState,
    learning_rate: float,
    step: int,
) -> None:
    """
    Apply AdamW optimizer updates to all parameters.

    AdamW update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)

    Args:
        backend: GPU backend state
        batch: Command batch to add operations to
        params: Model parameters (will be updated in-place)
        grads: Gradients for all parameters
        opt_state: Optimizer state (momentum and velocity)
        learning_rate: Learning rate
        step: Current training step (for bias correction)
    """
    device = backend.device
    config = backend.config
    pipeline_cache = backend.pipeline_cache

    # AdamW hyperparameters
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.01

    # Bias correction
    bias_correction1 = 1.0 - (beta1 ** (step + 1))
    bias_correction2 = 1.0 - (beta2 ** (step + 1))

    # ========================================================================
    # UPDATE EMBEDDING
    # ========================================================================
    gpu.adamw_update(
        device,
        config,
        pipeline_cache,
        batch,
        params.embedding,  # parameter
        grads.embedding,  # gradient
        opt_state.m_embedding,  # momentum
        opt_state.v_embedding,  # velocity
        learning_rate,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
    )

    # ========================================================================
    # UPDATE TRANSFORMER LAYERS
    # ========================================================================
    for layer_idx in range(len(params.layers)):
        layer = params.layers[layer_idx]
        grad_layer = grads.layers[layer_idx]
        m_layer = opt_state.m_layers[layer_idx]
        v_layer = opt_state.v_layers[layer_idx]

        # Update all attention weights
        gpu.adamw_update(
            device,
            config,
            pipeline_cache,
            batch,
            layer.attn_wq,
            grad_layer.attn_wq,
            m_layer.attn_wq,
            v_layer.attn_wq,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )
        gpu.adamw_update(
            device,
            config,
            pipeline_cache,
            batch,
            layer.attn_wk,
            grad_layer.attn_wk,
            m_layer.attn_wk,
            v_layer.attn_wk,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )
        gpu.adamw_update(
            device,
            config,
            pipeline_cache,
            batch,
            layer.attn_wv,
            grad_layer.attn_wv,
            m_layer.attn_wv,
            v_layer.attn_wv,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )
        gpu.adamw_update(
            device,
            config,
            pipeline_cache,
            batch,
            layer.attn_wo,
            grad_layer.attn_wo,
            m_layer.attn_wo,
            v_layer.attn_wo,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )

        # Update MLP weights
        gpu.adamw_update(
            device,
            config,
            pipeline_cache,
            batch,
            layer.ff_w1,
            grad_layer.ff_w1,
            m_layer.ff_w1,
            v_layer.ff_w1,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )
        gpu.adamw_update(
            device,
            config,
            pipeline_cache,
            batch,
            layer.ff_w2,
            grad_layer.ff_w2,
            m_layer.ff_w2,
            v_layer.ff_w2,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )
        gpu.adamw_update(
            device,
            config,
            pipeline_cache,
            batch,
            layer.ff_b1,
            grad_layer.ff_b1,
            m_layer.ff_b1,
            v_layer.ff_b1,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )
        gpu.adamw_update(
            device,
            config,
            pipeline_cache,
            batch,
            layer.ff_b2,
            grad_layer.ff_b2,
            m_layer.ff_b2,
            v_layer.ff_b2,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )

        # Update layer norm parameters
        gpu.adamw_update(
            device,
            config,
            pipeline_cache,
            batch,
            layer.ln_gamma1,
            grad_layer.ln_gamma1,
            m_layer.ln_gamma1,
            v_layer.ln_gamma1,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )
        gpu.adamw_update(
            device,
            config,
            pipeline_cache,
            batch,
            layer.ln_beta1,
            grad_layer.ln_beta1,
            m_layer.ln_beta1,
            v_layer.ln_beta1,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )
        gpu.adamw_update(
            device,
            config,
            pipeline_cache,
            batch,
            layer.ln_gamma2,
            grad_layer.ln_gamma2,
            m_layer.ln_gamma2,
            v_layer.ln_gamma2,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )
        gpu.adamw_update(
            device,
            config,
            pipeline_cache,
            batch,
            layer.ln_beta2,
            grad_layer.ln_beta2,
            m_layer.ln_beta2,
            v_layer.ln_beta2,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )

    # ========================================================================
    # UPDATE FINAL LAYER NORM
    # ========================================================================
    # Note: Don't use weight decay for layer norm parameters
    gpu.adamw_update(
        device,
        config,
        pipeline_cache,
        batch,
        params.final_ln_gamma,
        grads.final_ln_gamma,
        opt_state.m_final_ln_gamma,
        opt_state.v_final_ln_gamma,
        learning_rate,
        beta1,
        beta2,
        eps,
        0.0,  # No weight decay
        bias_correction1,
        bias_correction2,
    )
    gpu.adamw_update(
        device,
        config,
        pipeline_cache,
        batch,
        params.final_ln_beta,
        grads.final_ln_beta,
        opt_state.m_final_ln_beta,
        opt_state.v_final_ln_beta,
        learning_rate,
        beta1,
        beta2,
        eps,
        0.0,  # No weight decay
        bias_correction1,
        bias_correction2,
    )

    # Update output projection
    gpu.adamw_update(
        device,
        config,
        pipeline_cache,
        batch,
        params.output_projection,
        grads.output_projection,
        opt_state.m_output_projection,
        opt_state.v_output_projection,
        learning_rate,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
    )

    # ========================================================================
    # UPDATE OUTPUT PROJECTION
    # ========================================================================
    # Note: Since output_projection is tied to embedding, skip this
    # (already updated via embedding update)

    # Increment step counter
    opt_state.step += 1
