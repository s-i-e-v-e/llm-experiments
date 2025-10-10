import dataclasses
import json
import random
import time

import numpy as np

import v2.gpu as gpu
from common.util import deserialize, get_model_file_names, serialize

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclasses.dataclass
class TransformerModelParams:
    vocab_size: int
    embedding_dim: int
    context_length: int
    n_heads: int
    n_layers: int
    epochs: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TransformerModel:
    tm_params: TransformerModelParams
    params: gpu.GPUModelParams
    opt_state: gpu.GPUOptimizerState
    learning_rate: float
    total_steps: int


# ============================================================================
# TRAINING OPERATIONS
# ============================================================================


def train_step(model: TransformerModel, batch_inputs, batch_targets):
    """Single training step - pure GPU"""

    new_params, new_opt_state, loss = gpu.train_step_pure_gpu(
        model.params,
        model.opt_state,
        batch_inputs,
        batch_targets,
        n_heads=model.tm_params.n_heads,
        lr=model.learning_rate,
    )

    return TransformerModel(
        tm_params=model.tm_params,
        params=new_params,
        opt_state=new_opt_state,
        learning_rate=model.learning_rate,
        total_steps=model.total_steps,
    ), loss


def train_epoch(
    model: TransformerModel,
    token_data: list,
    batch_size: int,
    context_length: int,
    epoch_num: int,
):
    """Train for one epoch - pure GPU"""

    # Create sequences
    sequences = []
    stride = batch_size
    for i in range(0, len(token_data) - context_length, stride):
        seq = token_data[i : i + context_length + 1]
        if len(seq) == context_length + 1:
            input_seq = seq[:-1]
            target_seq = seq[1:]
            sequences.append((input_seq, target_seq))

    random.shuffle(sequences)
    steps_in_epoch = max(1, len(sequences) // batch_size)

    if len(sequences) == 0:
        return model, {"avg_loss": 0.0, "smooth_loss": 0.0, "steps": 0}

    # Training state
    epoch_loss = 0.0
    step = 0
    smooth_loss = 0.0
    start_time = time.time()

    # Training loop
    for batch_start in range(0, len(sequences), batch_size):
        batch = sequences[batch_start : batch_start + batch_size]
        if len(batch) == 0:
            continue

        batch_inputs = np.array([seq[0] for seq in batch], dtype=np.int32)
        batch_targets = np.array([seq[1] for seq in batch], dtype=np.int32)

        # Single GPU training step
        model, loss = train_step(model, batch_inputs, batch_targets)

        # Update metrics
        epoch_loss += float(loss)
        smooth_loss = (
            smooth_loss * 0.999 + float(loss) * 0.001
            if smooth_loss > 0
            else float(loss)
        )
        step += 1

        # Progress reporting
        elapsed = time.time() - start_time
        tokens_per_sec = (
            (batch_size * context_length * step) / elapsed if elapsed > 0 else 0
        )

        print(
            f"\rEpoch {epoch_num + 1}, Step {step}/{steps_in_epoch} | "
            f"Loss: {float(loss):.4f} | Smooth: {smooth_loss:.4f} | "
            f"TPS: {tokens_per_sec:.0f}",
            end="",
        )

    print()

    avg_loss = epoch_loss / step if step > 0 else 0.0

    return model, {
        "avg_loss": avg_loss,
        "smooth_loss": smooth_loss,
        "steps": step,
    }


# ============================================================================
# MODEL LIFECYCLE
# ============================================================================


def initialize_model(
    vocab_size,
    embedding_dim,
    context_length,
    n_heads,
    n_layers,
    epochs,
    learning_rate,
    total_steps,
):
    """Initialize model on GPU"""

    # Set random seed for reproducibility
    np.random.seed(0)

    # Initialize GPU device
    print("Initializing WGPU device...")
    device = gpu.get_device()
    if device is None:
        raise RuntimeError("Failed to initialize WGPU device")

    # Create GPU model parameters
    print(
        f"Creating model: {n_layers} layers, {embedding_dim} dims, {vocab_size} vocab"
    )
    gpu_params = gpu.create_gpu_model_params(
        vocab_size, embedding_dim, context_length, n_layers
    )

    # Create optimizer state
    print("Initializing optimizer state...")
    opt_state = gpu.create_optimizer_state(gpu_params)

    print("✅ Model initialized on GPU")

    return TransformerModel(
        tm_params=TransformerModelParams(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_length=context_length,
            n_heads=n_heads,
            n_layers=n_layers,
            epochs=epochs,
        ),
        params=gpu_params,
        opt_state=opt_state,
        learning_rate=learning_rate,
        total_steps=total_steps,
    )


def save_model(model: TransformerModel, model_path: str):
    """Save model to disk"""
    print(f"Saving model to {model_path}...")
    xs = get_model_file_names(model_path)

    # Convert GPU params to numpy for serialization
    params_dict = {
        "embedding": gpu.gpu_to_numpy(model.params.embedding),
        "pos_encoding": gpu.gpu_to_numpy(model.params.pos_encoding),
        "layers": [gpu.gpu_layer_to_dict(layer) for layer in model.params.layers],
    }
    serialize(params_dict, xs[0])

    # Convert optimizer state
    opt_state_dict = {
        "m_embedding": gpu.gpu_to_numpy(model.opt_state.m_embedding),
        "v_embedding": gpu.gpu_to_numpy(model.opt_state.v_embedding),
        "m_layers": [
            gpu.gpu_layer_to_dict(layer) for layer in model.opt_state.m_layers
        ],
        "v_layers": [
            gpu.gpu_layer_to_dict(layer) for layer in model.opt_state.v_layers
        ],
        "step": model.opt_state.step,
    }
    serialize(opt_state_dict, xs[1])

    with open(xs[2], "w") as f:
        json.dump(dataclasses.asdict(model.tm_params), f, indent=2)

    print("✅ Model saved")


def load_model(learning_rate: float, total_steps: int, model_path: str):
    """Load model from disk"""
    print(f"Loading model from {model_path}...")
    xs = get_model_file_names(model_path)

    with open(xs[2], "r") as f:
        config_data = json.load(f)
        if "epochs" not in config_data:
            config_data["epochs"] = []
        tm_params = TransformerModelParams(**config_data)

    # Initialize GPU device
    device = gpu.get_device()
    if device is None:
        raise RuntimeError("Failed to initialize WGPU device")

    # Load params
    params_dict = deserialize(None, xs[0])

    gpu_params = gpu.GPUModelParams(
        embedding=gpu.create_gpu_buffer(
            params_dict["embedding"].shape, params_dict["embedding"], device
        ),
        pos_encoding=gpu.create_gpu_buffer(
            params_dict["pos_encoding"].shape, params_dict["pos_encoding"], device
        ),
        layers=[
            gpu.dict_to_gpu_layer(layer_dict, tm_params.embedding_dim, device)
            for layer_dict in params_dict["layers"]
        ],
    )

    # Load optimizer state
    opt_state_dict = deserialize(None, xs[1])

    opt_state = gpu.GPUOptimizerState(
        m_embedding=gpu.create_gpu_buffer(
            opt_state_dict["m_embedding"].shape, opt_state_dict["m_embedding"], device
        ),
        v_embedding=gpu.create_gpu_buffer(
            opt_state_dict["v_embedding"].shape, opt_state_dict["v_embedding"], device
        ),
        m_layers=[
            gpu.dict_to_gpu_layer(m_layer, tm_params.embedding_dim, device)
            for m_layer in opt_state_dict["m_layers"]
        ],
        v_layers=[
            gpu.dict_to_gpu_layer(v_layer, tm_params.embedding_dim, device)
            for v_layer in opt_state_dict["v_layers"]
        ],
        step=opt_state_dict["step"],
    )

    print("✅ Model loaded from disk")

    return TransformerModel(
        tm_params=tm_params,
        params=gpu_params,
        opt_state=opt_state,
        learning_rate=learning_rate,
        total_steps=total_steps,
    )


# ============================================================================
# SAMPLING (for inference/generation)
# ============================================================================


def sample_token(
    logits,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    min_p=0.0,
    repetition_penalty=1.0,
    recent_tokens=None,
):
    """
    Sample next token from logits - pure NumPy implementation.

    Args:
        logits: np.ndarray or GPUBuffer - raw logits (vocab_size,)
        temperature: Controls randomness (0.0 = greedy, higher = more random)
        top_k: Only sample from top k tokens (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        min_p: Minimum probability threshold (0.0 = disabled)
        repetition_penalty: Penalize recently used tokens (1.0 = disabled)
        recent_tokens: List of recent token IDs

    Returns:
        Sampled token ID (int)
    """
    if recent_tokens is None:
        recent_tokens = []

    # Convert to NumPy if needed
    if hasattr(logits, "device"):
        logits = gpu.gpu_to_numpy(logits)
    elif hasattr(logits, "__array__"):
        logits = np.array(logits, dtype=np.float32)
    else:
        logits = np.array(logits, dtype=np.float32)

    # Ensure 1D
    if len(logits.shape) > 1:
        logits = logits.flatten()

    # Apply repetition penalty
    if repetition_penalty != 1.0 and len(recent_tokens) > 0:
        for token_id in set(recent_tokens):
            if 0 <= token_id < len(logits):
                logits[token_id] = logits[token_id] / repetition_penalty

    # Apply temperature scaling
    if temperature == 0.0:
        # Greedy sampling
        return int(np.argmax(logits))

    if temperature != 1.0:
        logits = logits / temperature

    # Convert to probabilities (softmax)
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)

    # Apply min-p filtering
    if min_p > 0.0:
        max_prob = np.max(probs)
        min_threshold = min_p * max_prob
        mask = probs >= min_threshold
        probs = np.where(mask, probs, 0.0)
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            # Fallback: uniform distribution
            probs = np.ones_like(probs) / len(probs)

    # Apply top-k filtering
    if top_k > 0:
        top_k_indices = np.argsort(probs)[-top_k:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[top_k_indices] = True
        probs = np.where(mask, probs, 0.0)
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum

    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)

        # Find cutoff
        mask = cumsum_probs <= top_p
        mask[0] = True  # Always include first token

        filtered_probs = np.zeros_like(probs)
        filtered_probs[sorted_indices] = np.where(mask, sorted_probs, 0.0)
        probs = filtered_probs
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum

    # Sample from distribution
    try:
        token_id = np.random.choice(len(probs), p=probs)
        return int(token_id)
    except ValueError:
        # Fallback to greedy if sampling fails
        return int(np.argmax(probs))
