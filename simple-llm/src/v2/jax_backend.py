import dataclasses
import json
import random
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import struct

from common.util import deserialize, get_model_file_names, serialize

# JAX configuration
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "bfloat16")


# ============================================================================
# DATA STRUCTURES
# ============================================================================


def positional_encoding(seq_len, dim):
    pos = jnp.arange(seq_len)[:, None]
    i = jnp.arange(dim)[None, :]
    angle_rates = 1 / (10000 ** (2 * (i // 2) / dim))
    angle_rads = pos * angle_rates
    sines = jnp.sin(angle_rads[:, 0::2])
    cosines = jnp.cos(angle_rads[:, 1::2])
    pos_encoding = jnp.concatenate([sines, cosines], axis=-1)
    return pos_encoding


@struct.dataclass
class LayerParams:
    """Parameters for a single transformer layer"""

    attn_wq: jnp.ndarray
    attn_wk: jnp.ndarray
    attn_wv: jnp.ndarray
    attn_wo: jnp.ndarray
    ff_w1: jnp.ndarray
    ff_b1: jnp.ndarray
    ff_w2: jnp.ndarray
    ff_b2: jnp.ndarray
    ln_gamma1: jnp.ndarray
    ln_beta1: jnp.ndarray
    ln_gamma2: jnp.ndarray
    ln_beta2: jnp.ndarray


@struct.dataclass
class ModelParams:
    embedding: jnp.ndarray
    pos_encoding: jnp.ndarray
    layers: list


@dataclasses.dataclass
class TransformerModelParams:
    vocab_size: int
    embedding_dim: int
    context_size: int
    n_heads: int
    n_layers: int
    epochs: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TransformerModel:
    tm_params: TransformerModelParams
    params: ModelParams
    opt_state: optax.OptState
    optimizer: optax.GradientTransformation


# ============================================================================
# MODEL OPERATIONS
# ============================================================================


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    norm = (x - mean) / jnp.sqrt(var + eps)
    return norm * gamma + beta


def split_heads(x, n_heads, head_dim):
    B, T, C = x.shape
    x = x.reshape(B, T, n_heads, head_dim)
    return x.transpose(0, 2, 1, 3)


def combine_heads(x):
    B, H, T, D = x.shape
    x = x.transpose(0, 2, 1, 3).reshape(B, T, H * D)
    return x


def scaled_dot_product_attention(q, k, v):
    matmul_qk = jnp.matmul(q, jnp.swapaxes(k, -1, -2))
    scale = q.shape[-1] ** 0.5
    logits = matmul_qk / scale

    seq_len = q.shape[2]
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1) * -1e9
    logits = logits + mask[None, None, :, :]

    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.matmul(weights, v)


def multi_head_attention(x, params, n_heads, head_dim):
    q = jnp.dot(x, params.attn_wq)
    k = jnp.dot(x, params.attn_wk)
    v = jnp.dot(x, params.attn_wv)

    q = split_heads(q, n_heads, head_dim)
    k = split_heads(k, n_heads, head_dim)
    v = split_heads(v, n_heads, head_dim)

    attn_output = scaled_dot_product_attention(q, k, v)
    attn_output = combine_heads(attn_output)
    attn_output = jnp.dot(attn_output, params.attn_wo)
    return attn_output


def feed_forward(x, params):
    ff = jnp.dot(x, params.ff_w1) + params.ff_b1
    ff = jax.nn.gelu(ff)
    ff = jnp.dot(ff, params.ff_w2) + params.ff_b2
    return ff


def transformer_block(x, layer_params, n_heads, head_dim):
    x_norm = layer_norm(x, layer_params.ln_gamma1, layer_params.ln_beta1)
    attn_output = multi_head_attention(x_norm, layer_params, n_heads, head_dim)
    x = x + attn_output

    x_norm = layer_norm(x, layer_params.ln_gamma2, layer_params.ln_beta2)
    ff_output = feed_forward(x_norm, layer_params)
    out = x + ff_output
    return out


def forward(params, inputs, n_layers, n_heads, head_dim):
    x = params.embedding[inputs] + params.pos_encoding[None, :, :]
    for i, layer_params in enumerate(params.layers[:n_layers]):
        x = transformer_block(x, layer_params, n_heads, head_dim)
    logits = jnp.dot(x, params.embedding.T)
    return logits


def loss_fn(
    params, batch_inputs, batch_targets, vocab_size, n_layers, n_heads, head_dim
):
    logits = forward(params, batch_inputs, n_layers, n_heads, head_dim)
    log_probs = jax.nn.log_softmax(logits)
    one_hot_targets = jax.nn.one_hot(batch_targets, vocab_size)
    loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    return jnp.mean(loss)


# ============================================================================
# TRAINING OPERATIONS
# ============================================================================


@partial(
    jax.jit,
    static_argnames=["vocab_size", "n_layers", "n_heads", "head_dim", "optimizer"],
)
def train_step(
    params,
    opt_state,
    batch_inputs,
    batch_targets,
    vocab_size,
    n_layers,
    n_heads,
    head_dim,
    optimizer,
):
    """Single training step - JIT compiled"""

    def internal_loss_fn(p):
        return loss_fn(
            p, batch_inputs, batch_targets, vocab_size, n_layers, n_heads, head_dim
        )

    loss, grads = jax.value_and_grad(internal_loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


# ============================================================================
# TRAINING EPOCH (matches OLD performance)
# ============================================================================


def train_epoch(
    model: TransformerModel,
    token_data: list,
    batch_size: int,
    context_size: int,
    epoch_num: int,
):
    """
    Train for one epoch - matches OLD code's batching strategy.
    Uses pre-created sequences for best performance.
    """
    import time

    # Create sequences EXACTLY like OLD code
    sequences = []
    stride = batch_size
    for i in range(0, len(token_data) - context_size, stride):
        seq = token_data[i : i + context_size + 1]
        if len(seq) == context_size + 1:
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

    # Simple training loop like OLD
    for batch_start in range(0, len(sequences), batch_size):
        # Batch sequences like OLD
        batch = sequences[batch_start : batch_start + batch_size]
        if len(batch) == 0:
            continue

        batch_inputs = [seq[0] for seq in batch]
        batch_targets = [seq[1] for seq in batch]

        # Convert to JAX once per batch (like OLD's to_backend)
        batch_inputs_jax = jnp.array(batch_inputs, dtype=jnp.int32)
        batch_targets_jax = jnp.array(batch_targets, dtype=jnp.int32)

        # Single training step
        new_params, new_opt_state, loss = train_step(
            model.params,
            model.opt_state,
            batch_inputs_jax,
            batch_targets_jax,
            vocab_size=model.tm_params.vocab_size,
            n_layers=model.tm_params.n_layers,
            n_heads=model.tm_params.n_heads,
            head_dim=model.tm_params.embedding_dim // model.tm_params.n_heads,
            optimizer=model.optimizer,
        )

        # Update model
        model = TransformerModel(
            tm_params=model.tm_params,
            params=new_params,
            opt_state=new_opt_state,
            optimizer=model.optimizer,
        )

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
            (batch_size * context_size * step) / elapsed if elapsed > 0 else 0
        )

        print(
            f"\rEpoch {epoch_num + 1}, Step {step}/{steps_in_epoch} | "
            f"Loss: {float(loss):.4f} | Smooth: {smooth_loss:.4f} | "
            f"TPS: {tokens_per_sec:.0f}",
            end="",
        )

    print()  # New line after progress

    avg_loss = epoch_loss / step if step > 0 else 0.0

    return model, {
        "avg_loss": avg_loss,
        "smooth_loss": smooth_loss,
        "steps": step,
    }


# ============================================================================
# MODEL LIFECYCLE
# ============================================================================


def create_optimizer_with_schedule(scaled_lr, total_steps):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=scaled_lr,
        warmup_steps=total_steps // 10,
        decay_steps=total_steps,
        end_value=scaled_lr * 0.1,
    )
    return optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=0.1)


def initialize_model(
    vocab_size,
    embedding_dim,
    context_size,
    n_heads,
    n_layers,
    epochs,
    learning_rate,
    total_steps,
):
    """Initialize model - no pre-compilation overhead"""
    key = jax.random.PRNGKey(0)

    layers = []
    for i in range(n_layers):
        key, subkey = jax.random.split(key)
        k_wq, k_wk, k_wv, k_wo, k_w1, k_w2 = jax.random.split(subkey, 6)

        layers.append(
            LayerParams(
                attn_wq=jax.random.normal(k_wq, (embedding_dim, embedding_dim)) * 0.01,
                attn_wk=jax.random.normal(k_wk, (embedding_dim, embedding_dim)) * 0.01,
                attn_wv=jax.random.normal(k_wv, (embedding_dim, embedding_dim)) * 0.01,
                attn_wo=jax.random.normal(k_wo, (embedding_dim, embedding_dim)) * 0.01,
                ff_w1=jax.random.normal(k_w1, (embedding_dim, 4 * embedding_dim))
                * 0.01,
                ff_b1=jnp.zeros((4 * embedding_dim,)),
                ff_w2=jax.random.normal(k_w2, (4 * embedding_dim, embedding_dim))
                * 0.01,
                ff_b2=jnp.zeros((embedding_dim,)),
                ln_gamma1=jnp.ones((embedding_dim,)),
                ln_beta1=jnp.zeros((embedding_dim,)),
                ln_gamma2=jnp.ones((embedding_dim,)),
                ln_beta2=jnp.zeros((embedding_dim,)),
            )
        )

    key, k1 = jax.random.split(key)
    params = ModelParams(
        embedding=jax.random.normal(k1, (vocab_size, embedding_dim)) * 0.01,
        pos_encoding=positional_encoding(context_size, embedding_dim),
        layers=layers,
    )

    optimizer = create_optimizer_with_schedule(
        scaled_lr=learning_rate, total_steps=total_steps
    )
    opt_state = optimizer.init(params)

    return TransformerModel(
        TransformerModelParams(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_size=context_size,
            n_heads=n_heads,
            n_layers=n_layers,
            epochs=epochs,
        ),
        params=params,
        opt_state=opt_state,
        optimizer=optimizer,
    )


def save_model(model: TransformerModel, model_path: str):
    xs = get_model_file_names(model_path)
    serialize(model.params, xs[0])
    serialize(model.opt_state, xs[1])
    with open(xs[2], "w") as f:
        json.dump(dataclasses.asdict(model.tm_params), f, indent=2)


def load_model(learning_rate: float, total_steps: int, model_path: str):
    xs = get_model_file_names(model_path)
    with open(xs[2], "r") as f:
        config_data = json.load(f)
        if "epochs" not in config_data:
            config_data["epochs"] = []
        tm_params = TransformerModelParams(**config_data)

    q = dataclasses.asdict(tm_params)
    q["learning_rate"] = learning_rate
    q["total_steps"] = total_steps
    dummy_model = initialize_model(**q)

    params = deserialize(dummy_model.params, xs[0])
    opt_state = deserialize(dummy_model.opt_state, xs[1])

    return TransformerModel(
        tm_params=tm_params,
        params=params,
        opt_state=opt_state,
        optimizer=dummy_model.optimizer,
    )


def sample_token(
    key, logits, temperature, top_k, top_p, min_p, repetition_penalty, recent_tokens
):
    """
    Sample next token from logits with various filtering strategies.

    Args:
        logits: Raw logits from model (vocab_size,)
        temperature: Controls randomness
            - Range: 0.0 to 2.0+
            - Default: 1.0 (disabled/neutral)
            - Lower (0.1-0.7): More deterministic, focused
            - Higher (1.1-2.0): More creative, diverse
            - 0.0: Greedy sampling (always picks most likely token)

        top_k: Only sample from top k most likely tokens
            - Range: 1 to vocab_size
            - Default: 0 (disabled - considers all tokens)
            - Common: 20-100
            - Lower: More focused
            - Higher: More diverse

        top_p: Nucleus sampling - cumulative probability threshold
            - Range: 0.0 to 1.0
            - Default: 1.0 (disabled)
            - Common: 0.9-0.95
            - Lower: More focused on likely tokens
            - Higher: Considers more tokens

        min_p: Minimum probability threshold (relative to max probability)
            - Range: 0.0 to 1.0
            - Default: 0.0 (disabled)
            - Common: 0.05-0.1
            - Filters tokens with prob < min_p * max_prob

        repetition_penalty: Penalize recently used tokens
            - Range: 1.0 to 1.5
            - Default: 1.0 (disabled)
            - Common: 1.05-1.2
            - WARNING: Values > 1.2 can severely degrade quality
            - Higher: Less repetition (but may force awkward phrasing)

        recent_tokens: List of recently generated token IDs for repetition penalty

    Returns:
        Sampled token ID (int)
    """
    # Apply repetition penalty
    if repetition_penalty != 1.0 and len(recent_tokens) > 0:
        for token_id in set(recent_tokens):
            # Divide logit by penalty if > 1.0 (decreases probability)
            logits = logits.at[token_id].set(logits[token_id] / repetition_penalty)

    # Apply temperature scaling
    if temperature == 0.0:
        # Greedy sampling
        return jnp.argmax(logits).item()

    logits = logits / temperature

    # Convert to probabilities
    probs = jax.nn.softmax(logits)

    # Apply min-p filtering
    if min_p > 0.0:
        max_prob = jnp.max(probs)
        min_threshold = min_p * max_prob
        mask = probs >= min_threshold
        probs = jnp.where(mask, probs, 0.0)
        # Renormalize
        probs = probs / jnp.sum(probs)

    # Apply top-k filtering
    if top_k > 0:
        top_k_indices = jnp.argsort(probs)[-top_k:]
        mask = jnp.zeros_like(probs, dtype=bool)
        mask = mask.at[top_k_indices].set(True)
        probs = jnp.where(mask, probs, 0.0)
        # Renormalize
        probs = probs / jnp.sum(probs)

    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = jnp.argsort(probs)[::-1]  # Descending order
        sorted_probs = probs[sorted_indices]
        cumsum_probs = jnp.cumsum(sorted_probs)

        # Find cutoff where cumulative probability exceeds top_p
        mask = cumsum_probs <= top_p
        # Always include at least the first token
        mask = mask.at[0].set(True)

        # Create filtered probability distribution
        filtered_probs = jnp.zeros_like(probs)
        filtered_probs = filtered_probs.at[sorted_indices].set(
            jnp.where(mask, sorted_probs, 0.0)
        )
        probs = filtered_probs / jnp.sum(filtered_probs)

    # Sample from filtered distribution
    token_id = jax.random.choice(key, jnp.arange(len(probs)), p=probs)

    return int(token_id)
