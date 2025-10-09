import dataclasses
import json
import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from jax import jit, random, value_and_grad

from common.util import get_model_config_path, get_model_weights_path


@dataclasses.dataclass
class JaxModel:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    max_seq_len: int
    optimizer: optax.GradientTransformation


jax.config.update("jax_enable_x64", False)


# --- UTILITY FUNCTIONS ---
def get_causal_mask(seq_len):
    """Create causal mask for decoder, compatible with JAX einsum/where."""
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
    return mask * -1e9


def model_save(params, model_path):
    """Save model using Flax's efficient binary format"""
    try:
        file = get_model_weights_path(model_path)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as f:
            f.write(serialization.to_bytes(params))
        print(f"Model saved to {file}")
    except Exception as e:
        print(f"Error saving JAX model: {e}")


def model_load(model_path):
    """Load model using Flax's binary format"""
    try:
        file = get_model_weights_path(model_path)
        with open(file, "rb") as f:
            return serialization.msgpack_restore(f.read())
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading JAX model: {e}")
        return None


def load_model_config(model_path: str, vocab_size: int, args):
    file = get_model_config_path(model_path)
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        x = {
            "vocab_size": vocab_size,
            "d_model": args.embedding_dim,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "max_seq_len": args.context_length,
            "epochs": [],
        }
        save_model_config(x, model_path)
        return x


def save_model_config(config, model_path):
    file = get_model_config_path(model_path)
    with open(file, "w") as f:
        json.dump(config, f, indent=2)


@dataclasses.dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    max_seq_len: int


def model_init(model_path, model_config, resume, learning_rate, context_length):
    if learning_rate:
        optimizer = create_optimizer(learning_rate)
    else:
        optimizer = None

    config = ModelConfig(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        max_seq_len=model_config["max_seq_len"],
    )

    key = random.PRNGKey(42)
    model_key, data_key = random.split(key)

    model_params = model_load(model_path)
    if resume and model_params is not None:
        print("Resuming training")
    else:
        print("Initializing new JAX/Flax transformer model")
        model_params = init_transformer_params(
            model_key,
            config.vocab_size,
            config.d_model,
            config.n_heads,
            config.n_layers,
            config.max_seq_len,
        )
    opt_state = optimizer.init(model_params)
    return data_key, config, optimizer, opt_state, model_params


def int_array(xs):
    return jnp.array(xs, dtype=jnp.int32)


def main_train_step(
    data_key, X, Y, batch_size, seq_length, model_params, config, opt_state, optimizer
):
    data_key, batch_tokens, batch_targets = get_batch(
        data_key, X, Y, batch_size, seq_length
    )

    model_params, opt_state, loss_v = train_step(
        model_params,
        opt_state,
        batch_tokens,
        batch_targets,
        config.vocab_size,
        config.d_model,
        config.n_heads,
        config.n_layers,
        config.max_seq_len,
        optimizer,
    )

    return data_key, model_params, opt_state, float(loss_v.item())


# Create a function to generate batches
def get_batch(key, X, Y, batch_size, seq_length):
    key, subkey = random.split(key)
    """Generate a random batch of data"""
    # Generate random starting indices
    start_idxs = random.randint(
        subkey, shape=(batch_size,), minval=0, maxval=len(X) - seq_length
    )

    # Create batches
    batch_X = jnp.stack([X[i : i + seq_length] for i in start_idxs])
    batch_Y = jnp.stack([Y[i : i + seq_length] for i in start_idxs])

    return key, batch_X, batch_Y


def layer_norm(x, gamma, beta, eps=1e-5):
    """Apply layer normalization."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    norm = (x - mean) / jnp.sqrt(var + eps)
    return norm * gamma + beta


def multi_head_attention(params, x, d_model, n_heads, mask=None):
    """Multi-head attention mechanism."""
    batch_size, seq_len, _ = x.shape

    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_k = d_model // n_heads

    # Linear Projections
    q = jnp.dot(x, params["w_q"])
    k = jnp.dot(x, params["w_k"])
    v = jnp.dot(x, params["w_v"])

    # Reshape to [B, H, S, Dk]
    q = q.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
    v = v.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)

    # Attention Scores
    scores = jnp.einsum("bhid,bhjd->bhij", q, k) / jnp.sqrt(d_k)

    # Masking
    if mask is not None:
        mask = mask[jnp.newaxis, jnp.newaxis, :, :]
        scores = scores + mask

    # Softmax and Attention Output
    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_output = jnp.einsum("bhij,bhjd->bhid", attn_weights, v)

    # Combine heads
    attn_output_combined = attn_output.transpose(0, 2, 1, 3).reshape(
        batch_size, seq_len, d_model
    )

    # Output projection
    output = jnp.dot(attn_output_combined, params["w_o"])
    return output


def feed_forward(params, x, d_model, d_ff):
    """Feed-forward network."""
    h = jnp.dot(x, params["w1"]) + params["b1"]
    h = jax.nn.gelu(h)
    output = jnp.dot(h, params["w2"]) + params["b2"]
    return output


def transformer_block(params, x, d_model, n_heads, mask=None):
    """Single transformer block with pre-norm architecture."""
    # Pre-norm architecture
    x_norm = layer_norm(x, params["ln1_gamma"], params["ln1_beta"])
    attn_out = multi_head_attention(params["attn"], x_norm, d_model, n_heads, mask)
    x = x + attn_out

    x_norm = layer_norm(x, params["ln2_gamma"], params["ln2_beta"])
    d_ff = 4 * d_model
    ffn_out = feed_forward(params["ffn"], x_norm, d_model, d_ff)
    output = x + ffn_out

    return output


def jax_transformer(
    params, tokens, vocab_size, d_model, n_heads, n_layers, max_seq_len
):
    """Pure functional transformer."""
    batch_size, seq_len = tokens.shape

    if seq_len > max_seq_len:
        raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {max_seq_len}")

    # Token Embeddings
    token_embeds = params["token_embedding"][tokens]

    # Positional Embeddings
    pos_embeds = params["pos_embedding"][jnp.arange(seq_len)]

    # Combine embeddings
    x = token_embeds + pos_embeds

    # Create causal mask
    mask = get_causal_mask(seq_len)

    # Transformer blocks
    for i in range(n_layers):
        x = transformer_block(params[f"block_{i}"], x, d_model, n_heads, mask)

    # Final layer norm
    x = layer_norm(x, params["ln_f_gamma"], params["ln_f_beta"])

    # Output projection
    logits = jnp.dot(x, params["output_proj_w"]) + params["output_proj_b"]

    return logits


# --- TRAINING UTILITIES ---
def cross_entropy_loss(
    params, tokens, targets, vocab_size, d_model, n_heads, n_layers, max_seq_len
):
    """Calculate cross-entropy loss."""
    logits = jax_transformer(
        params, tokens, vocab_size, d_model, n_heads, n_layers, max_seq_len
    )

    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat)
    return jnp.mean(loss)


def calculate_lr(global_step, max_lr, warmup_steps):
    """Calculates learning rate with linear warmup."""
    return max_lr * jnp.minimum(1.0, (global_step + 1) / warmup_steps)


def create_optimizer(max_lr, beta1=0.9, beta2=0.95, weight_decay=0.1):
    """Create the AdamW optimizer."""
    return optax.adamw(
        learning_rate=max_lr, b1=beta1, b2=beta2, weight_decay=weight_decay
    )


@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9))
def train_step(
    params,
    opt_state,
    tokens,
    targets,
    vocab_size,
    d_model,
    n_heads,
    n_layers,
    max_seq_len,
    optimizer,
):
    """Performs one full training step."""

    def internal_loss_fn(p):
        return cross_entropy_loss(
            p, tokens, targets, vocab_size, d_model, n_heads, n_layers, max_seq_len
        )

    # Compute loss and gradients
    loss, grads = value_and_grad(internal_loss_fn)(params)

    # Apply updates
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss


# --- PARAMETER INITIALIZATION ---


def init_transformer_params(rng, vocab_size, d_model, n_heads, n_layers, max_seq_len):
    """Initialize all transformer parameters."""
    params = {}
    d_ff = 4 * d_model

    # Split RNG keys
    rng, token_rng, pos_rng = jax.random.split(rng, 3)

    # Token and positional embeddings
    params["token_embedding"] = (
        jax.random.normal(token_rng, (vocab_size, d_model)) * 0.02
    )
    params["pos_embedding"] = jax.random.normal(pos_rng, (max_seq_len, d_model)) * 0.02

    # Transformer blocks
    for i in range(n_layers):
        rng, block_rng = jax.random.split(rng)
        params[f"block_{i}"] = init_transformer_block(block_rng, d_model, d_ff)

    # Final layer norm
    params["ln_f_gamma"] = jnp.ones(d_model)
    params["ln_f_beta"] = jnp.zeros(d_model)

    # Output projection
    rng, output_rng = jax.random.split(rng)
    params["output_proj_w"] = (
        jax.random.normal(output_rng, (d_model, vocab_size)) * 0.02
    )
    params["output_proj_b"] = jnp.zeros(vocab_size)

    return params


def init_transformer_block(rng, d_model, d_ff):
    """Initialize parameters for a single transformer block."""
    block_params = {}

    # Attention parameters
    rng, attn_rng = jax.random.split(rng)
    block_params["attn"] = init_attention(attn_rng, d_model)

    # Feed-forward parameters
    rng, ff_rng = jax.random.split(rng)
    block_params["ffn"] = init_feedforward(ff_rng, d_model, d_ff)

    # Layer norm parameters
    block_params["ln1_gamma"] = jnp.ones(d_model)
    block_params["ln1_beta"] = jnp.zeros(d_model)
    block_params["ln2_gamma"] = jnp.ones(d_model)
    block_params["ln2_beta"] = jnp.zeros(d_model)

    return block_params


def init_attention(rng, d_model):
    """Initialize multi-head attention parameters."""
    rngs = jax.random.split(rng, 4)

    return {
        "w_q": jax.random.normal(rngs[0], (d_model, d_model)) * 0.02,
        "w_k": jax.random.normal(rngs[1], (d_model, d_model)) * 0.02,
        "w_v": jax.random.normal(rngs[2], (d_model, d_model)) * 0.02,
        "w_o": jax.random.normal(rngs[3], (d_model, d_model)) * 0.02,
    }


def init_feedforward(rng, d_model, d_ff):
    """Initialize feed-forward network parameters."""
    rngs = jax.random.split(rng, 2)

    return {
        "w1": jax.random.normal(rngs[0], (d_model, d_ff)) * 0.02,
        "b1": jnp.zeros(d_ff),
        "w2": jax.random.normal(rngs[1], (d_ff, d_model)) * 0.02,
        "b2": jnp.zeros(d_model),
    }
