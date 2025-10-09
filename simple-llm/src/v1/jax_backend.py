import dataclasses
import json
import os
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from jax import jit, random, value_and_grad

from common.util import get_model_config_path, get_model_weights_path


@dataclasses.dataclass
class JaxModel:
    logits: jax.Array
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


def model_init(model_path, model_config, resume, learning_rate, context_length):
    if learning_rate:
        optimizer = create_optimizer(learning_rate)
    else:
        optimizer = None

    logits = jax_transformer(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        max_seq_len=model_config["max_seq_len"],
    )

    x_model = JaxModel(logits, optimizer)

    key = random.PRNGKey(42)
    model_key, data_key = random.split(key)

    model_params = model_load(model_path)
    if resume and model_params is not None:
        print("Resuming training")
    else:
        print("Initializing new JAX/Flax transformer model")
        dummy_input = jnp.ones((1, context_length), dtype=jnp.int32)
        initial_vars = x_model.model.init(model_key, dummy_input)
        model_params = initial_vars["params"]  # Extract params from the variables dict
    opt_state = x_model.optimizer.init(model_params)
    return data_key, x_model, opt_state, model_params


def int_array(xs):
    return jnp.array(xs, dtype=jnp.int32)


def main_train_step(
    data_key,
    X,
    Y,
    batch_size,
    seq_length,
    model_params,
    x_model,
    opt_state,
):
    data_key, batch_tokens, batch_targets = get_batch(
        data_key, X, Y, batch_size, seq_length
    )

    model_params, opt_state, loss_v = train_step(
        model_params,
        opt_state,
        batch_tokens,
        batch_targets,
        x_model.model,
        x_model.optimizer,
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


@nn.compact
def multi_head_attention(d_model: int, n_heads: int, x, mask=None):
    batch_size, seq_len, d_model = x.shape

    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_k = d_model // n_heads

    # Linear Projections
    q = nn.Dense(features=d_model, use_bias=False, name="w_q")(x)
    k = nn.Dense(features=d_model, use_bias=False, name="w_k")(x)
    v = nn.Dense(features=d_model, use_bias=False, name="w_v")(x)

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
    attn_weights = nn.softmax(scores, axis=-1)
    attn_output = jnp.einsum("bhij,bhjd->bhid", attn_weights, v)

    # Combine heads
    attn_output_combined = attn_output.transpose(0, 2, 1, 3).reshape(
        batch_size, seq_len, d_model
    )

    # Output projection
    output = nn.Dense(features=d_model, name="w_o")(attn_output_combined)
    return output


@nn.compact
def feed_forward(d_model: int, d_ff: int, x):
    h = nn.Dense(features=d_ff, name="w1")(x)
    h = nn.gelu(h)
    output = nn.Dense(features=d_model, name="w2")(h)
    return output


@nn.compact
def transformer_block(d_model: int, n_heads: int, x, mask=None):
    # Pre-norm architecture
    x_norm = nn.LayerNorm(name="ln1")(x)
    attn_out = multi_head_attention(d_model, n_heads, x_norm, mask, name="attn")
    x = x + attn_out

    x_norm = nn.LayerNorm(name="ln2")(x)
    ffn_out = feed_forward(d_model, 4 * d_model, x_norm, name="ffn")
    output = x + ffn_out

    return output


@nn.compact
def jax_transformer(
    vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int, tokens
):
    batch_size, seq_len = tokens.shape

    if seq_len > max_seq_len:
        raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {max_seq_len}")

    # Token Embeddings
    token_embeds = nn.Embed(
        num_embeddings=vocab_size,
        features=d_model,
        name="token_embedding",
    )(tokens)

    # Positional Embeddings
    pos_embeds = nn.Embed(
        num_embeddings=max_seq_len, features=d_model, name="pos_embedding"
    )(jnp.arange(seq_len))

    # Combine embeddings
    x = token_embeds + pos_embeds

    # Create causal mask
    mask = get_causal_mask(seq_len)

    # Transformer blocks
    for i in range(n_layers):
        x = transformer_block(d_model, n_heads, x, mask, name=f"block_{i}")

    # Final layer norm
    x = nn.LayerNorm(name="ln_f")(x)

    # Output projection
    logits = nn.Dense(features=vocab_size, name="output_proj")(x)

    return logits


# --- TRAINING UTILITIES ---
def cross_entropy_loss(params, tokens, targets, model):
    """Calculate cross-entropy loss."""
    logits = model.apply({"params": params}, tokens)  # FIX: Use proper params structure

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


@partial(jit, static_argnums=(4, 5))  # model (4) AND optimizer (5) are static
def train_step(
    params,
    opt_state,
    tokens,
    targets,
    model,
    optimizer,  # This is the 5th argument (index 5)
):
    """
    Performs one full training step.
    """
    # Compute loss and gradients
    loss, grads = value_and_grad(cross_entropy_loss)(params, tokens, targets, model)

    # Apply updates
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss
