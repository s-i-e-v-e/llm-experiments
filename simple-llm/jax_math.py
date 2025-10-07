from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import optax
from jax import jit, lax, random, value_and_grad

# --- UTILITY FUNCTIONS ---


def get_causal_mask(seq_len):
    """Create causal mask for decoder, compatible with JAX einsum/where."""
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
    return mask * -1e9


# --- JAX/FLAX TRANSFORMER COMPONENTS ---


class MultiHeadAttention(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        assert d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        d_k = d_model // self.n_heads

        # Linear Projections
        q = nn.Dense(features=d_model, use_bias=False, name="w_q")(x)
        k = nn.Dense(features=d_model, use_bias=False, name="w_k")(x)
        v = nn.Dense(features=d_model, use_bias=False, name="w_v")(x)

        # Reshape to [B, H, S, Dk]
        q = q.reshape(batch_size, seq_len, self.n_heads, d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, d_k).transpose(0, 2, 1, 3)

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


class FeedForward(nn.Module):
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(features=self.d_ff, name="w1")(x)
        h = nn.gelu(h)
        output = nn.Dense(features=self.d_model, name="w2")(h)
        return output


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, mask=None):
        # Pre-norm architecture
        x_norm = nn.LayerNorm(name="ln1")(x)
        attn_out = MultiHeadAttention(self.d_model, self.n_heads, name="attn")(
            x_norm, mask
        )
        x = x + attn_out

        x_norm = nn.LayerNorm(name="ln2")(x)
        ffn_out = FeedForward(self.d_model, d_ff=4 * self.d_model, name="ffn")(x_norm)
        output = x + ffn_out

        return output


class JAXTransformer(nn.Module):
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    max_seq_len: int

    @nn.compact
    def __call__(self, tokens):
        batch_size, seq_len = tokens.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        # Token Embeddings
        token_embeds = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            name="token_embedding",
        )(tokens)

        # Positional Embeddings
        pos_embeds = nn.Embed(
            num_embeddings=self.max_seq_len, features=self.d_model, name="pos_embedding"
        )(jnp.arange(seq_len))

        # Combine embeddings
        x = token_embeds + pos_embeds

        # Create causal mask
        mask = get_causal_mask(seq_len)

        # Transformer blocks
        for i in range(self.n_layers):
            x = TransformerBlock(self.d_model, self.n_heads, name=f"block_{i}")(x, mask)

        # Final layer norm
        x = nn.LayerNorm(name="ln_f")(x)

        # Output projection
        logits = nn.Dense(features=self.vocab_size, name="output_proj")(x)

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


# FIX: Add optimizer to static_argnums since it's passed as an argument
@partial(jit, static_argnums=(4, 5))  # model (4) AND optimizer (5) are static
def train_step(
    params,
    opt_state,
    tokens,
    targets,
    model,
    optimizer,  # This is the 5th argument (index 5)
    learning_rate,
):
    """
    Performs one full training step.
    """
    # Compute loss and gradients
    loss, grads = value_and_grad(cross_entropy_loss)(params, tokens, targets, model)

    # Apply updates
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss, learning_rate


@partial(
    jit, static_argnums=(1, 3, 4, 5)
)  # model, max_length, temperature, top_k are static
def generate(
    params,
    model,
    prompt_tokens,  # This is dynamic (not in static_argnums)
    max_length,
    temperature=1.0,
    top_k=None,
):
    """
    Generate tokens autoregressively without scan.
    """
    tokens = prompt_tokens
    key = random.PRNGKey(42)

    for i in range(max_length - len(prompt_tokens)):
        # Get model predictions for current sequence
        logits = model.apply({"params": params}, tokens[None, :])
        next_token_logits = logits[0, -1, :]

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k is not None:
            top_k_val = min(top_k, next_token_logits.shape[-1])
            top_values, top_indices = lax.top_k(next_token_logits, top_k_val)
            min_top_val = top_values[-1]
            next_token_logits = jnp.where(
                next_token_logits >= min_top_val, next_token_logits, -jnp.inf
            )

        # Sample from distribution
        key, subkey = random.split(key)
        next_token = random.categorical(subkey, next_token_logits)

        # Append to sequence
        tokens = jnp.concatenate([tokens, next_token[None]], axis=0)

    return tokens
