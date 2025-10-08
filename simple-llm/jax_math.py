from functools import partial

import flax.linen as nn
import jax
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


@partial(jit, static_argnums=(1, 3, 4, 5, 6, 7, 8, 9))
def generate(
    params,
    model,
    prompt_tokens,
    max_length,
    temperature=1.0,  # 1.0 = no temperature scaling
    top_k=-1,  # -1 = no top-k filtering
    top_p=1.0,  # 1.0 = no top-p filtering
    min_p=0.0,  # 0.0 = no min-p filtering
    typical_p=1.0,  # 1.0 = no typical-p filtering
    repetition_penalty=1.0,  # 1.0 = no repetition penalty
):
    key = random.PRNGKey(42)
    tokens = prompt_tokens

    # Pre-compute all static conditions
    use_temperature = temperature != 1.0
    use_top_k = top_k > 0
    use_top_p = top_p < 1.0
    use_min_p = min_p > 0.0
    use_typical_p = typical_p < 1.0
    use_repetition_penalty = repetition_penalty != 1.0

    for i in range(max_length - len(prompt_tokens)):
        # Get model predictions for current sequence
        logits = model.apply({"params": params}, tokens[None, :])
        next_token_logits = logits[0, -1, :]

        # Apply all filters in sequence
        next_token_logits = apply_temperature(
            next_token_logits, use_temperature, temperature
        )
        next_token_logits = apply_repetition_penalty(
            next_token_logits, use_repetition_penalty, repetition_penalty
        )
        next_token_logits = apply_top_k(next_token_logits, use_top_k, top_k)
        next_token_logits = apply_top_p(next_token_logits, use_top_p, top_p)
        next_token_logits = apply_min_p(next_token_logits, use_min_p, min_p)
        next_token_logits = apply_typical_p(next_token_logits, use_typical_p, typical_p)

        # Sample from distribution
        key, subkey = random.split(key)
        next_token = random.categorical(subkey, next_token_logits)

        # Append to sequence
        tokens = jnp.concatenate([tokens, next_token[None]], axis=0)

    return tokens


def apply_temperature(logits, use_temperature, temperature):
    return lax.cond(use_temperature, lambda x: x / temperature, lambda x: x, logits)


def apply_top_k(logits, use_top_k, top_k):
    def _apply(logits):
        top_k_val = min(top_k, logits.shape[-1])
        top_values, _ = lax.top_k(logits, top_k_val)
        min_top_val = top_values[-1]
        return jnp.where(logits >= min_top_val, logits, -jnp.inf)

    return lax.cond(use_top_k, _apply, lambda x: x, logits)


def apply_min_p(logits, use_min_p, min_p):
    def _apply(logits):
        max_logit = jnp.max(logits)
        threshold = max_logit + jnp.log(min_p)
        return jnp.where(logits >= threshold, logits, -jnp.inf)

    return lax.cond(use_min_p, _apply, lambda x: x, logits)


def apply_top_p(logits, use_top_p, top_p):
    def _apply(logits):
        probs = jax.nn.softmax(logits)
        sorted_probs = jnp.sort(probs)[::-1]
        cumulative_probs = jnp.cumsum(sorted_probs)

        # Find how many tokens to keep
        cutoff_index = jnp.sum((cumulative_probs - 1e-8) <= top_p)

        # Use jnp.where instead of Python if statement
        cutoff_prob = jnp.where(
            cutoff_index < len(sorted_probs), sorted_probs[cutoff_index], 0.0
        )

        return jnp.where(probs >= cutoff_prob, logits, -jnp.inf)

    return lax.cond(use_top_p, _apply, lambda x: x, logits)


def apply_typical_p(logits, use_typical_p, typical_p):
    def _apply(logits):
        # Typical-p filtering implementation
        probs = jax.nn.softmax(logits)
        log_probs = jnp.log(probs)
        entropy = -jnp.sum(probs * log_probs)
        typical_threshold = jnp.exp(-entropy) * typical_p

        # Keep tokens where log_prob > log(typical_threshold)
        return jnp.where(log_probs > jnp.log(typical_threshold), logits, -jnp.inf)

    return lax.cond(use_typical_p, _apply, lambda x: x, logits)


def apply_repetition_penalty(logits, use_repetition_penalty, repetition_penalty):
    def _apply(logits):
        # Simple repetition penalty - in practice you'd need the previous tokens
        # This is a simplified version
        return logits  # Implement proper repetition penalty based on token history

    return lax.cond(use_repetition_penalty, _apply, lambda x: x, logits)
