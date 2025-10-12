from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, random

from v1.jax_backend import int_array, jax_transformer


def main_generate(
    params,
    config,
    xs,
    max_length,
):
    # Generate tokens
    generated_tokens = generate(
        params=params,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.max_seq_len,
        prompt_tokens=int_array(xs),
        max_length=max_length,
        top_k=40,
    )

    return generated_tokens.tolist()


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13))
def generate(
    params,
    vocab_size,
    d_model,
    n_heads,
    n_layers,
    max_seq_len,
    prompt_tokens,
    max_length,
    temperature=1.0,
    top_k=-1,
    top_p=1.0,
    min_p=0.0,
    typical_p=1.0,
    repetition_penalty=1.0,
):
    """Generate text using the transformer."""
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
        logits = jax_transformer(
            params, tokens[None, :], vocab_size, d_model, n_heads, n_layers, max_seq_len
        )
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
