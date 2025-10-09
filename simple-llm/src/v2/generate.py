import jax
import jax.numpy as jnp

from common.util import load_tokenizer
from v2.jax_backend import forward, load_model, sample_token


def generate_text(
    model,
    prompt_tokens,
    max_tokens=100,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    min_p=0.0,
    repetition_penalty=1.0,
    repetition_window=64,
):
    """Generate text - simple non-JIT version for compatibility"""

    context_length = model.tm_params.context_length
    n_heads = model.tm_params.n_heads
    head_dim = model.tm_params.embedding_dim // n_heads

    # Simple Python loop (no JIT complications)
    tokens = list(prompt_tokens)
    key = jax.random.PRNGKey(42)

    for _ in range(max_tokens):
        # Get context window
        context_start = max(0, len(tokens) - context_length)
        context = tokens[context_start:]

        # Pad if necessary
        if len(context) < context_length:
            context = [0] * (context_length - len(context)) + context

        # Convert to JAX array
        context_array = jnp.array([context], dtype=jnp.int32)

        # Forward pass
        logits = forward(
            model.params,
            context_array,
            n_layers=model.tm_params.n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
        )

        next_token_logits = logits[0, -1, :]

        # Get recent tokens for repetition penalty
        recent_start = max(0, len(tokens) - repetition_window)
        recent_tokens = jnp.array(tokens[recent_start:], dtype=jnp.int32)

        # Sample next token
        key, subkey = jax.random.split(key)
        next_token = sample_token(
            subkey,
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            recent_tokens=recent_tokens,
        )

        tokens.append(int(next_token))

    return tokens


def generate_command(args):
    """
    Generate text from a prompt using trained model.

    Args:
        args should contain:
        - model_path: Path to trained model
        - prompt: Input text prompt
        - max_tokens: Number of tokens to generate (default 100)
        - temperature: Sampling temperature (default 1.0)
        - top_k: Top-k sampling (default 0 disabled)
        - top_p: Nucleus sampling (default 1.0 disabled)
        - min_p: Min-p sampling (default 0.0 disabled)
        - repetition_penalty: Repetition penalty (default 1.0 disabled)
        - repetition_window: Tokens to consider for repetition (default 64)
        - learning_rate: Learning rate used during training
    """

    # Load trained model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.learning_rate, 1000, args.model_path)
    print("Model loaded successfully")

    # Load tokenizer
    tokenizer, tokenizer_path = load_tokenizer(
        args.model_path,
        model.tm_params.vocab_size,
        corpus_text=None,  # Not needed for loading existing tokenizer
    )
    print(f"Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt, tokenizer_path)
    print(f"\n{args.prompt}")
    print(f"Prompt tokens: {prompt_tokens} (length: {len(prompt_tokens)})")

    # Set default values
    temperature = getattr(args, "temperature", 1.0)
    top_k = getattr(args, "top_k", 0)
    top_p = getattr(args, "top_p", 1.0)
    min_p = getattr(args, "min_p", 0.0)
    repetition_penalty = getattr(args, "repetition_penalty", 1.0)
    repetition_window = getattr(args, "repetition_window", 64)

    # Log parameters
    print("\nGeneration parameters:")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}")
    print(f"  Top-p: {top_p}")
    print(f"  Min-p: {min_p}")
    print(f"  Repetition penalty: {repetition_penalty}")
    print(f"  Repetition window: {repetition_window}")

    # Generate
    print("\nGenerating...")
    generated_tokens = generate_text(
        model=model,
        prompt_tokens=prompt_tokens,
        max_tokens=args.max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
    )

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens)

    # Display results
    print("\n" + "=" * 60)
    print("GENERATED TEXT")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)
    print(f"\nNew tokens generated: {len(generated_tokens) - len(prompt_tokens)}")
    print(f"Total tokens (including prompt): {len(generated_tokens)}")
