import json
import math
import os
import sys
import time

import jax.numpy as jnp
from flax import serialization
from jax import random

from bpe_tokenizer import BPETokenizer
from jax_math import JAXTransformer, create_optimizer, generate, train_step

MODEL_CONFIG_FILE = "model_config.json"
MODEL_WEIGHTS_FILE = "model.jax"
TOKENIZER_FILE = "tokenizer.json"


def get_tokenizer_path(model_path: str):
    return os.path.join(model_path, TOKENIZER_FILE)


def get_model_config_path(model_path: str):
    return os.path.join(model_path, MODEL_CONFIG_FILE)


def get_model_weights_path(model_path: str):
    return os.path.join(model_path, MODEL_WEIGHTS_FILE)


def load_corpus(file):
    """Load text corpus from file"""
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


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


def save_jax_model(params, model_path):
    """Save model using Flax's efficient binary format"""
    try:
        file = get_model_weights_path(model_path)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as f:
            f.write(serialization.to_bytes(params))
        print(f"Model saved to {file}")
    except Exception as e:
        print(f"Error saving JAX model: {e}")


def load_jax_model(model_path):
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


def load_tokenizer(model_path: str, vocab_size: int = 0, corpus_text: str = ""):
    tokenizer = BPETokenizer()
    file = get_tokenizer_path(model_path)
    if os.path.exists(file):
        print(f"Loading tokenizer from {file}")
        tokenizer = BPETokenizer.load(file)
    else:
        assert vocab_size > 0
        assert corpus_text != ""
        print(f"Training new BPE tokenizer with vocab_size {vocab_size}")
        tokenizer.train(corpus_text, vocab_size, file, verbose=True)
    return tokenizer, file


def train_command_jax(args):
    """Main training function for transformer using JAX/Flax backend"""
    os.makedirs(args.model_path, exist_ok=True)

    # Load corpus and tokenizer
    corpus_text = load_corpus(args.input_file)
    print(f"Loaded corpus: {len(corpus_text)} characters")

    tokenizer, tokenizer_path = load_tokenizer(
        args.model_path, args.vocab_size, corpus_text
    )

    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {vocab_size}")

    # Encode corpus
    print("Encoding corpus...")
    corpus_tokens = jnp.array(
        tokenizer.encode(corpus_text, tokenizer_path), dtype=jnp.int32
    )
    print(f"Tokenized corpus: {len(corpus_tokens)} tokens")

    # JAX initialization
    key = random.PRNGKey(42)
    model_key, data_key = random.split(key)

    # Create optimizer
    optimizer = create_optimizer(args.learning_rate)

    model_config = load_model_config(args.model_path, vocab_size, args)
    model = JAXTransformer(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        max_seq_len=model_config["max_seq_len"],
    )

    # Load or initialize model parameters
    loaded_params = load_jax_model(args.model_path)
    if args.resume and loaded_params is not None:
        model_params = loaded_params
        print(f"Resuming training from {args.model_path}")
        opt_state = optimizer.init(model_params)
    else:
        print("Initializing new JAX/Flax transformer model")
        dummy_input = jnp.ones((1, args.context_length), dtype=jnp.int32)
        initial_vars = model.init(model_key, dummy_input)
        model_params = initial_vars["params"]  # Extract params from the variables dict
        opt_state = optimizer.init(model_params)

    batch_size = args.batch_size
    seq_length = args.context_length

    # Prepare training data
    print("Preparing training data...")
    X = corpus_tokens[:-1]
    Y = corpus_tokens[1:]

    # Create a function to generate batches
    def get_batch(key, batch_size, seq_length):
        """Generate a random batch of data"""
        # Generate random starting indices
        start_idxs = random.randint(
            key, shape=(batch_size,), minval=0, maxval=len(X) - seq_length
        )

        # Create batches
        batch_X = jnp.stack([X[i : i + seq_length] for i in start_idxs])
        batch_Y = jnp.stack([Y[i : i + seq_length] for i in start_idxs])

        return batch_X, batch_Y

    total_sequences = (len(X) - seq_length) // batch_size
    steps_in_epoch = min(
        1000, total_sequences // batch_size
    )  # Reasonable steps per epoch
    print(f"Total training sequences: {total_sequences}")

    print("\nTraining parameters:")
    print(f"  Sequence length: {seq_length}")
    print(f"  Total tokens: {len(corpus_tokens)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Model dimensions: {args.embedding_dim}")
    print(f"  Number of heads: {args.n_heads}")
    print(f"  Number of layers: {args.n_layers}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Steps/Epoch: {steps_in_epoch}")

    # Training loop
    smooth_loss = jnp.array(-math.log(1.0 / vocab_size), dtype=jnp.float32)
    start_time = time.time()

    global_step = 0
    global_epoch = 0
    for x in model_config["epochs"]:
        global_step += x["steps"]
        global_epoch = x["epoch"]
        smooth_loss = x["smooth_loss"]
    global_epoch += 1

    for epoch in range(args.epochs):
        epoch_loss = 0

        for step in range(steps_in_epoch):
            # Generate random batch
            data_key, subkey = random.split(data_key)
            batch_tokens, batch_targets = get_batch(subkey, batch_size, seq_length)

            # JAX training step - FIXED: removed unnecessary parameters
            model_params, opt_state, loss_v, current_lr = train_step(
                model_params,
                opt_state,
                batch_tokens,
                batch_targets,
                model,
                optimizer,
                args.learning_rate,
            )

            loss_py = loss_v.item()
            epoch_loss += loss_py
            smooth_loss = smooth_loss * 0.999 + loss_v * 0.001
            global_step += 1

            # Progress reporting
            if global_step >= 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (batch_size * seq_length * global_step) / elapsed

                sys.stdout.write(
                    f"\rEpoch {epoch + 1}/{args.epochs}, Step {step}/{steps_in_epoch} | "
                    f"Loss: {loss_py:.4f} | Smooth: {smooth_loss.item():.4f} | "
                    f"LR: {current_lr:.6f} | TPS: {tokens_per_sec:.0f}"
                )
                if global_step % 20 == 0:
                    generation_output = generate_during_training(
                        model=model,
                        params=model_params,
                        tokenizer=tokenizer,
                        tokenizer_path=tokenizer_path,
                        step_count=global_step - 1,
                        temperature=0.8,
                    )
                    sys.stdout.write(generation_output)
                sys.stdout.flush()

        # End of epoch
        avg_epoch_loss = epoch_loss / steps_in_epoch
        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        save_jax_model(model_params, args.model_path)
        model_config["epochs"].append(
            {
                "epoch": global_epoch + epoch,
                "steps": steps_in_epoch,
                "smooth_loss": float(smooth_loss),
                "learning_rate": float(current_lr),
                "batch_size": batch_size,
            }
        )
        save_model_config(model_config, args.model_path)
        print(f"Checkpoint saved to {args.model_path}")

    print("Training completed!")
    save_jax_model(model_params, args.model_path)


def generate_sample_jax(
    model,
    params,
    tokenizer,
    tokenizer_path,
    prompt,
    max_length=100,
    temperature=1.0,
    top_k=None,
):
    """
    Generate text sample using the trained JAX model.
    """
    # Encode prompt
    prompt_tokens = jnp.array(tokenizer.encode(prompt, tokenizer_path), dtype=jnp.int32)

    # Generate tokens
    generated_tokens = generate(
        params=params,
        model=model,
        prompt_tokens=prompt_tokens,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
    )

    # Decode back to text
    generated_text = tokenizer.decode(generated_tokens.tolist())
    return generated_text


def generate_next(
    tokenizer, tokenizer_path: str, params, model, temperature, prompt: str
):
    # Encode prompt
    prompt_tokens_list = tokenizer.encode(prompt, tokenizer_path)

    # Limit prompt length to avoid exceeding context
    max_prompt_len = min(len(prompt_tokens_list), 20)
    prompt_tokens_list = prompt_tokens_list[:max_prompt_len]

    prompt_tokens = jnp.array(prompt_tokens_list, dtype=jnp.int32)

    # Generate
    generated_tokens = generate(
        params=params,
        model=model,
        prompt_tokens=prompt_tokens,
        max_length=min(10, model.max_seq_len),
        temperature=temperature,
        top_k=40,
    )

    return tokenizer.decode(generated_tokens.tolist())


def generate_during_training(
    model, params, tokenizer, tokenizer_path, step_count, temperature=0.8
):
    """Generate a sample during training."""
    prompts = [
        "The future of",
        "Once upon a time",
        "In a world where",
        "The secret to",
    ]

    # Use a different prompt each time
    import random

    prompt = prompts[random.randint(0, len(prompts) - 1)]

    try:
        generated_text = generate_next(
            tokenizer, tokenizer_path, params, model, temperature, prompt
        )

        return f"\nStep {step_count} | Prompt: '{prompt}' → '{generated_text}'\n"

    except Exception as e:
        return f"\nGeneration failed at step {step_count}: {e}\n"


def interactive_generation(args):
    """
    Interactive generation loop.
    """
    # Load tokenizer
    tokenizer, tokenizer_path = load_tokenizer(args.model_path)

    model_config = load_model_config(args.model_path, tokenizer.vocab_size, args)
    model = JAXTransformer(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        max_seq_len=model_config["max_seq_len"],
    )

    # Load trained parameters
    params = load_jax_model(args.model_path)
    if params is None:
        print("No trained model found!")
        return

    print("Model loaded.")

    generated_text = generate_next(
        tokenizer, tokenizer_path, params, model, 0.8, args.prompt
    )

    print(f"Generated: {generated_text}")
