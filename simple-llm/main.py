#!/usr/bin/env python3
import json
import math
import os
import sys
import time

import numpy as np

from bpe_tokenizer_fast import BPETokenizer
from transformer import numpy_math


def load_corpus(file_path):
    """Load text corpus from file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_model_config(config, output_dir):
    """Save model configuration"""
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def train_command(args):
    """Main training function for transformer"""

    # Create output directory
    os.makedirs("out", exist_ok=True)

    # Load or train tokenizer
    corpus_text = load_corpus(args.input_file)
    print(f"Loaded corpus: {len(corpus_text)} characters")

    tokenizer = BPETokenizer()

    if os.path.exists(args.tokenizer_path):
        print(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = BPETokenizer.load(args.tokenizer_path)
    else:
        print(f"Training new BPE tokenizer with vocab_size {args.vocab_size}")
        tokenizer.train(corpus_text, args.vocab_size, args.tokenizer_path, verbose=True)

    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {vocab_size}")

    # Encode corpus
    print("Encoding corpus...")
    corpus_tokens = np.array(
        tokenizer.encode(corpus_text, args.tokenizer_path), dtype=np.uint32
    )
    print(f"Tokenized corpus: {len(corpus_tokens)} tokens")

    # Model configuration
    model_config = {
        "vocab_size": vocab_size,
        "d_model": args.embedding_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "max_seq_len": args.context_length,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    }

    save_model_config(model_config, "out")

    # Initialize model
    if args.resume and os.path.exists(args.model_path):
        print(f"Resuming training from {args.model_path}")
        model = numpy_math.NumpyTransformer.load_model(args.model_path)
    else:
        print("Initializing new transformer model")
        model = numpy_math.NumpyTransformer(
            vocab_size=vocab_size,
            d_model=args.embedding_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_seq_len=args.context_length,
        )

    batch_size = args.batch_size  # e.g., 32, 64, 128

    # Training parameters
    data_size = len(corpus_tokens)
    seq_length = args.context_length
    total_steps = (data_size - 1) // seq_length // args.batch_size
    steps_per_epoch = total_steps // args.epochs if args.epochs > 0 else total_steps

    warmup_steps = 1000
    max_lr = args.learning_rate

    print("\nTraining parameters:")
    print(f"  Sequence length: {seq_length}")
    print(f"  Total tokens: {data_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Model dimensions: {args.embedding_dim}")
    print(f"  Number of heads: {args.n_heads}")
    print(f"  Number of layers: {args.n_layers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")

    # Initialize smooth loss
    smooth_loss = -math.log(1.0 / vocab_size)
    start_time = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0
        steps_in_epoch = 0

        for step in range(steps_per_epoch):
            # Create batch of sequences
            batch_tokens = []
            batch_targets = []

            for i in range(batch_size):
                offset = (step * batch_size + i) * seq_length

                if offset + seq_length + 1 >= data_size:
                    # If we run out of data, wrap around or break
                    offset = (i * seq_length) % (data_size - seq_length - 1)

                tokens = corpus_tokens[offset : offset + seq_length]
                targets = corpus_tokens[offset + 1 : offset + seq_length + 1]

                batch_tokens.append(tokens)
                batch_targets.append(targets)

            # Convert to numpy arrays
            batch_tokens = np.array(
                batch_tokens, dtype=np.uint32
            )  # [batch_size, seq_len]
            batch_targets = np.array(
                batch_targets, dtype=np.uint32
            )  # [batch_size, seq_len]

            # Forward pass (now with batch)
            logits = model.forward(batch_tokens)  # [batch_size, seq_len, vocab_size]

            # Calculate loss (averaged over batch and sequence)
            loss = model.calculate_loss(logits, batch_targets)
            epoch_loss += loss
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # Backward pass
            d_logits = model.get_loss_gradient(logits, batch_targets)
            model.backward(d_logits)

            # Learning rate warmup
            current_lr = max_lr * min(1.0, (global_step + 1) / warmup_steps)

            # Update weights
            model.update_weights(current_lr)

            steps_in_epoch += 1
            global_step += 1

            # Progress reporting
            if global_step % 2 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed if elapsed > 0 else 0
                tokens_per_sec = (batch_size * seq_length * global_step) / elapsed

                sys.stdout.write(
                    f"\rEpoch {epoch + 1}/{args.epochs}, Step {global_step} | "
                    f"Loss: {loss:.4f} | Smooth: {smooth_loss:.4f} | "
                    f"LR: {current_lr:.6f} | TPS: {tokens_per_sec:.0f}"
                )
                sys.stdout.flush()

        # End of epoch
        avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save checkpoint and generate sample
        model.save_model(args.model_path)
        generate_sample(model, tokenizer, "The mystery", length=50)

    print("Training completed!")
    model.save_model(args.model_path)


def generate_sample(model, tokenizer, prompt_text, length=100):
    """Generate text sample from model"""
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt_text)
    current_tokens = np.array(prompt_tokens, dtype=np.uint32)

    generated_tokens = []

    for i in range(length):
        # Ensure we don't exceed max sequence length
        if len(current_tokens) > model.max_seq_len:
            current_tokens = current_tokens[-model.max_seq_len :]

        # Forward pass
        batch_tokens = current_tokens.reshape(1, -1)
        logits = model.forward(batch_tokens)

        # Get next token probabilities (last position)
        next_logits = logits[0, -1, :]
        probs = softmax(next_logits)

        # Sample from distribution (you can use temperature sampling here)
        next_token = np.random.choice(model.vocab_size, p=probs)
        generated_tokens.append(next_token)
        current_tokens = np.append(current_tokens, next_token)

    # Decode tokens back to text
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


def softmax(x):
    """Softmax function"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Transformer Model")
    parser.add_argument("--backend", default="numpy", help="Backend to use")
    parser.add_argument("train", action="store_true", help="Training mode")
    parser.add_argument("input_file", help="Input text file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--embedding-dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument(
        "--context-length", type=int, default=256, help="Context length"
    )
    parser.add_argument(
        "--n-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--n-layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=5000, help="Vocabulary size for BPE"
    )
    parser.add_argument(
        "--model-path", default="out/model.json", help="Model save path"
    )
    parser.add_argument(
        "--tokenizer-path", default="out/tokenizer.json", help="Tokenizer path"
    )
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    if args.train:
        train_command(args)
