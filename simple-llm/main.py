#!/usr/bin/env python3
import argparse
import json
import math
import sys
import os
import time
import numpy as np

# --- CHOOSE YOUR BACKEND ---
import pure_math as rnn_math
#import numpy_math as rnn_math
#import wgpu_math as rnn_math

# ==============================================================================
# DATA PREPARATION & VOCABULARY
# ==============================================================================

def load_corpus(filepath: str) -> str:
    """Reads the entire text content from a file."""
    print(f"INFO: Loading corpus from '{filepath}'...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found at '{filepath}'", file=sys.stderr)
        sys.exit(1)


def create_vocabulary(corpus: str) -> tuple[dict, dict]:
    """Creates character-to-index and index-to-character mappings."""
    chars = sorted(list(set(corpus)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    print(f"INFO: Created vocabulary with {len(chars)} unique characters.")
    return char_to_int, int_to_char


def save_debug_files(char_to_int: dict, model_config: dict, output_dir: str):
    """Saves vocabulary and model configuration for inspection and reuse."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, 'w') as f:
        json.dump(char_to_int, f, indent=2)
    print(f"INFO: Saved vocabulary mapping to '{vocab_path}'")

    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"INFO: Saved model configuration to '{config_path}'")


# ==============================================================================
# TRAINING PROCESS
# ==============================================================================

def train(args):
    """Main training function."""
    corpus = load_corpus(args.input_file)
    char_to_int, _ = create_vocabulary(corpus)
    vocab_size = len(char_to_int)

    model_config = {
        "vocab_size": vocab_size,
        "hidden_size": args.hidden_size,
        "embedding_dim": args.embedding_dim,
        "context_length": args.context_length,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    }

    output_dir = os.path.dirname(args.model_path)
    save_debug_files(char_to_int, model_config, output_dir)

    model = rnn_math.SimpleRNN(vocab_size, args.hidden_size, args.embedding_dim)

    vectorized_corpus = np.array([char_to_int[ch] for ch in corpus], dtype=np.uint32)
    corpus_gpu = model.device.create_buffer_with_data(
        data=vectorized_corpus, usage=rnn_math.STORAGE_BUFFER_USAGE
    )
    data_size = len(vectorized_corpus)

    print("\n--- Starting Training ---")
    print(f"Using math backend: {rnn_math.__name__}")

    smooth_loss = -math.log(1.0 / vocab_size) * args.context_length
    h_state_gpu = model.get_initial_hidden_state_gpu()

    # Performance monitoring
    last_log_time = time.time()
    steps_since_log = 0

    for epoch in range(args.epochs):
        model.zero_buffer(h_state_gpu)  # Reset hidden state at the start of each epoch
        p = 0
        steps_in_epoch = (data_size - 1) // args.context_length

        for step in range(steps_in_epoch):
            if p + args.context_length + 1 >= data_size:
                break

            target_idx = vectorized_corpus[p + args.context_length]

            # --- Forward, Backward, Update ---
            # These operations are now sent to the GPU without blocking the CPU
            logits_gpu, h_history_gpu = model.forward_sequence(
                corpus_gpu, h_state_gpu, p, args.context_length
            )

            model.backward_sequence(
                corpus_gpu, p, args.context_length, target_idx, logits_gpu, h_history_gpu
            )
            model.update_weights(args.learning_rate)
            model.update_hidden_state(h_source=h_history_gpu, h_dest=h_state_gpu)

            p += args.context_length
            steps_since_log += 1

            # Log performance and loss periodically instead of every step
            # This avoids frequent, slow CPU-GPU synchronization.
            current_time = time.time()
            if current_time - last_log_time >= 1.0:
                # This is our only sync-point, where we read back the loss from the GPU.
                loss = model.calculate_loss_gpu(logits_gpu, target_idx)
                smooth_loss = smooth_loss * 0.999 + loss * 0.001

                elapsed_time = current_time - last_log_time
                steps_per_second = steps_since_log / elapsed_time

                # Logging progress
                progress_bar = f"Epoch {epoch + 1}/{args.epochs}, Step {step}/{steps_in_epoch}"
                loss_info = f"Smooth Loss: {smooth_loss:.4f}"
                sps_info = f"SPS: {steps_per_second:.2f}"
                sys.stdout.write(f"\r{progress_bar} | {loss_info} | {sps_info}   ")
                sys.stdout.flush()

                last_log_time = current_time
                steps_since_log = 0

    print("\n--- Training Finished ---")
    model.save_model(args.model_path)


# ==============================================================================
# INFERENCE & GENERATION
# ==============================================================================

def generate(args):
    """Main generation function."""
    model_dir = os.path.dirname(args.model_path)
    vocab_path = os.path.join(model_dir, "vocab.json")

    try:
        with open(vocab_path, 'r') as f:
            char_to_int = json.load(f)
        int_to_char = {i: str(ch) for ch, i in char_to_int.items()}
        # Ensure keys are integers for direct lookup
        int_to_char = {int(k): v for k, v in int_to_char.items()}
    except FileNotFoundError:
        print(f"ERROR: vocab.json not found in '{model_dir}'. Train a model first.", file=sys.stderr)
        sys.exit(1)

    model = rnn_math.SimpleRNN.load_model(args.model_path)

    print(f"--- Generating {args.length} characters from seed: '{args.prompt}' ---")

    h_gpu = model.get_initial_hidden_state_gpu()
    generated_text = args.prompt

    # "Warm up" the hidden state with the prompt on the GPU
    # This loop is now much more efficient as forward_step reuses GPU resources.
    for char in args.prompt:
        if char not in char_to_int:
            print(f"Warning: Character '{char}' not in vocabulary. Skipping.", file=sys.stderr)
            continue
        char_idx = char_to_int[char]
        model.forward_step(char_idx, h_gpu)

    # Use the last character of the prompt to start generation
    last_char = args.prompt[-1] if args.prompt and args.prompt[-1] in char_to_int else ' '
    input_idx = char_to_int.get(last_char, 0)

    for _ in range(args.length):
        # The entire generation step (forward, softmax, sample) happens on the GPU
        # in a single, efficient command buffer submission.
        next_idx = model.generate_step(input_idx, h_gpu)

        next_char = int_to_char[next_idx]
        generated_text += next_char
        input_idx = next_idx

    print("\n" + generated_text)


# ==============================================================================
# CLI ORCHESTRATION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="A tiny, wgpu-accelerated character-level RNN for text generation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Training Sub-command
    train_parser = subparsers.add_parser("train", help="Train a new model on a text file.")
    train_parser.add_argument("input_file", type=str, help="Path to the training text file (e.g., sherlock.txt).")
    train_parser.add_argument("--model-path", type=str, default="out/model.json", help="Path to save the trained model.")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    train_parser.add_argument("--hidden-size", type=int, default=100, help="Number of neurons in the hidden layer.")
    train_parser.add_argument("--embedding-dim", type=int, default=30, help="Dimension of character embeddings.")
    train_parser.add_argument("--context-length", type=int, default=25, help="Length of the sequence for backpropagation.")
    train_parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    train_parser.set_defaults(func=train)

    # Generation Sub-command
    gen_parser = subparsers.add_parser("generate", help="Generate text using a trained model.")
    gen_parser.add_argument("prompt", type=str, help="The initial string to start generation.")
    gen_parser.add_argument("--model-path", type=str, default="out/model.json", help="Path to the trained model file.")
    gen_parser.add_argument("--length", type=int, default=500, help="Number of characters to generate.")
    gen_parser.set_defaults(func=generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()