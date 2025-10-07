#!/usr/bin/env python3
import argparse
import importlib
import json
import math
import os
import sys
import time

import numpy as np

# --- CUSTOM TOKENIZER IMPORTS ---
from bpe_tokenizer_fast import BPETokenizer
from char_tokenizer import CharTokenizer

TOKENIZER_CLASSES = {"char": CharTokenizer, "bpe": BPETokenizer}


# ==============================================================================
# DATA PREPARATION
# ==============================================================================
def load_corpus(filepath: str) -> str:
    """Reads the entire text content from a file."""
    print(f"INFO: Loading corpus from '{filepath}'...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found at '{filepath}'", file=sys.stderr)
        sys.exit(1)


def save_model_config(model_config: dict, output_dir: str):
    """Saves model configuration for inspection and reuse."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"INFO: Saved model configuration to '{config_path}'")


# ==============================================================================
# SUB-COMMAND FUNCTIONS
# ==============================================================================


def train_tokenizer_command(args):
    """Trains a new tokenizer from a text file."""
    if args.tokenizer == "char":
        print(
            "INFO: The 'char' tokenizer does not require a separate training step.",
            file=sys.stderr,
        )
        print(
            "It is built automatically during model training if a vocab file is not found.",
            file=sys.stderr,
        )
        return

    corpus = load_corpus(args.input_file)
    tokenizer = TOKENIZER_CLASSES[args.tokenizer]()

    print(f"\n--- Training {args.tokenizer.upper()} Tokenizer ---")
    print(f"Target vocabulary size: {args.vocab_size}")

    # Pass vocab_size for BPE training
    tokenizer.train(
        corpus, vocab_size=args.vocab_size, verbose=True, token_file=args.tokenizer_path
    )


def train_command(args, rnn_math):
    """Main model training function."""
    corpus = load_corpus(args.input_file)
    output_dir = os.path.dirname(args.model_path)

    # --- Dynamically load or train the selected tokenizer ---
    try:
        tokenizer = TOKENIZER_CLASSES[args.tokenizer].load(args.tokenizer_path)
    except FileNotFoundError:
        if args.tokenizer == "char":
            print(
                f"INFO: Tokenizer file not found at '{args.tokenizer_path}'. Training a new 'char' tokenizer from the corpus."
            )
            tokenizer = CharTokenizer()
            tokenizer.train(corpus, args.tokenizer_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        else:
            print(
                f"ERROR: Tokenizer file not found at '{args.tokenizer_path}'.",
                file=sys.stderr,
            )
            print(
                f"Please train a '{args.tokenizer}' tokenizer first using the 'train-tokenizer' command.",
                file=sys.stderr,
            )
            sys.exit(1)

    vocab_size = tokenizer.vocab_size
    model_config = {
        "vocab_size": vocab_size,
        "hidden_size": args.hidden_size,
        "embedding_dim": args.embedding_dim,
        "context_length": args.context_length,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    }
    save_model_config(model_config, output_dir)

    # --- LOGIC TO RESUME OR CREATE A NEW MODEL ---
    if args.resume and os.path.exists(args.model_path):
        print(f"\n--- Resuming Training from '{args.model_path}' ---")
        model = rnn_math.SimpleRNN.load_model(args.model_path)
        # Verify that the loaded model config matches the current settings
        if (
            model.vocab_size != vocab_size
            or model.hidden_size != args.hidden_size
            or model.embedding_dim != args.embedding_dim
        ):
            print(
                "WARNING: Model architecture in the saved file does not match current settings.",
                file=sys.stderr,
            )
            print(
                "         This may lead to errors or unexpected behavior.",
                file=sys.stderr,
            )
    else:
        if args.resume:
            print(
                f"INFO: --resume flag was given, but no model found at '{args.model_path}'."
            )
        print("\n--- Starting New Model Training ---")
        model = rnn_math.SimpleRNN(vocab_size, args.hidden_size, args.embedding_dim)
    vectorized_corpus = np.array(
        tokenizer.encode(corpus, token_file=args.tokenizer_path), dtype=np.uint32
    )

    corpus_gpu = model.device.create_buffer_with_data(
        data=vectorized_corpus, usage=rnn_math.STORAGE_BUFFER_USAGE
    )
    data_size = len(vectorized_corpus)

    print(f"Using math backend: {args.backend}_math")
    print(f"Using tokenizer: {args.tokenizer}")
    print(f"Corpus tokenized into {data_size} tokens.")

    smooth_loss = -math.log(1.0 / vocab_size)
    h_state_gpu = model.get_initial_hidden_state_gpu()
    last_log_time = time.time()
    steps_since_log = 0

    for epoch in range(args.epochs):
        model.zero_buffer(h_state_gpu)
        p = 0
        steps_in_epoch = (data_size - 1) // args.context_length
        for step in range(steps_in_epoch):
            if p + args.context_length + 1 >= data_size:
                break
            model.reset_gradients_gpu(model.gradient_buffers)
            target_idx = vectorized_corpus[p + args.context_length]
            logits_gpu, h_history_gpu = model.forward_sequence(
                corpus_gpu, h_state_gpu, p, args.context_length
            )
            model.backward_sequence(
                corpus_gpu,
                p,
                args.context_length,
                target_idx,
                logits_gpu,
                h_history_gpu,
            )

            model.update_weights(args.learning_rate)
            model.update_hidden_state(h_source=h_history_gpu, h_dest=h_state_gpu)
            p += args.context_length
            steps_since_log += 1
            current_time = time.time()
            if current_time - last_log_time >= 1.0:
                # single token loss
                # loss = model.calculate_loss_gpu(logits_gpu, target_idx)

                # Calculate average loss over the entire sequence
                loss = model.calculate_sequence_loss(
                    corpus_gpu, h_state_gpu, p, args.context_length
                )
                smooth_loss = smooth_loss * 0.999 + loss * 0.001

                elapsed_time = current_time - last_log_time
                steps_per_second = steps_since_log / elapsed_time
                progress = (
                    f"Epoch {epoch + 1}/{args.epochs}, Step {step}/{steps_in_epoch}"
                )
                sys.stdout.write(
                    f"\r{progress} | Smooth Loss: {smooth_loss:.4f} | SPS: {steps_per_second:.2f}   "
                )
                sys.stdout.flush()
                last_log_time = current_time
                steps_since_log = 0

    print("\n--- Training Finished ---")
    model.save_model(args.model_path)


def generate_command(args, rnn_math):
    """Main generation function."""
    try:
        tokenizer = TOKENIZER_CLASSES[args.tokenizer].load(args.tokenizer_path)
    except FileNotFoundError:
        print(
            f"ERROR: Tokenizer file not found at '{args.tokenizer_path}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = rnn_math.SimpleRNN.load_model(args.model_path)
    print(f"--- Generating {args.length} tokens from seed: '{args.prompt}' ---")

    h_gpu = model.get_initial_hidden_state_gpu()
    input_ids = tokenizer.encode(args.prompt, args.tokenizer_path)
    for token_id in input_ids:
        model.forward_step(token_id, h_gpu)

    input_idx = input_ids[-1] if input_ids else 0
    generated_ids = list(input_ids)
    for _ in range(args.length):
        next_idx = model.generate_step(input_idx, h_gpu)
        generated_ids.append(next_idx)
        input_idx = next_idx

    generated_text = tokenizer.decode(generated_ids)
    print("\n" + generated_text)


# ==============================================================================
# CLI ORCHESTRATION
# ==============================================================================


def main():
    # Main parser
    parser = argparse.ArgumentParser(
        description="A pedagogical, modular RNN for text generation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Global arguments applicable to all sub-commands
    parser.add_argument(
        "--backend",
        choices=["pure", "numpy", "wgpu"],
        default="numpy",
        help="The math backend to use for computation.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["char", "bpe"],
        default="bpe",
        help="The type of tokenizer to use.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- "train-tokenizer" command ---
    tt_parser = subparsers.add_parser(
        "train-tokenizer", help="Train a BPE tokenizer from a text file."
    )
    tt_parser.add_argument(
        "input_file", type=str, help="Path to the training text file."
    )
    tt_parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="out/tokenizer.json",
        help="Path to save the trained tokenizer.",
    )
    tt_parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Target vocabulary size (for BPE).",
    )
    tt_parser.set_defaults(func=train_tokenizer_command)

    # --- "train" command ---
    train_parser = subparsers.add_parser(
        "train", help="Train a new model on a text file."
    )
    train_parser.add_argument(
        "input_file", type=str, help="Path to the training text file."
    )
    train_parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="out/tokenizer.json",
        help="Path to the tokenizer file.",
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default="out/model.json",
        help="Path to save the trained model.",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs."
    )
    train_parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Number of neurons in the hidden layer.",
    )
    train_parser.add_argument(
        "--embedding-dim", type=int, default=128, help="Dimension of token embeddings."
    )
    train_parser.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Length of the sequence for backpropagation.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    # --- THIS IS THE NEW ARGUMENT ---
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the model specified in --model-path.",
    )

    # --- "generate" command ---
    gen_parser = subparsers.add_parser(
        "generate", help="Generate text using a trained model."
    )
    gen_parser.add_argument(
        "prompt", type=str, help="The initial string to start generation."
    )
    gen_parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="out/tokenizer.json",
        help="Path to the tokenizer file.",
    )
    gen_parser.add_argument(
        "--model-path",
        type=str,
        default="out/model.json",
        help="Path to the trained model file.",
    )
    gen_parser.add_argument(
        "--length", type=int, default=100, help="Number of new tokens to generate."
    )

    args = parser.parse_args()

    # --- DYNAMICALLY LOAD THE MATH BACKEND ---
    try:
        backend_module_name = f"{args.backend}_math"
        rnn_math = importlib.import_module(backend_module_name)
    except ImportError:
        print(
            f"ERROR: Could not import the math backend '{backend_module_name}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- ROUTE TO THE CORRECT SUB-COMMAND FUNCTION ---
    if args.command == "train-tokenizer":
        train_tokenizer_command(args)
    elif args.command == "train":
        train_command(args, rnn_math)
    elif args.command == "generate":
        generate_command(args, rnn_math)


if __name__ == "__main__":
    main()
