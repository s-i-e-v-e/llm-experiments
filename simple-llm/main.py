#!/usr/bin/env python3
import argparse
import json
import math
import sys
import os

# --- CHOOSE YOUR BACKEND ---
# Switch the comment to change from pure Python to NumPy.
# The rest of the code works with either one.

# import pure_math as rnn_math
#import numpy_math as rnn_math
import wgpu_math as rnn_math

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
        "vocab_size": vocab_size, "hidden_size": args.hidden_size,
        "embedding_dim": args.embedding_dim, "context_length": args.context_length,
        "learning_rate": args.learning_rate, "epochs": args.epochs,
    }
    
    output_dir = os.path.dirname(args.model_path)
    save_debug_files(char_to_int, model_config, output_dir)
    
    model = rnn_math.SimpleRNN(vocab_size, args.hidden_size, args.embedding_dim)
    
    vectorized_corpus = [char_to_int[ch] for ch in corpus]
    data_size = len(vectorized_corpus)
    
    print("\n--- Starting Training ---")
    print(f"Using math backend: {rnn_math.__name__}")
    
    p = 0
    smooth_loss = -math.log(1.0 / vocab_size) * args.context_length
    
    for epoch in range(args.epochs):
        h_prev = model.get_initial_hidden_state()
        p = 0
        steps_in_epoch = (data_size - 1) // args.context_length
        
        for step in range(steps_in_epoch):
            if p + args.context_length + 1 >= data_size: break
            
            inputs = vectorized_corpus[p : p + args.context_length]
            targets = vectorized_corpus[p + 1 : p + args.context_length + 1]
            
            # --- Forward, Backward, Update ---
            logits, h_prev, cache = model.forward_pass(inputs, h_prev)
            
            probs = rnn_math.softmax(logits)
            loss = -math.log(probs[targets[-1]]) # Loss on the last character
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            
            grads = model.backward_pass(inputs, targets[-1], cache)
            model.update_weights(grads, args.learning_rate)
            
            # Logging progress
            progress_bar = f"Epoch {epoch+1}/{args.epochs}, Step {step}/{steps_in_epoch}"
            loss_info = f"Smooth Loss: {smooth_loss:.4f}"
            sys.stdout.write(f"\r{progress_bar} | {loss_info}")
            sys.stdout.flush()

            p += args.context_length
            
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
        with open(vocab_path, 'r') as f: char_to_int = json.load(f)
        int_to_char = {i: ch for ch, i in char_to_int.items()}
    except FileNotFoundError:
        print(f"ERROR: vocab.json not found in '{model_dir}'. Train a model first.", file=sys.stderr)
        sys.exit(1)
        
    model = rnn_math.SimpleRNN.load_model(args.model_path)
    
    print(f"--- Generating {args.length} characters from seed: '{args.prompt}' ---")
    
    h = model.get_initial_hidden_state()
    generated_text = args.prompt
    
    # "Warm up" the hidden state with the prompt
    for char in args.prompt:
        if char not in char_to_int:
            print(f"Warning: Character '{char}' not in vocabulary. Skipping.", file=sys.stderr)
            continue
        char_idx = char_to_int[char]
        _, h, _ = model.forward_pass([char_idx], h)

    last_char = args.prompt[-1] if args.prompt and args.prompt[-1] in char_to_int else ' '
    input_idx = char_to_int.get(last_char, 0)

    for _ in range(args.length):
        logits, h, _ = model.forward_pass([input_idx], h)
        probs = rnn_math.softmax(logits)
        
        # Sample the next character index
        # Using a simple `max` for deterministic generation, or `random.choices` for variety
        # next_idx = probs.index(max(probs)) # Deterministic
        next_idx = rnn_math.sample_from_dist(probs) # Stochastic
        
        next_char = int_to_char[next_idx]
        generated_text += next_char
        input_idx = next_idx
        
    print("\n" + generated_text)

# ==============================================================================
# CLI ORCHESTRATION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="A tiny, pure-Python character-level RNN for text generation.",
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