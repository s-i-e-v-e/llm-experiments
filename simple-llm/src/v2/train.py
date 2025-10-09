import math
import os
import time

from common.util import get_model_weights_path, load_corpus, load_tokenizer


def train_command(args):
    os.makedirs(args.model_path, exist_ok=True)

    # STEP 1: Load and preprocess corpus
    corpus_text = load_corpus(args.input_file)
    print(f"Loaded corpus: {len(corpus_text)} characters")

    # STEP 2: Load or train tokenizer
    tokenizer, tokenizer_path = load_tokenizer(
        args.model_path, args.vocab_size, corpus_text
    )
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # STEP 3: Tokenize corpus
    input_tokens = tokenizer.encode(corpus_text, tokenizer_path)
    print(f"Tokenized corpus: {len(input_tokens)} tokens")

    # STEP 4: Prepare train data (split into sequences, batching)
    # For language modeling, create (input, target) pairs with context_length stride.
    sequences = []
    for i in range(0, len(input_tokens) - args.context_length, args.context_length):
        seq = input_tokens[i : i + args.context_length + 1]
        if len(seq) == args.context_length + 1:
            input_seq = seq[:-1]
            target_seq = seq[1:]
            sequences.append((input_seq, target_seq))

    steps_in_epoch = math.ceil(len(sequences) / args.batch_size)
    print(f"Total training sequences: {len(sequences)}")

    # STEP 5: Backend/model selection
    if args.backend == "jax":
        from v2.jax_backend import (
            initialize_model,
            load_model,
            save_model,
            to_backend,
            update_params,
        )
    elif args.backend == "numpy":
        raise NotImplementedError(f"Not implemented {args.backend}")
    elif args.backend == "python":
        raise NotImplementedError(f"Not implemented {args.backend}")
    elif args.backend == "wgpu":
        raise NotImplementedError(f"Not implemented {args.backend}")
    else:
        raise ValueError(f"Unknown backend {args.backend}")

    # LOG
    print("\nTraining parameters:")
    print(f"  Context length: {args.context_length}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Model dimensions: {args.embedding_dim}")
    print(f"  Number of heads: {args.n_heads}")
    print(f"  Number of layers: {args.n_layers}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps/Epoch: {steps_in_epoch}")

    # STEP 6: Initialize model and optimizer
    model = initialize_model(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        context_length=args.context_length,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        learning_rate=args.learning_rate,
    )

    if args.resume and os.path.exists(get_model_weights_path(args.model_path)):
        print("Resuming training")
        model = load_model(args.learning_rate, args.model_path)

    # STEP 7: Training loop
    global_step = 0
    global_epoch = 0
    smooth_loss = -math.log(1.0 / tokenizer.vocab_size)
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0
        step = 0
        for batch_start in range(0, len(sequences), args.batch_size):
            batch = sequences[batch_start : batch_start + args.batch_size]
            batch_inputs = [seq[0] for seq in batch]
            batch_targets = [seq[1] for seq in batch]

            batch_inputs_backend = to_backend(batch_inputs)
            batch_targets_backend = to_backend(batch_targets)

            model, loss = update_params(
                model, batch_inputs_backend, batch_targets_backend
            )

            epoch_loss += loss
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            step += 1
            global_step += 1

            if global_step >= 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (
                    args.batch_size * args.context_length * global_step
                ) / elapsed

                print(
                    f"\rEpoch {epoch + 1}/{args.epochs}, Step {step}/{steps_in_epoch} | "
                    f"Loss: {loss:.4f} | Smooth: {smooth_loss:.4f} | "
                    f"LR: {args.learning_rate:.6f} | TPS: {tokens_per_sec:.0f}"
                )

        global_epoch += 1

        print(f"Epoch {epoch + 1} completed")

    # STEP 8: Save model weights and hyperparams
    save_model(model, args.model_path)
