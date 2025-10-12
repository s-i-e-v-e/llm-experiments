import math
import os
import random
import time

from common.util import (
    get_model_weights_path,
    load_corpus,
    load_tokenizer,
)
from generate import generate_text
from hyper import get_hyperparams_auto


def train_command(args):
    os.makedirs(args.model_path, exist_ok=True)

    # STEP 1: Load and preprocess corpus
    corpus_text = load_corpus(args.input_file)
    hyper = get_hyperparams_auto(args, len(corpus_text), 4.0)
    print(f"Loaded corpus: {len(corpus_text)} characters")

    # STEP 2: Load or train tokenizer
    tokenizer, tokenizer_path = load_tokenizer(
        args.model_path, hyper.vocab_size, corpus_text
    )
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # STEP 3: Tokenize corpus
    input_tokens = tokenizer.encode(corpus_text, tokenizer_path)
    print(f"Tokenized corpus: {len(input_tokens)} tokens")

    # STEP 5: Backend selection
    if args.backend == "jax":
        from jax_backend import (
            initialize_model,
            load_model,
            save_model,
            train_epoch,
        )
    elif args.backend == "numpy":
        raise NotImplementedError(f"Not implemented {args.backend}")
    elif args.backend == "python":
        raise NotImplementedError(f"Not implemented {args.backend}")
    elif args.backend == "wgpu":
        from wgpu_backend import (
            initialize_model,
            load_model,
            save_model,
            train_epoch,
        )
    else:
        raise ValueError(f"Unknown backend {args.backend}")

    # LOG
    print("\nTraining parameters:")
    print(f"  Context length: {hyper.context_size}")
    print(f"  Batch size: {hyper.batch_size}")
    print(f"  Model dimensions: {hyper.embedding_dim}")
    print(f"  Number of heads: {hyper.n_heads}")
    print(f"  Number of layers: {hyper.n_layers}")
    print(f"  Learning rate: {hyper.learning_rate}")
    print(f"  Epochs: {args.epochs}")

    # STEP 6: Initialize model
    model = initialize_model(
        hp=hyper,
        epochs=[],
        total_steps=args.epochs * 1000,
    )

    if os.path.exists(get_model_weights_path(args.model_path)):
        print("Resuming training")
        model = load_model(hyper, args.epochs * 1000, args.model_path)

    # STEP 7: Training loop (backend-agnostic)
    global_step = 0
    global_epoch = len(model.tm_params.epochs)
    smooth_loss = -math.log(1.0 / tokenizer.vocab_size)
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Backend handles ALL optimization internally
        model, epoch_metrics = train_epoch(
            model=model,
            token_data=input_tokens,
            batch_size=hyper.batch_size,
            context_size=hyper.context_size,
            epoch_num=epoch,
        )

        # Update metrics from backend
        global_step += epoch_metrics["steps"]
        global_epoch += 1
        smooth_loss = epoch_metrics["smooth_loss"]

        # Calculate tokens per second
        elapsed = time.time() - start_time
        tokens_per_sec = (hyper.batch_size * hyper.context_size * global_step) / elapsed

        # Log epoch results
        print(f"\nEpoch {epoch + 1} completed:")
        print(f"  Average loss: {epoch_metrics['avg_loss']:.4f}")
        print(f"  Smooth loss: {smooth_loss:.4f}")
        print(f"  Tokens/sec: {tokens_per_sec:.0f}")

        # Add epoch history
        model.tm_params.epochs.append(
            {
                "epoch": global_epoch,
                "steps": epoch_metrics["steps"],
                "smooth_loss": smooth_loss,
                "learning_rate": hyper.learning_rate,
                "batch_size": hyper.batch_size,
            }
        )

        # Save model
        save_model(model, args.model_path)

        # Generate sample
        # generation_output = generate_during_training(
        #     model=model,
        #     tokenizer=tokenizer,
        #     tokenizer_path=tokenizer_path,
        #     step_count=global_step,
        # )
        # print(generation_output)

    # STEP 8: Final save
    print(f"\nTraining completed! Model saved to {args.model_path}")
    save_model(model, args.model_path)


def generate_during_training(model, tokenizer, tokenizer_path, step_count):
    """Generate a sample during training."""
    prompts = [
        "when ",
        "he ",
        "I ",
        "the ",
        " of",
    ]

    prompt = prompts[random.randint(0, len(prompts) - 1)]
    prompt_tokens = tokenizer.encode(prompt, tokenizer_path)

    generated_tokens = generate_text(
        model=model,
        prompt_tokens=prompt_tokens,
        max_tokens=20,
    )

    generated_text = tokenizer.decode(generated_tokens)
    return f"\nStep {step_count} | Prompt: '{prompt}' → '{generated_text}'\n"
