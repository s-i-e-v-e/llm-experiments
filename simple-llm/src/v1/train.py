import math
import os
import sys
import time

from common.util import load_corpus, load_tokenizer
from v1.jax_backend import (
    int_array,
    load_model_config,
    main_train_step,
    model_init,
    model_save,
    save_model_config,
)


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
    input_tokens = int_array(tokenizer.encode(corpus_text, tokenizer_path))
    print(f"Tokenized corpus: {len(input_tokens)} tokens")

    batch_size = args.batch_size
    seq_length = args.context_length

    # Prepare training data
    print("Preparing training data...")
    X = input_tokens[:-1]
    Y = input_tokens[1:]

    total_sequences = (len(X) - seq_length) // batch_size
    steps_in_epoch = math.ceil(total_sequences / args.batch_size)
    print(f"Total training sequences: {total_sequences}")

    # LOG
    print("\nTraining parameters:")
    print(f"  Sequence length: {seq_length}")
    print(f"  Total tokens: {len(input_tokens)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Model dimensions: {args.embedding_dim}")
    print(f"  Number of heads: {args.n_heads}")
    print(f"  Number of layers: {args.n_layers}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Steps/Epoch: {steps_in_epoch}")

    # STEP 4:
    model_config = load_model_config(args.model_path, tokenizer.vocab_size, args)
    data_key, x_model, opt_state, model_params = model_init(
        args.model_path,
        model_config,
        args.resume,
        args.learning_rate,
        args.context_length,
    )
    smooth_loss = -math.log(1.0 / tokenizer.vocab_size)
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
            data_key, model_params, opt_state, loss = main_train_step(
                data_key, X, Y, batch_size, seq_length, model_params, x_model, opt_state
            )

            epoch_loss += loss
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            global_step += 1

            # Progress reporting
            if global_step >= 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (batch_size * seq_length * global_step) / elapsed

                sys.stdout.write(
                    f"\rEpoch {epoch + 1}/{args.epochs}, Step {step}/{steps_in_epoch} | "
                    f"Loss: {loss:.4f} | Smooth: {smooth_loss:.4f} | "
                    f"LR: {args.learning_rate:.6f} | TPS: {tokens_per_sec:.0f}"
                )
                if global_step % 20 == 0:
                    generation_output = generate_during_training(
                        model=x_model.model,
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
        model_save(model_params, args.model_path)
        model_config["epochs"].append(
            {
                "epoch": global_epoch + epoch,
                "steps": steps_in_epoch,
                "smooth_loss": smooth_loss,
                "learning_rate": args.learning_rate,
                "batch_size": batch_size,
            }
        )
        save_model_config(model_config, args.model_path)
        print(f"Checkpoint saved to {args.model_path}")

    print("Training completed!")
    model_save(model_params, args.model_path)
