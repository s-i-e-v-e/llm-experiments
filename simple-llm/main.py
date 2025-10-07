import argparse

from jax_train import train_command_jax

if __name__ == "__main__":
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
        train_command_jax(args)
