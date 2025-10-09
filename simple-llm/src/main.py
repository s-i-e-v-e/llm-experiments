import argparse


def main(train_command, generate_command):
    parser = argparse.ArgumentParser(description="Tooling for Transformer Models")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train a Transformer Model")
    train_parser.add_argument("--backend", default="jax", help="Backend to use")
    train_parser.add_argument("input_file", help="Input text file")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    train_parser.add_argument(
        "--embedding-dim", type=int, default=256, help="Embedding dimension"
    )
    train_parser.add_argument(
        "--context-length", type=int, default=512, help="Context length"
    )
    train_parser.add_argument(
        "--n-heads", type=int, default=16, help="Number of attention heads"
    )
    train_parser.add_argument(
        "--n-layers", type=int, default=8, help="Number of transformer layers"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    train_parser.add_argument(
        "--vocab-size", type=int, default=5000, help="Vocabulary size for BPE"
    )
    train_parser.add_argument("--model-path", default="out/m0", help="Model save path")
    train_parser.add_argument("--resume", action="store_true", help="Resume training")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    generate_parser = subparsers.add_parser(
        "generate", help="Train a Transformer Model"
    )
    generate_parser.add_argument(
        "--model-path", default="out/m0", help="Model save path"
    )
    generate_parser.add_argument("prompt", help="Prompt")

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "generate":
        generate_command(args)
    else:
        parser.print_help()
