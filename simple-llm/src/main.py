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
    train_parser.add_argument(
        "--model-path", type=str, required=True, help="Model save path"
    )
    train_parser.add_argument("--resume", action="store_true", help="Resume training")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # For generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate text from trained model"
    )
    generate_parser.add_argument("--model-path", type=str, required=True)
    generate_parser.add_argument("--prompt", type=str, required=True)
    generate_parser.add_argument("--max-tokens", type=int, default=100)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--top-k", type=int, default=0)
    generate_parser.add_argument("--top-p", type=float, default=1.0)
    generate_parser.add_argument("--min-p", type=float, default=0.0)
    generate_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    generate_parser.add_argument("--repetition-window", type=int, default=64)
    generate_parser.add_argument("--learning-rate", type=float, default=0.001)

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "generate":
        generate_command(args)
    else:
        parser.print_help()
