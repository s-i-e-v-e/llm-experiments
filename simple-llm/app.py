if __name__ == "__main__":
    import sys

    sys.path.append("src")

    from generate import generate_command
    from main import main
    from train import train_command

    main(train_command, generate_command)
