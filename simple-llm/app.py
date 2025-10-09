if __name__ == "__main__":
    import sys

    sys.path.append("src")

    from main import main
    from v1.generate import generate_command
    from v1.train import train_command

    main(train_command, generate_command)
