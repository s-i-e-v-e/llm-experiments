if __name__ == "__main__":
    import sys

    sys.path.append("src")

    from main import main
    from v2.generate import generate_command
    from v2.train import train_command

    main(train_command, generate_command)
