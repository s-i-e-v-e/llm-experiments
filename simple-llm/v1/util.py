import subprocess
import sys


def run_command_simple(cmd, input_text=None):
    """
    Simplest approach: stderr goes to terminal, stdout captured for JSON
    """
    result = subprocess.run(
        cmd,
        input=input_text,  # Pipe in text if provided
        stdout=subprocess.PIPE,  # Capture stdout for JSON parsing
        stderr=sys.stderr,  # Direct stderr to terminal
        text=True,  # Work with strings instead of bytes
        encoding="utf-8",  # Ensure proper encoding
    )

    return result.stdout


MODEL_CONFIG_FILE = "model_config.json"
MODEL_WEIGHTS_FILE = "model.jax"
TOKENIZER_FILE = "tokenizer.json"

import os


def get_tokenizer_path(model_path: str):
    return os.path.join(model_path, TOKENIZER_FILE)


def get_model_config_path(model_path: str):
    return os.path.join(model_path, MODEL_CONFIG_FILE)


def get_model_weights_path(model_path: str):
    return os.path.join(model_path, MODEL_WEIGHTS_FILE)
