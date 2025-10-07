import json
import os
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


def load_corpus(file_path):
    """Load text corpus from file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_model_config(config, output_dir):
    """Save model configuration"""
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
