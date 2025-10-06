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
