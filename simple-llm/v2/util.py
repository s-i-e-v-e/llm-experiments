import os
import subprocess
import sys

# from compression import zstd
import zstandard as zstd
from flax import serialization


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
TOKENIZER_FILE = "tokenizer.json"


def get_model_file_names(model_path: str):
    return [
        os.path.join(model_path, "model.flax.zst"),
        os.path.join(model_path, "model_opt.flax.zst"),
        os.path.join(model_path, "model_config.json"),
    ]


def get_tokenizer_path(model_path: str):
    return os.path.join(model_path, TOKENIZER_FILE)


def get_model_config_path(model_path: str):
    return os.path.join(model_path, MODEL_CONFIG_FILE)


def get_model_weights_path(model_path: str):
    return get_model_file_names(model_path)[1]


def serialize_314(params, path: str):
    data = serialization.to_bytes(params)
    with zstd.open(path, "w") as f:
        f.write(data)


def deserialize313(cls, path: str):
    with zstd.open(path) as f:
        data = f.read()
    return serialization.msgpack_restore(cls, data)


def serialize(params, path: str):
    data = serialization.to_bytes(params)
    with open(path, "wb") as f:
        x = zstd.ZstdCompressor()
        f.write(x.compress(data))


def deserialize(cls, path: str):
    with open(path, "rb") as f:
        x = zstd.ZstdDecompressor()
        data = x.decompress(f.read())
    return serialization.from_bytes(cls, data)
