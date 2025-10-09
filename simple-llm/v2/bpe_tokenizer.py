import json
import os

from util import run_command_simple

# The path to our compiled C executable.
# Assumes it's in the same directory as the script.
BPE_EXECUTABLE = os.path.join(os.path.dirname(__file__), "bpe_tokenizer")


class BPETokenizer:
    """
    A Python wrapper for a high-performance C implementation of a BPE tokenizer.
    """

    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text: str, vocab_size: int, token_file: str, verbose: bool = False):
        """
        Trains the tokenizer by calling the C binary.
        Feeds the text via stdin and reads the resulting JSON merges from stdout.
        """
        if not os.path.exists(BPE_EXECUTABLE):
            raise FileNotFoundError(
                f"BPE executable not found at '{BPE_EXECUTABLE}'. Please compile it first."
            )

        cmd = [BPE_EXECUTABLE, "train", str(vocab_size)]
        output = run_command_simple(cmd, text)
        serializable_merges = json.loads(output)

        # We still need to populate the Python object with the results
        self.merges = {
            tuple(map(int, k.split(","))): v for k, v in serializable_merges.items()
        }
        self._reconstruct_vocab()

        if verbose:
            print(f"BPE training complete. Learned {len(self.merges)} merges.")
        self._save_merges_to_file(token_file)

    def encode(self, text: str, token_file: str) -> list[int]:
        """
        Encodes text by calling the C binary.
        Feeds the text via stdin and reads the token IDs from stdout.
        """

        if not os.path.exists(BPE_EXECUTABLE):
            raise FileNotFoundError(
                f"BPE executable not found at '{BPE_EXECUTABLE}'. Please compile it first."
            )

        cmd = [BPE_EXECUTABLE, "encode", token_file]
        output = run_command_simple(cmd, text)

        # The C program prints token IDs separated by spaces
        token_ids = [int(i) for i in output.strip().split()]
        return token_ids

    def decode(self, ids: list[int]) -> str:
        """
        Decoding is simple and can remain in Python as it's not a bottleneck.
        """
        tokens_bytes = b"".join(self.vocab.get(idx, b"?") for idx in ids)
        text = tokens_bytes.decode("utf-8", errors="replace")
        return text

    def _reconstruct_vocab(self):
        """Rebuilds the self.vocab dictionary from the self.merges rules."""
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def _save_merges_to_file(self, filepath: str):
        """Internal helper to save merges to a file for the C encoder."""
        serializable_merges = {f"{p[0]},{p[1]}": idx for p, idx in self.merges.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_merges, f)
        print(f"INFO: Custom BPE tokenizer (C-accelerated) saved to '{filepath}'")

    def save(self, filepath: str):
        """Saves the tokenizer's merge rules to a file."""
        pass

    @classmethod
    def load(cls, filepath: str):
        """Loads a tokenizer from a saved merge file."""
        tokenizer = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            serializable_merges = json.load(f)

        tokenizer.merges = {
            tuple(map(int, k.split(","))): v for k, v in serializable_merges.items()
        }
        tokenizer._reconstruct_vocab()
        print(f"INFO: Custom BPE tokenizer (C-accelerated) loaded from '{filepath}'")
        return tokenizer
