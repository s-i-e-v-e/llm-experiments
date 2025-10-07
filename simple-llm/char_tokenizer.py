import json


class CharTokenizer:
    """A simple, pedagogical character-level tokenizer."""

    def __init__(self):
        self.char_to_int = {}
        self.int_to_char = {}

    @property
    def vocab_size(self):
        return len(self.char_to_int)

    def train(self, text: str, **kwargs):
        """Builds the vocabulary from the unique characters in the text."""
        chars = sorted(list(set(text)))
        self.char_to_int = {ch: i for i, ch in enumerate(chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(chars)}
        print(
            f"INFO: Character tokenizer created with {self.vocab_size} unique characters."
        )

    def encode(self, text: str) -> list[int]:
        """Encodes a string into a list of character indices."""
        return [self.char_to_int.get(ch, -1) for ch in text if ch in self.char_to_int]

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of indices back into a string."""
        return "".join([self.int_to_char.get(i, "") for i in ids])

    def save(self, filepath: str):
        """Saves the character-to-index map to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.char_to_int, f)
        print(f"INFO: Character tokenizer vocabulary saved to '{filepath}'")

    @classmethod
    def load(cls, filepath: str):
        """Loads a character tokenizer from a saved vocabulary file."""
        tokenizer = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            tokenizer.char_to_int = json.load(f)
        tokenizer.int_to_char = {i: ch for ch, i in tokenizer.char_to_int.items()}
        print(f"INFO: Character tokenizer loaded from '{filepath}'")
        return tokenizer
