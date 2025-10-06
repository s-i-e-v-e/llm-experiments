# FILE: bpe_tokenizer.py
import json

def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs.
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """
    In a list of integers, replace all consecutive occurrences of pair with the new integer idx.
    Example: merge([1, 2, 3, 1, 2], (1, 2), 4) -> [4, 3, 4]
    """
    new_ids = []
    i = 0
    while i < len(ids):
        # If we are at the end of the list and the last element is not part of a pair
        if i == len(ids) - 1:
            new_ids.append(ids[i])
            break
        # If the current pair matches the one we want to merge
        if (ids[i], ids[i+1]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class BPETokenizer:
    """A pedagogical implementation of a Byte-Pair Encoding tokenizer."""
    def __init__(self):
        # The vocabulary starts with the 256 base "tokens" (the bytes)
        # and grows as we learn new merges.
        # The `merges` dict maps (int, int) -> int
        self.merges = {}
        # The `vocab` dict maps int -> bytes
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """
        Trains the tokenizer on a given text to achieve a specific vocabulary size.
        """
        if vocab_size < 256:
            raise ValueError("Vocabulary size must be at least 256.")

        # 1. Start with the raw UTF-8 bytes of the text
        ids = list(text.encode("utf-8"))
        
        # 2. Iteratively merge the most frequent pair
        num_merges = vocab_size - 256
        for i in range(num_merges):
            # Find the most frequent pair
            stats = get_stats(ids)
            if not stats:
                break # No more pairs to merge
            
            top_pair = max(stats, key=stats.get)
            
            # The new token ID is the next available integer
            new_idx = 256 + i
            
            # Perform the merge
            ids = merge(ids, top_pair, new_idx)
            
            # Store the merge rule and the new token's byte representation
            self.merges[top_pair] = new_idx
            self.vocab[new_idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
            if verbose:
                # Instead of converting the ID back to a byte, we look up its
                # actual byte representation in our vocabulary. This works for
                # both base tokens (<256) and new merged tokens (>=256).
                p0 = self.vocab[top_pair[0]].decode('utf-8', 'ignore')
                p1 = self.vocab[top_pair[1]].decode('utf-8', 'ignore')
                new_tok_str = self.vocab[new_idx].decode('utf-8', 'ignore')
                print(f"Merge {i+1}/{num_merges}: Merging ('{p0}', '{p1}') -> '{new_tok_str}' (ID: {new_idx})")

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a list of token IDs.
        """
        # Start with the raw UTF-8 byte sequence
        ids = list(text.encode("utf-8"))
        
        # Keep merging until no more merges are possible
        while True:
            stats = get_stats(ids)
            # Find the best merge available from our learned vocabulary
            # We look for the merge that appeared earliest in training (has the lowest new_idx)
            possible_merges = {pair: self.merges[pair] for pair in stats if pair in self.merges}
            
            if not possible_merges:
                break # No more merges can be applied
                
            best_pair = min(possible_merges, key=lambda p: self.merges[p])
            
            # Apply the merge
            ids = merge(ids, best_pair, self.merges[best_pair])
            
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        """
        # Concatenate the byte representations of each token
        tokens_bytes = b"".join(self.vocab[idx] for idx in ids)
        # Decode the final byte stream into a string, ignoring errors
        text = tokens_bytes.decode("utf-8", errors="replace")
        return text

    def save(self, filepath: str):
        """Saves the tokenizer's merge rules to a file."""
        # We only need to save the merges. The vocab can be reconstructed from them.
        # We convert tuple keys to strings for JSON compatibility.
        serializable_merges = {f"{p[0]},{p[1]}": idx for p, idx in self.merges.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_merges, f)
        print(f"INFO: Custom BPE tokenizer saved to '{filepath}'")

    @classmethod
    def load(cls, filepath: str):
        """Loads a tokenizer from a saved merge file."""
        tokenizer = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            serializable_merges = json.load(f)
        
        # Convert string keys back to integer tuples
        merges = {tuple(map(int, k.split(','))): v for k, v in serializable_merges.items()}
        tokenizer.merges = merges
        
        # Reconstruct the vocabulary from the merges
        for (p0, p1), idx in merges.items():
            tokenizer.vocab[idx] = tokenizer.vocab[p0] + tokenizer.vocab[p1]
            
        print(f"INFO: Custom BPE tokenizer loaded from '{filepath}'")
        return tokenizer