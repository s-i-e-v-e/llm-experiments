import numpy as np
import json
import random

# ==============================================================================
# MATH PRIMITIVES & HELPERS
# ==============================================================================

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1.0 - y**2

def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)

def sample_from_dist(probs):
    return np.random.choice(len(probs), p=probs)

# ==============================================================================
# DUMMY DEVICE & CONSTANT FOR API COMPATIBILITY
# ==============================================================================

# Add a constant for the buffer usage flag, set to None for the CPU backend.
STORAGE_BUFFER_USAGE = None

class DummyDevice:
    """A dummy class to mimic the wgpu device API for compatibility."""
    def create_buffer_with_data(self, data, usage):
        # In the numpy version, we just pass the data (numpy array) through.
        return data

# ==============================================================================
# MODEL CLASS
# ==============================================================================

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        # Initialize with NumPy arrays
        self.W_embed = np.random.uniform(-0.01, 0.01, (embedding_dim, vocab_size)).astype(np.float32)
        self.W_xh = np.random.uniform(-0.01, 0.01, (hidden_size, embedding_dim)).astype(np.float32)
        self.W_hh = np.random.uniform(-0.01, 0.01, (hidden_size, hidden_size)).astype(np.float32)
        self.W_hy = np.random.uniform(-0.01, 0.01, (vocab_size, hidden_size)).astype(np.float32)
        self.b_h = np.zeros(hidden_size, dtype=np.float32)
        self.b_y = np.zeros(vocab_size, dtype=np.float32)

        # Add placeholders for compatibility with the main script's API
        self.device = DummyDevice()
        self.cache = {}
        self.grads = {}


    def get_initial_hidden_state_gpu(self):
        """Returns a zeroed hidden state vector (API compatibility)."""
        return np.zeros(self.hidden_size, dtype=np.float32)

    def zero_buffer(self, buffer):
        """Zeros out a numpy array in-place."""
        buffer.fill(0.0)

    def calculate_loss_gpu(self, logits, target_idx):
        """Calculates the cross-entropy loss from logits (API compatibility)."""
        # Numerically stable cross-entropy loss
        max_logit = np.max(logits)
        log_sum_exp = np.log(np.sum(np.exp(logits - max_logit)))
        log_prob = (logits[target_idx] - max_logit) - log_sum_exp
        return -log_prob

    def update_hidden_state(self, h_source, h_dest):
        """Copies the last hidden state from a history list to the destination."""
        np.copyto(h_dest, h_source[-1])

    def _get_params(self):
        # Convert numpy arrays to lists for JSON serialization
        return {'W_embed': self.W_embed.tolist(), 'W_xh': self.W_xh.tolist(), 'W_hh': self.W_hh.tolist(),
                'W_hy': self.W_hy.tolist(), 'b_h': self.b_h.tolist(), 'b_y': self.b_y.tolist()}

    def _set_params(self, params):
        # Convert lists back to numpy arrays
        self.W_embed = np.array(params['W_embed'], dtype=np.float32); self.W_xh = np.array(params['W_xh'], dtype=np.float32)
        self.W_hh = np.array(params['W_hh'], dtype=np.float32); self.W_hy = np.array(params['W_hy'], dtype=np.float32)
        self.b_h = np.array(params['b_h'], dtype=np.float32); self.b_y = np.array(params['b_y'], dtype=np.float32)
        
    def save_model(self, filepath: str):
        model_data = {
            'config': {'vocab_size': self.vocab_size, 'hidden_size': self.hidden_size, 'embedding_dim': self.embedding_dim},
            'params': self._get_params()
        }
        with open(filepath, 'w') as f: json.dump(model_data, f)
        print(f"INFO: Model saved to '{filepath}'")

    @classmethod
    def load_model(cls, filepath: str):
        print(f"INFO: Loading model from '{filepath}'...")
        with open(filepath, 'r') as f: model_data = json.load(f)
        config = model_data['config']
        model = cls(config['vocab_size'], config['hidden_size'], config['embedding_dim'])
        model._set_params(model_data['params'])
        return model

    def forward_sequence(self, corpus, h_prev, offset, seq_length):
        """Performs a full forward pass and caches intermediate values."""
        inputs = corpus[offset : offset + seq_length]
        self.cache = {'h': {-1: h_prev}, 'x_embed': {}}
        h = np.copy(h_prev)
        for t, x_idx in enumerate(inputs):
            x_embed = self.W_embed[:, x_idx]
            self.cache['x_embed'][t] = x_embed
            h = tanh(self.W_xh @ x_embed + self.W_hh @ h + self.b_h)
            self.cache['h'][t] = h
        logits = self.W_hy @ h + self.b_y
        
        h_history = [self.cache['h'][t] for t in sorted(self.cache['h'].keys())]
        return logits, h_history

    def backward_sequence(self, corpus, offset, seq_length, target_idx, logits, h_history):
        """Performs a full backward pass (BPTT) and stores gradients."""
        inputs = corpus[offset : offset + seq_length]
        cache = self.cache # Use cache from the forward pass

        grads = {'W_embed': np.zeros_like(self.W_embed), 'W_xh': np.zeros_like(self.W_xh),
                 'W_hh': np.zeros_like(self.W_hh), 'W_hy': np.zeros_like(self.W_hy),
                 'b_h': np.zeros_like(self.b_h), 'b_y': np.zeros_like(self.b_y)}
        
        probs = softmax(logits)
        dy = np.copy(probs); dy[target_idx] -= 1

        h_final = cache['h'][len(inputs) - 1]
        grads['W_hy'] = np.outer(dy, h_final)
        grads['b_y'] = dy
        
        dh = self.W_hy.T @ dy
        for t in reversed(range(len(inputs))):
            h_t, h_prev, x_embed = cache['h'][t], cache['h'][t-1], cache['x_embed'][t]
            
            dh_raw = dtanh(h_t) * dh
            
            grads['b_h'] += dh_raw
            grads['W_xh'] += np.outer(dh_raw, x_embed)
            grads['W_hh'] += np.outer(dh_raw, h_prev)
            
            dx_embed = self.W_xh.T @ dh_raw
            grads['W_embed'][:, inputs[t]] += dx_embed
            
            dh = self.W_hh.T @ dh_raw
        
        self.grads = grads

    def update_weights(self, learning_rate):
        """Updates model parameters using stored gradients."""
        for param_name, grad_matrix in self.grads.items():
            np.clip(grad_matrix, -5, 5, out=grad_matrix)
            param = getattr(self, param_name)
            param -= learning_rate * grad_matrix

    def forward_step(self, x_idx, h_state):
        """Performs a single forward step to warm up the hidden state."""
        x_embed = self.W_embed[:, x_idx]
        h_state[:] = tanh(self.W_xh @ x_embed + self.W_hh @ h_state + self.b_h)
        return h_state

    def generate_step(self, x_idx, h_state):
        """Performs one step of generation: forward, softmax, and sample."""
        h_new = self.forward_step(x_idx, h_state)
        logits = self.W_hy @ h_new + self.b_y
        probs = softmax(logits)
        return sample_from_dist(probs)