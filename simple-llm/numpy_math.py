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
# MODEL CLASS
# ==============================================================================

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        # Initialize with NumPy arrays
        self.W_embed = np.random.uniform(-0.01, 0.01, (embedding_dim, vocab_size))
        self.W_xh = np.random.uniform(-0.01, 0.01, (hidden_size, embedding_dim))
        self.W_hh = np.random.uniform(-0.01, 0.01, (hidden_size, hidden_size))
        self.W_hy = np.random.uniform(-0.01, 0.01, (vocab_size, hidden_size))
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(vocab_size)

    def get_initial_hidden_state(self):
        return np.zeros(self.hidden_size)

    def _get_params(self):
        # Convert numpy arrays to lists for JSON serialization
        return {'W_embed': self.W_embed.tolist(), 'W_xh': self.W_xh.tolist(), 'W_hh': self.W_hh.tolist(),
                'W_hy': self.W_hy.tolist(), 'b_h': self.b_h.tolist(), 'b_y': self.b_y.tolist()}

    def _set_params(self, params):
        # Convert lists back to numpy arrays
        self.W_embed = np.array(params['W_embed']); self.W_xh = np.array(params['W_xh'])
        self.W_hh = np.array(params['W_hh']); self.W_hy = np.array(params['W_hy'])
        self.b_h = np.array(params['b_h']); self.b_y = np.array(params['b_y'])
        
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

    def forward_pass(self, inputs, h_prev):
        cache = {'h': {-1: h_prev}, 'x_embed': {}}
        h = np.copy(h_prev)
        for t, x_idx in enumerate(inputs):
            x_embed = self.W_embed[:, x_idx]
            cache['x_embed'][t] = x_embed
            h = tanh(self.W_xh @ x_embed + self.W_hh @ h + self.b_h)
            cache['h'][t] = h
        logits = self.W_hy @ h + self.b_y
        return logits, h, cache

    def backward_pass(self, inputs, target_idx, cache):
        grads = {'W_embed': np.zeros_like(self.W_embed), 'W_xh': np.zeros_like(self.W_xh),
                 'W_hh': np.zeros_like(self.W_hh), 'W_hy': np.zeros_like(self.W_hy),
                 'b_h': np.zeros_like(self.b_h), 'b_y': np.zeros_like(self.b_y)}
        
        logits, _, _ = self.forward_pass(inputs, cache['h'][-1])
        probs = softmax(logits)
        dy = np.copy(probs); dy[target_idx] -= 1

        h_final = cache['h'][len(inputs) - 1]
        grads['W_hy'] = np.outer(dy, h_final)
        grads['b_y'] = dy
        
        dh_next = np.zeros_like(self.b_h)
        for t in reversed(range(len(inputs))):
            h_t, h_prev, x_embed = cache['h'][t], cache['h'][t-1], cache['x_embed'][t]
            
            dh = self.W_hy.T @ dy + dh_next
            dh_raw = dtanh(h_t) * dh
            
            grads['b_h'] += dh_raw
            grads['W_xh'] += np.outer(dh_raw, x_embed)
            grads['W_hh'] += np.outer(dh_raw, h_prev)
            
            dx_embed = self.W_xh.T @ dh_raw
            grads['W_embed'][:, inputs[t]] += dx_embed
            
            dh_next = self.W_hh.T @ dh_raw
        return grads

    def update_weights(self, grads, learning_rate):
        for param_name, grad_matrix in grads.items():
            # Clip gradients
            np.clip(grad_matrix, -5, 5, out=grad_matrix)
            # Update parameters
            param = getattr(self, param_name)
            param -= learning_rate * grad_matrix