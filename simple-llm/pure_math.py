import math
import random
import json

# ==============================================================================
# MATH PRIMITIVES & HELPERS
# ==============================================================================

def random_matrix(rows, cols):
    return [[random.uniform(-0.01, 0.01) for _ in range(cols)] for _ in range(rows)]

def zeros_matrix(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def tanh(x):
    return math.tanh(x)

def dtanh(y):
    return 1.0 - y**2

def softmax(logits):
    if not logits: return []
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def sample_from_dist(probs):
    return random.choices(range(len(probs)), weights=probs, k=1)[0]

# ==============================================================================
# MODEL CLASS
# ==============================================================================

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.W_embed = random_matrix(embedding_dim, vocab_size)
        self.W_xh = random_matrix(hidden_size, embedding_dim)
        self.W_hh = random_matrix(hidden_size, hidden_size)
        self.W_hy = random_matrix(vocab_size, hidden_size)
        self.b_h = [0.0] * hidden_size
        self.b_y = [0.0] * vocab_size

    def get_initial_hidden_state(self):
        return [0.0] * self.hidden_size

    def _get_params(self):
        return {'W_embed': self.W_embed, 'W_xh': self.W_xh, 'W_hh': self.W_hh,
                'W_hy': self.W_hy, 'b_h': self.b_h, 'b_y': self.b_y}

    def _set_params(self, params):
        self.W_embed = params['W_embed']; self.W_xh = params['W_xh']
        self.W_hh = params['W_hh']; self.W_hy = params['W_hy']
        self.b_h = params['b_h']; self.b_y = params['b_y']
        
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
        h = list(h_prev)
        for t, x_idx in enumerate(inputs):
            x_embed = [self.W_embed[i][x_idx] for i in range(self.embedding_dim)]
            cache['x_embed'][t] = x_embed
            xh_dot = [sum(self.W_xh[i][j] * x_embed[j] for j in range(self.embedding_dim)) for i in range(self.hidden_size)]
            hh_dot = [sum(self.W_hh[i][j] * h[j] for j in range(self.hidden_size)) for i in range(self.hidden_size)]
            h_raw = [xh_dot[i] + hh_dot[i] + self.b_h[i] for i in range(self.hidden_size)]
            h = [tanh(val) for val in h_raw]
            cache['h'][t] = h
        logits = [sum(self.W_hy[i][j] * h[j] for j in range(self.hidden_size)) + self.b_y[i] for i in range(self.vocab_size)]
        return logits, h, cache

    def backward_pass(self, inputs, target_idx, cache):
        grads = {'W_embed': zeros_matrix(self.embedding_dim, self.vocab_size), 'W_xh': zeros_matrix(self.hidden_size, self.embedding_dim),
                 'W_hh': zeros_matrix(self.hidden_size, self.hidden_size), 'W_hy': zeros_matrix(self.vocab_size, self.hidden_size),
                 'b_h': [0.0] * self.hidden_size, 'b_y': [0.0] * self.vocab_size}
        logits, _, _ = self.forward_pass(inputs, cache['h'][-1])
        probs = softmax(logits)
        dy = list(probs); dy[target_idx] -= 1
        h_final = cache['h'][len(inputs) - 1]
        for i in range(self.vocab_size):
            for j in range(self.hidden_size): grads['W_hy'][i][j] = dy[i] * h_final[j]
            grads['b_y'][i] = dy[i]
        dh_next = [0.0] * self.hidden_size
        for t in reversed(range(len(inputs))):
            h_t, h_prev, x_embed = cache['h'][t], cache['h'][t-1], cache['x_embed'][t]
            dh = [sum(self.W_hy[i][j] * dy[i] for i in range(self.vocab_size)) for j in range(self.hidden_size)]
            dh = [dh[i] + dh_next[i] for i in range(self.hidden_size)]
            dh_raw = [dtanh(h_t[i]) * dh[i] for i in range(self.hidden_size)]
            grads['b_h'] = [grads['b_h'][i] + dh_raw[i] for i in range(self.hidden_size)]
            for i in range(self.hidden_size):
                for j in range(self.embedding_dim): grads['W_xh'][i][j] += dh_raw[i] * x_embed[j]
                for j in range(self.hidden_size): grads['W_hh'][i][j] += dh_raw[i] * h_prev[j]
            dx_embed = [sum(self.W_xh[i][j] * dh_raw[i] for i in range(self.hidden_size)) for j in range(self.embedding_dim)]
            for i in range(self.embedding_dim): grads['W_embed'][i][inputs[t]] += dx_embed[i]
            dh_next = [sum(self.W_hh[j][i] * dh_raw[j] for j in range(self.hidden_size)) for i in range(self.hidden_size)]
        return grads

    def update_weights(self, grads, learning_rate):
        for p_name, g_matrix in grads.items():
            if isinstance(g_matrix, list) and isinstance(g_matrix[0], list): # Matrix
                for row in g_matrix:
                    for i in range(len(row)): row[i] = max(-5.0, min(5.0, row[i]))
            else: # Vector
                for i in range(len(g_matrix)): g_matrix[i] = max(-5.0, min(5.0, g_matrix[i]))
        params = self._get_params()
        for p_name, p_matrix in params.items():
            g_matrix = grads[p_name]
            if isinstance(p_matrix[0], list):
                for i in range(len(p_matrix)):
                    for j in range(len(p_matrix[0])): p_matrix[i][j] -= learning_rate * g_matrix[i][j]
            else:
                for i in range(len(p_matrix)): p_matrix[i] -= learning_rate * g_matrix[i]