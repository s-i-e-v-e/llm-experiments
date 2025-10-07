import math
import random
import json

STORAGE_BUFFER_USAGE = None

# ==============================================================================
# MATH PRIMITIVES & HELPERS
# ==============================================================================

def random_matrix(rows, cols):
    """Creates a matrix with random values."""
    return [[random.uniform(-0.01, 0.01) for _ in range(cols)] for _ in range(rows)]

def zeros_matrix(rows, cols):
    """Creates a matrix of zeros."""
    return [[0.0] * cols for _ in range(rows)]

def tanh(x):
    """Hyperbolic tangent activation function."""
    return math.tanh(x)

def dtanh(y):
    """Derivative of tanh, where y = tanh(x)."""
    return 1.0 - y**2

def softmax(logits):
    """Computes softmax probabilities from logits in a numerically stable way."""
    if not logits: return []
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def sample_from_dist(probs):
    """Samples an index from a probability distribution."""
    return random.choices(range(len(probs)), weights=probs, k=1)[0]

# ==============================================================================
# DUMMY DEVICE CLASS FOR API COMPATIBILITY
# ==============================================================================

class DummyDevice:
    """A dummy class to mimic the wgpu device API for compatibility."""
    def create_buffer_with_data(self, data, usage):
        # In the CPU version, we just pass the data (numpy array) through.
        return data

# ==============================================================================
# MODEL CLASS
# ==============================================================================

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

        # Model parameters
        self.W_embed = random_matrix(embedding_dim, vocab_size)
        self.W_xh = random_matrix(hidden_size, embedding_dim)
        self.W_hh = random_matrix(hidden_size, hidden_size)
        self.W_hy = random_matrix(vocab_size, hidden_size)
        self.b_h = [0.0] * hidden_size
        self.b_y = [0.0] * vocab_size

        # Placeholders for intermediate values, similar to the GPU version
        self.device = DummyDevice()
        self.cache = {}
        self.grads = {}

    # --------------------------------------------------------------------------
    # API methods for compatibility with main.py
    # --------------------------------------------------------------------------

    def get_initial_hidden_state_gpu(self):
        """Returns a zeroed hidden state vector."""
        return [0.0] * self.hidden_size

    def zero_buffer(self, buffer):
        """Zeros out a list in-place."""
        for i in range(len(buffer)):
            buffer[i] = 0.0

    def calculate_loss_gpu(self, logits, target_idx):
        """Calculates the cross-entropy loss from logits."""
        # Numerically stable cross-entropy loss calculation
        if not logits: return 0.0
        max_logit = max(logits)
        exps = [math.exp(x - max_logit) for x in logits]
        sum_exps = sum(exps)
        log_sum_exps = math.log(sum_exps)
        log_prob = (logits[target_idx] - max_logit) - log_sum_exps
        return -log_prob

    def update_hidden_state(self, h_source, h_dest):
        """Copies the last hidden state from a history list to the destination."""
        last_h = h_source[-1]
        for i in range(len(last_h)):
            h_dest[i] = last_h[i]

    # --------------------------------------------------------------------------
    # Core Training Loop Methods
    # --------------------------------------------------------------------------

    def forward_sequence(self, corpus, h_prev, offset, seq_length):
        """
        Performs a full forward pass over a sequence.
        Stores intermediate values in self.cache for the backward pass.
        """
        inputs = corpus[offset : offset + seq_length]
        self.cache = {'h': {-1: list(h_prev)}, 'x_embed': {}}
        h = list(h_prev)

        for t, x_idx in enumerate(inputs):
            x_embed = [self.W_embed[i][x_idx] for i in range(self.embedding_dim)]
            self.cache['x_embed'][t] = x_embed

            xh_dot = [sum(self.W_xh[i][j] * x_embed[j] for j in range(self.embedding_dim)) for i in range(self.hidden_size)]
            hh_dot = [sum(self.W_hh[i][j] * h[j] for j in range(self.hidden_size)) for i in range(self.hidden_size)]
            h_raw = [xh_dot[i] + hh_dot[i] + self.b_h[i] for i in range(self.hidden_size)]
            h = [tanh(val) for val in h_raw]
            self.cache['h'][t] = h

        logits = [sum(self.W_hy[i][j] * h[j] for j in range(self.hidden_size)) + self.b_y[i] for i in range(self.vocab_size)]

        # Return history as a list of state vectors, like the GPU version
        h_history = [self.cache['h'][t] for t in sorted(self.cache['h'].keys())]
        
        return logits, h_history

    def backward_sequence(self, corpus, offset, seq_length, target_idx, logits, h_history):
        """
        Performs a full backward pass (BPTT) and stores gradients in self.grads.
        """
        inputs = corpus[offset : offset + seq_length]
        cache = self.cache # Use the cache from the forward pass

        grads = {
            'W_embed': zeros_matrix(self.embedding_dim, self.vocab_size),
            'W_xh': zeros_matrix(self.hidden_size, self.embedding_dim),
            'W_hh': zeros_matrix(self.hidden_size, self.hidden_size),
            'W_hy': zeros_matrix(self.vocab_size, self.hidden_size),
            'b_h': [0.0] * self.hidden_size, 'b_y': [0.0] * self.vocab_size
        }

        probs = softmax(logits)
        dy = list(probs)
        dy[target_idx] -= 1

        h_final = cache['h'][len(inputs) - 1]
        for i in range(self.vocab_size):
            for j in range(self.hidden_size):
                grads['W_hy'][i][j] = dy[i] * h_final[j]
            grads['b_y'][i] = dy[i]

        dh = [sum(self.W_hy[i][j] * dy[i] for i in range(self.vocab_size)) for j in range(self.hidden_size)]

        for t in reversed(range(len(inputs))):
            h_t, h_prev = cache['h'][t], cache['h'][t-1]
            x_embed = cache['x_embed'][t]
            
            dh_raw = [dtanh(h_t[i]) * dh[i] for i in range(self.hidden_size)]

            for i in range(self.hidden_size):
                grads['b_h'][i] += dh_raw[i]
                for j in range(self.embedding_dim):
                    grads['W_xh'][i][j] += dh_raw[i] * x_embed[j]
                for j in range(self.hidden_size):
                    grads['W_hh'][i][j] += dh_raw[i] * h_prev[j]

            dx_embed = [sum(self.W_xh[i][j] * dh_raw[i] for i in range(self.hidden_size)) for j in range(self.embedding_dim)]
            for i in range(self.embedding_dim):
                grads['W_embed'][i][inputs[t]] += dx_embed[i]

            dh = [sum(self.W_hh[j][i] * dh_raw[j] for j in range(self.hidden_size)) for i in range(self.hidden_size)]
        
        self.grads = grads

    def update_weights(self, learning_rate):
        """Updates model parameters using stored gradients."""
        grads = self.grads
        params = self._get_params()

        for p_name, g_matrix in grads.items():
            # Clip gradients
            if isinstance(g_matrix[0], list): # Matrix
                for r in range(len(g_matrix)):
                    for c in range(len(g_matrix[0])):
                        g_matrix[r][c] = max(-5.0, min(5.0, g_matrix[r][c]))
            else: # Vector
                for i in range(len(g_matrix)):
                    g_matrix[i] = max(-5.0, min(5.0, g_matrix[i]))
            
            # Update parameters
            p_matrix = params[p_name]
            if isinstance(p_matrix[0], list):
                for i in range(len(p_matrix)):
                    for j in range(len(p_matrix[0])):
                        p_matrix[i][j] -= learning_rate * g_matrix[i][j]
            else:
                for i in range(len(p_matrix)):
                    p_matrix[i] -= learning_rate * g_matrix[i]

    # --------------------------------------------------------------------------
    # Generation Methods
    # --------------------------------------------------------------------------

    def forward_step(self, x_idx, h_prev):
        """Performs a single forward step to warm up the hidden state."""
        x_embed = [self.W_embed[i][x_idx] for i in range(self.embedding_dim)]
        xh_dot = [sum(self.W_xh[i][j] * x_embed[j] for j in range(self.embedding_dim)) for i in range(self.hidden_size)]
        hh_dot = [sum(self.W_hh[i][j] * h_prev[j] for j in range(self.hidden_size)) for i in range(self.hidden_size)]
        h_raw = [xh_dot[i] + hh_dot[i] + self.b_h[i] for i in range(self.hidden_size)]
        
        # Update h_prev in-place
        for i in range(self.hidden_size):
            h_prev[i] = tanh(h_raw[i])
        return h_prev

    def generate_step(self, x_idx, h_state):
        """Performs one step of generation: forward, softmax, and sample."""
        # 1. Forward pass for one step, updating h_state in-place
        h_new = self.forward_step(x_idx, h_state)

        # 2. Calculate logits from the new state
        logits = [sum(self.W_hy[i][j] * h_new[j] for j in range(self.hidden_size)) + self.b_y[i] for i in range(self.vocab_size)]

        # 3. Get probabilities and sample
        probs = softmax(logits)
        next_idx = sample_from_dist(probs)
        
        return next_idx

    # --------------------------------------------------------------------------
    # Model Serialization
    # --------------------------------------------------------------------------

    def _get_params(self):
        """Returns a dictionary of all model parameters."""
        return {'W_embed': self.W_embed, 'W_xh': self.W_xh, 'W_hh': self.W_hh,
                'W_hy': self.W_hy, 'b_h': self.b_h, 'b_y': self.b_y}

    def _set_params(self, params):
        """Sets model parameters from a dictionary."""
        self.W_embed = params['W_embed']; self.W_xh = params['W_xh']
        self.W_hh = params['W_hh']; self.W_hy = params['W_hy']
        self.b_h = params['b_h']; self.b_y = params['b_y']

    def save_model(self, filepath: str):
        """Saves the model configuration and parameters to a JSON file."""
        model_data = {
            'config': {'vocab_size': self.vocab_size, 'hidden_size': self.hidden_size, 'embedding_dim': self.embedding_dim},
            'params': self._get_params()
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        print(f"INFO: Model saved to '{filepath}'")

    @classmethod
    def load_model(cls, filepath: str):
        """Loads a model from a JSON file."""
        print(f"INFO: Loading model from '{filepath}'...")
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        config = model_data['config']
        model = cls(config['vocab_size'], config['hidden_size'], config['embedding_dim'])
        model._set_params(model_data['params'])
        return model