import numpy as np

"""
mimo_elm_v1.py - Initial Proof of Concept
Date: March 23, 2026

Core components:
1. Rayleigh MIMO Channel Model
2. Basic Extreme Learning Machine (ELM) Implementation
"""

def generate_mimo_data(Nt, Nr, N, snr_db=10):
    """Generates simple BPSK MIMO data through a Rayleigh channel."""
    # Rayleigh Channel Matrix
    H = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
    
    # BPSK Symbols (-1, +1)
    X = 2 * np.random.randint(0, 2, (N, Nt)) - 1
    
    # Signal + AWGN
    snr_lin = 10**(snr_db/10)
    sigma = np.sqrt(1 / (2 * snr_lin))
    noise = sigma * (np.random.randn(N, Nr) + 1j * np.random.randn(N, Nr))
    
    Y = X @ H.T + noise
    return X, Y, H

class SimpleELM:
    """Basic ELM with tanh activation and pseudo-inverse solution."""
    def __init__(self, n_input, n_hidden):
        self.W = np.random.randn(n_input, n_hidden)
        self.b = np.random.randn(n_hidden)
        self.beta = None

    def fit(self, X, T):
        # Hidden layer output
        H = np.tanh(X @ self.W + self.b)
        # Analytical solution using Moore-Penrose pseudo-inverse
        self.beta = np.linalg.pinv(H) @ T

    def predict(self, X):
        H = np.tanh(X @ self.W + self.b)
        return np.sign(H @ self.beta)

if __name__ == "__main__":
    # Test run
    Nt, Nr, N = 2, 2, 1000
    X, Y, H = generate_mimo_data(Nt, Nr, N)
    
    # Simple real-value representation for the neural network
    Y_real = np.hstack([Y.real, Y.imag])
    
    elm = SimpleELM(n_input=2*Nr, n_hidden=50)
    elm.fit(Y_real[:800], X[:800])
    
    preds = elm.predict(Y_real[800:])
    accuracy = np.mean(preds == X[800:])
    print(f"Initial ELM Accuracy (2x2 MIMO): {accuracy*100:.2f}%")
