import numpy as np

"""
ae_module_v1.py - Unsupervised Feature Extraction
Date: April 10, 2026

Core components:
1. Autoencoder Architecture (Input-Hidden-Output)
2. Gradient Descent Training for Signal Reconstruction
"""

class Autoencoder:
    """Three-layer Autoencoder for noise-robust feature extraction."""
    def __init__(self, input_dim, hidden_dim, lam=1e-4):
        self.We = np.random.randn(input_dim, hidden_dim) * 0.1
        self.be = np.zeros(hidden_dim)
        self.Wd = np.random.randn(hidden_dim, input_dim) * 0.1
        self.bd = np.zeros(input_dim)
        self.lam = lam

    def encode(self, X):
        """Map raw signal to hidden representation."""
        return np.tanh(X @ self.We + self.be)

    def decode(self, H):
        """Reconstruct input from hidden representation."""
        return np.tanh(H @ self.Wd + self.bd)

    def train(self, X, lr=0.01, epochs=200):
        """Unsupervised training to minimize reconstruction error."""
        print(f"Starting AE training for {epochs} epochs...")
        for epoch in range(epochs):
            # Forward
            h = self.encode(X)
            z = self.decode(h)
            
            # Loss (MSE + Regularization)
            error = z - X
            
            # Backprop (simplified for POC)
            delta_d = error * (1 - z**2)
            dWd = h.T @ delta_d / len(X)
            dbd = delta_d.mean(axis=0)
            
            delta_e = (delta_d @ self.Wd.T) * (1 - h**2)
            dWe = X.T @ delta_e / len(X)
            dbe = delta_e.mean(axis=0)
            
            # Update
            self.Wd -= lr * dWd
            self.bd -= lr * dbd
            self.We -= lr * dWe
            self.be -= lr * dbe
            
            if epoch % 50 == 0:
                mse = np.mean(error**2)
                print(f"Epoch {epoch}: MSE = {mse:.6f}")

if __name__ == "__main__":
    # Test AE on synthetic signal data
    X_synthetic = np.random.randn(500, 8) # 4x4 MIMO real-repr
    ae = Autoencoder(input_dim=8, hidden_dim=24)
    ae.train(X_synthetic, epochs=150)
    print("AE Module V1 training complete.")
