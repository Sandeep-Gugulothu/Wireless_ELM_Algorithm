import numpy as np

class ELM_AE:
    """Extreme Learning Machine based Autoencoder."""
    def __init__(self, input_dim, hidden_dim, activation='sigmoid'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Random orthonormal weights for the encoder
        W = np.random.normal(size=(input_dim, hidden_dim))
        u, s, vh = np.linalg.svd(W, full_matrices=False)
        self.W = u @ vh # Orthonormal mapping
        self.b = np.random.normal(size=(1, hidden_dim))
        self.beta = None # Decoder weights
        
        if activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
        else:
            self.activation = np.tanh

    def train(self, X, C=1.0):
        # Hidden layer output
        H = self.activation(np.dot(X, self.W) + self.b)
        # Solve H * beta = X (Autoencoder goal)
        I = np.eye(self.hidden_dim)
        self.beta = np.linalg.solve(np.dot(H.T, H) + I / C, np.dot(H.T, X))

    def extract_features(self, X):
        # Features are H * beta (Reconstructed features or just H)
        # In ELM-AE, the 'beta' acts as the feature mapping weights 'a' in the paper (Eq 17)
        H = self.activation(np.dot(X, self.W) + self.b)
        return np.dot(H, self.beta)

class ELM_Classifier:
    def __init__(self, input_dim, hidden_dim, output_dim, C=1.0):
        self.W = np.random.normal(size=(input_dim, hidden_dim))
        self.b = np.random.normal(size=(1, hidden_dim))
        self.beta = None
        self.C = C

    def _get_H(self, X):
        return np.tanh(np.dot(X, self.W) + self.b)

    def train(self, X, T):
        H = self._get_H(X)
        I = np.eye(H.shape[1])
        self.beta = np.linalg.solve(np.dot(H.T, H) + I / self.C, np.dot(H.T, T))

    def predict(self, X):
        H = self._get_H(X)
        return np.dot(H, self.beta)

class MIMO_OFDM_System:
    def __init__(self, n_tx=2, n_rx=2, n_fft=64, cp_len=16):
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_fft = n_fft
        self.cp_len = cp_len

    def generate_data(self, n_blocks, snr_db, H_fixed=None):
        # 1. Bits
        bits = np.random.randint(0, 2, (n_blocks, self.n_tx, self.n_fft, 2))
        x_freq = (2*bits[:,:,:,0]-1) + 1j*(2*bits[:,:,:,1]-1)
        x_freq = x_freq / np.sqrt(2)
        
        # 2. IFFT
        x_time = np.fft.ifft(x_freq, axis=2)
        
        # 3. CP
        cp = x_time[:, :, -self.cp_len:]
        x_cp = np.concatenate([cp, x_time], axis=2)
        
        # 4. Channel
        if H_fixed is None:
            H_fixed = (np.random.randn(self.n_rx, self.n_tx) + 
                       1j * np.random.randn(self.n_rx, self.n_tx)) / np.sqrt(2)
        
        y_time = np.zeros((n_blocks, self.n_rx, x_cp.shape[2]), dtype=complex)
        for b in range(n_blocks):
            y_time[b] = np.dot(H_fixed, x_cp[b])
            
        # 5. Noise
        snr_linear = 10**(snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        noise = (np.random.randn(*y_time.shape) + 1j * np.random.randn(*y_time.shape)) * np.sqrt(noise_power / 2)
        y_noisy = y_time + noise
        
        # 6. Receiver FFT
        y_no_cp = y_noisy[:, :, self.cp_len:]
        y_freq = np.fft.fft(y_no_cp, axis=2)
        
        # Features: (blocks * subcarriers, rx * 2)
        # Flattening subcarriers into the sample dimension makes detection easier for the ELM
        y_reshaped = y_freq.transpose(0, 2, 1).reshape(-1, self.n_rx)
        y_features = np.hstack([y_reshaped.real, y_reshaped.imag])
        
        t_reshaped = bits.transpose(0, 2, 1, 3).reshape(-1, self.n_tx * 2)
        
        return y_features, t_reshaped, H_fixed
