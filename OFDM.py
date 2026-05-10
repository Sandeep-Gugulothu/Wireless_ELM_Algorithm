1. The Proposed AE-ELM Algorithm
This is the most critical part of your methodology.

python
class AE_ELM:
    def __init__(self, input_size, ae_hidden_size, elm_hidden_size, output_size):
        # Random connection weights
        self.rand_w = np.random.randn(input_size, ae_hidden_size)
        self.rand_b = np.random.randn(ae_hidden_size)
        self.ae_beta = None
        
        # Detection ELM Stage
        self.elm_w = np.random.randn(input_size, elm_hidden_size)
        self.elm_b = np.random.randn(elm_hidden_size)
        self.elm_beta = None
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def train(self, x, t):
        # Stage 1: Autoencoder for Feature Reconstruction
        h_ae = self._sigmoid(np.dot(x, self.rand_w) + self.rand_b)
        self.ae_beta = np.dot(pinv(h_ae), x) # Solve reconstruction weights
        
        # Pass reconstructed features to the next stage
        features = np.dot(h_ae, self.ae_beta)
        
        # Stage 2: Detection ELM (Solving for Output Beta)
        h_elm = self._sigmoid(np.dot(features, self.elm_w) + self.elm_b)
        self.elm_beta = np.dot(pinv(h_elm), t) # Analytical solution
        
    def predict(self, x):
        h_ae = self._sigmoid(np.dot(x, self.rand_w) + self.rand_b)
        features = np.dot(h_ae, self.ae_beta)
        h_elm = self._sigmoid(np.dot(features, self.elm_w) + self.elm_b)
        return np.dot(h_elm, self.elm_beta)
def prepare_nn_input(fft_results):
    """Converts complex signals from multiple RX into a real-valued feature vector"""
    num_rx, n_symbols, n_fft = fft_results.shape
    flat = []
    for r in range(num_rx):
        flat.append(np.real(fft_results[r]))
        flat.append(np.imag(fft_results[r]))
    return np.hstack(flat)
2. The OFDM System Model
These define the physical layer environment.

python
def ofdm_tx(mod_symbols, n_fft, n_cp):
    """OFDM Transmitter: IFFT + Cyclic Prefix Insertion"""
    time_data = np.fft.ifft(mod_symbols, axis=1)
    cp = time_data[:, -n_cp:]
    return np.concatenate((cp, time_data), axis=1)
def channel_effect(tx_data, snr_db, num_rx):
    """Multi-Receiver Rayleigh Fading Channel with AWGN"""
    n_symbols, n_total = tx_data.shape
    rx_data_total = []
    snr = 10**(snr_db / 10.0)
    noise_power = np.mean(np.abs(tx_data)**2) / snr
    
    for _ in range(num_rx):
        # Rayleigh Channel Taps
        h = (np.random.randn(4) + 1j * np.random.randn(4)) / np.sqrt(8)
        rx_signal = np.array([np.convolve(s, h)[:n_total] for s in tx_data])
        # Add White Gaussian Noise
        noise = (np.random.randn(*rx_signal.shape) + 1j * np.random.randn(*rx_signal.shape)) * np.sqrt(noise_power / 2)
        rx_data_total.append(rx_signal + noise)
        
    return np.array(rx_data_total)
3. Comparison Baselines (Standard Methods)
Used in your "Results" section to prove AE-ELM is better.

python
def ls_detection(fft_results, pilots_tx, pilots_rx):
    """Least Squares (LS) Channel Estimation and Zero Forcing (ZF)"""
    num_rx = fft_results.shape[0]
    H_est = pilots_rx / pilots_tx # LS Estimation
    
    # Simple Diversity Combining + Zero Forcing
    H_mat = H_est.reshape(num_rx, 1, -1)
    # Combining multi-antenna signals using Pseudoinverse (ZF)
    # ... logic from your script ...
    return detected_symbols
def mmse_detection(fft_results, pilots_tx, pilots_rx, n_fft, snr_db):
    """Minimum Mean Square Error (MMSE) Detection"""
    noise_var = 1.0 / (10**(snr_db / 10.0))
    # ... logic for W = (H^H H + sigma^2 I)^-1 H^H ...
    return detected_symbols
4. Sensitivity Analysis (The 3D Plots)
Crucial for demonstrating model robustness (Figures in your paper).

python
def generate_3d_analysis():
    """Analyzes model sensitivity to Regularization (C) and Hidden Nodes (L)"""
    # Grid search for (C, L)
    c_values = np.logspace(-10, 10, 10)
    l_values = np.arange(50, 501, 50)
    # ... logic to calculate log-scale BER performance ...
    # Plotting 3D Surface for Performance Rate