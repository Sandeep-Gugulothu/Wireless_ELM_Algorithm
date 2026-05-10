import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv, inv

# OFDM Parameters
n_fft = 64
n_cp = 16
modulation_order = 4  # QPSK
num_receivers = 3
num_symbols = 5000  # Number of OFDM symbols for training
num_test_symbols = 1000
num_taps = 4  # Channel taps

def qpsk_mod(bits):
    bits = bits.reshape(-1, 2)
    symbols = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)
    return symbols / np.sqrt(2)

def qpsk_demod(symbols):
    real = (np.real(symbols) > 0).astype(int)
    imag = (np.imag(symbols) > 0).astype(int)
    return np.stack((real, imag), axis=1).flatten()

def generate_data(n_symbols, n_fft):
    bits = np.random.randint(0, 2, (n_symbols, n_fft, 2))
    mod_symbols = np.zeros((n_symbols, n_fft), dtype=complex)
    for i in range(n_symbols):
        mod_symbols[i] = qpsk_mod(bits[i])
    return bits, mod_symbols

def ofdm_tx(mod_symbols, n_fft, n_cp):
    time_data = np.fft.ifft(mod_symbols, axis=1)
    cp = time_data[:, -n_cp:]
    tx_data = np.concatenate((cp, time_data), axis=1)
    return tx_data

def channel_effect(tx_data, snr_db, num_rx):
    n_symbols, n_total = tx_data.shape
    rx_data_total = []
    channels = []
    
    snr = 10**(snr_db / 10.0)
    signal_power = np.mean(np.abs(tx_data)**2)
    noise_power = signal_power / snr
    
    for _ in range(num_rx):
        # Rayleigh Channel
        h = (np.random.randn(num_taps) + 1j * np.random.randn(num_taps)) / np.sqrt(2 * num_taps)
        channels.append(h)
        rx_signal = np.zeros((n_symbols, n_total + len(h) - 1), dtype=complex)
        for i in range(n_symbols):
            rx_signal[i] = np.convolve(tx_data[i], h)
        
        noise = (np.random.randn(*rx_signal.shape) + 1j * np.random.randn(*rx_signal.shape)) * np.sqrt(noise_power / 2)
        rx_signal_noisy = rx_signal + noise
        rx_data_total.append(rx_signal_noisy[:, :n_total])
        
    return np.array(rx_data_total), channels

def ofdm_rx_fft(rx_data, n_fft, n_cp):
    num_rx, n_symbols, n_total = rx_data.shape
    fft_results = []
    for r in range(num_rx):
        no_cp = rx_data[r, :, n_cp:]
        fft_data = np.fft.fft(no_cp, axis=1)
        fft_results.append(fft_data)
    return np.array(fft_results)

class AE_ELM:
    def __init__(self, input_size, ae_hidden_size, elm_hidden_size, output_size):
        # Random connection weights (as per diagram)
        self.rand_w = np.random.randn(input_size, ae_hidden_size)
        self.rand_b = np.random.randn(ae_hidden_size)
        
        # ELM-AE beta (reconstruction)
        self.ae_beta = None
        
        # Detection ELM
        # The input to this stage is either the hidden layer OR the reconstructed features.
        # Based on the diagram, the output of AE is fed to ELM. 
        # If AE is used for feature extraction, we use reconstructed 'features' (dim = input_size)
        self.elm_w = np.random.randn(input_size, elm_hidden_size)
        self.elm_b = np.random.randn(elm_hidden_size)
        self.elm_beta = None
        
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500) # prevent overflow
        return 1 / (1 + np.exp(-x))
    
    def train(self, x, t):
        # Hidden layer 1: Random mapping
        h_ae = self._sigmoid(np.dot(x, self.rand_w) + self.rand_b)
        
        # ELM-AE: Solve for beta to reconstruct input
        self.ae_beta = np.dot(pinv(h_ae), x)
        
        # Features passed to detection stage (reconstructed output of AE)
        features = np.dot(h_ae, self.ae_beta)
        
        # Hidden layer 2: Detection random mapping
        h_elm = self._sigmoid(np.dot(features, self.elm_w) + self.elm_b)
        self.elm_beta = np.dot(pinv(h_elm), t)
        
    def predict(self, x):
        h_ae = self._sigmoid(np.dot(x, self.rand_w) + self.rand_b)
        features = np.dot(h_ae, self.ae_beta)
        h_elm = self._sigmoid(np.dot(features, self.elm_w) + self.elm_b)
        return np.dot(h_elm, self.elm_beta)


def ls_detection(fft_results, pilots_tx, pilots_rx, n_fft):
    # Simple Least Squares Channel Estimation and Zero Forcing
    # pilots_tx: (Nfft,)
    # pilots_rx: (NumRx, Nfft)
    num_rx = fft_results.shape[0]
    n_symbols = fft_results.shape[1]
    
    # Estimate channel for each Rx
    H_est = pilots_rx / pilots_tx # (NumRx, Nfft)
    
    # Zero Forcing combiner (MRC followed by ZF or just average)
    # We'll use simple Averaging + ZF for each subcarrier
    detected_symbols = np.zeros((n_symbols, n_fft), dtype=complex)
    
    for k in range(n_fft):
        h_k = H_est[:, k] # (NumRx,)
        y_k = fft_results[:, :, k] # (NumRx, NSymbols)
        
        # MRC-like combination: w = h* / |h|^2
        # For simplicity, if we have multiple Rx, we can use ZF/MMSE formula:
        # x_hat = (H^H H)^-1 H^H y
        H_mat = h_k.reshape(-1, 1) # (NumRx, 1)
        W = pinv(H_mat) # (1, NumRx)
        detected_symbols[:, k] = np.dot(W, y_k).flatten()
        
    return detected_symbols

def prepare_nn_input(fft_results):
    num_rx, n_symbols, n_fft = fft_results.shape
    flat = []
    for r in range(num_rx):
        flat.append(np.real(fft_results[r]))
        flat.append(np.imag(fft_results[r]))
    return np.hstack(flat)

def prepare_nn_output(mod_symbols):
    return np.hstack([np.real(mod_symbols), np.imag(mod_symbols)])

class MLP_BP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
        self.lr = lr
        self.loss_history = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_derivative(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)

    def train(self, x, t, epochs=50):
        for epoch in range(epochs):
            # Forward
            z1 = np.dot(x, self.w1) + self.b1
            a1 = self._sigmoid(z1)
            z2 = np.dot(a1, self.w2) + self.b2
            a2 = z2  # Linear output for regression

            # Loss (MSE)
            loss = np.mean((a2 - t) ** 2)
            self.loss_history.append(loss)

            # Backward
            dz2 = 2 * (a2 - t) / x.shape[0]
            dw2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0)

            da1 = np.dot(dz2, self.w2.T)
            dz1 = da1 * self._sigmoid_derivative(z1)
            dw1 = np.dot(x.T, dz1)
            db1 = np.sum(dz1, axis=0)

            # Update
            self.w2 -= self.lr * dw2
            self.b2 -= self.lr * db2
            self.w1 -= self.lr * dw1
            self.b1 -= self.lr * db1

    def predict(self, x):
        a1 = self._sigmoid(np.dot(x, self.w1) + self.b1)
        return np.dot(a1, self.w2) + self.b2

def mmse_detection(fft_results, pilots_tx, pilots_rx, n_fft, snr_db):
    num_rx = fft_results.shape[0]
    n_symbols = fft_results.shape[1]
    snr_lin = 10**(snr_db / 10.0)
    noise_var = 1.0 / snr_lin
    
    H_est = pilots_rx / pilots_tx
    detected_symbols = np.zeros((n_symbols, n_fft), dtype=complex)
    
    for k in range(n_fft):
        h_k = H_est[:, k].reshape(-1, 1) # (NumRx, 1)
        y_k = fft_results[:, :, k] # (NumRx, NSymbols)
        
        # MMSE Filter: W = (H^H H + sigma^2 I)^-1 H^H
        H_H = h_k.conj().T
        W = inv(H_H @ h_k + noise_var * np.eye(1)) @ H_H
        detected_symbols[:, k] = (W @ y_k).flatten()
        
    return detected_symbols

import time

def run_simulation():
    snr_range = np.arange(0, 26, 1)  # 1 dB intervals
    ber_ae_elm = []
    ber_elm = []
    ber_mlp = []
    ber_ls_zf = []
    ber_mmse = []
    
    runtimes = {}
    
    # Training
    train_snr = 20
    train_bits, train_mod = generate_data(num_symbols, n_fft)
    tx_train = ofdm_tx(train_mod, n_fft, n_cp)
    rx_train, _ = channel_effect(tx_train, train_snr, num_receivers)
    fft_train = ofdm_rx_fft(rx_train, n_fft, n_cp)
    
    x_train = prepare_nn_input(fft_train)
    y_train = prepare_nn_output(train_mod)
    
    input_size = num_receivers * n_fft * 2
    output_size = n_fft * 2
    
    print("Training AE-ELM...")
    start = time.time()
    ae_elm = AE_ELM(input_size, 256, 1024, output_size)
    ae_elm.train(x_train, y_train)
    runtimes['AE-ELM'] = time.time() - start
    
    print("Training Standard ELM...")
    start = time.time()
    elm_w_rand = np.random.randn(input_size, 1024)
    h_elm_train = 1 / (1 + np.exp(-np.dot(x_train, elm_w_rand)))
    beta_elm = np.dot(pinv(h_elm_train), y_train)
    runtimes['ELM'] = time.time() - start
    
    print("Training MLP-BP (Calibration Model)...")
    start = time.time()
    mlp = MLP_BP(input_size, 256, output_size, lr=0.1)
    mlp.train(x_train, y_train, epochs=100)
    runtimes['MLP-BP'] = time.time() - start
    
    runtimes['LS/MMSE Setup'] = 0.0001
    
    # Learning Curve
    elm_learning_err = []
    sample_sizes = np.linspace(100, num_symbols, 20, dtype=int)
    for sz in sample_sizes:
        h_tmp = 1 / (1 + np.exp(-np.dot(x_train[:sz], elm_w_rand)))
        beta_tmp = np.dot(pinv(h_tmp), y_train[:sz])
        pred_tmp = np.dot(1 / (1 + np.exp(-np.dot(x_train, elm_w_rand))), beta_tmp)
        elm_learning_err.append(np.mean((pred_tmp - y_train)**2))

    pilot_bits, pilot_mod = generate_data(1, n_fft)
    pilot_tx = ofdm_tx(pilot_mod, n_fft, n_cp)

    for snr in snr_range:
        print(f"Testing SNR: {snr} dB")
        test_bits, test_mod = generate_data(num_test_symbols, n_fft)
        tx_test = ofdm_tx(test_mod, n_fft, n_cp)
        rx_test, _ = channel_effect(tx_test, snr, num_receivers)
        fft_test = ofdm_rx_fft(rx_test, n_fft, n_cp)
        
        rx_pilot, _ = channel_effect(pilot_tx, snr, num_receivers)
        fft_pilot = ofdm_rx_fft(rx_pilot, n_fft, n_cp)[:, 0, :]
        
        pred_zf = ls_detection(fft_test, pilot_mod[0], fft_pilot, n_fft)
        pred_mmse = mmse_detection(fft_test, pilot_mod[0], fft_pilot, n_fft, snr)
        
        x_test = prepare_nn_input(fft_test)
        pred_ae_elm_raw = ae_elm.predict(x_test)
        h_elm_test = 1 / (1 + np.exp(-np.dot(x_test, elm_w_rand)))
        pred_elm_raw = np.dot(h_elm_test, beta_elm)
        pred_mlp_raw = mlp.predict(x_test)

        def decode_bits(preds):
            res = preds[:, :n_fft] + 1j * preds[:, n_fft:]
            bits_extracted = []
            for s in range(num_test_symbols):
                bits_extracted.append(qpsk_demod(res[s]))
            return np.array(bits_extracted).flatten()

        def decode_symbols(syms):
            bits_extracted = []
            for s in range(num_test_symbols):
                bits_extracted.append(qpsk_demod(syms[s]))
            return np.array(bits_extracted).flatten()

        ber_ae_elm.append(np.mean(decode_bits(pred_ae_elm_raw) != test_bits.flatten()))
        ber_elm.append(np.mean(decode_bits(pred_elm_raw) != test_bits.flatten()))
        ber_mlp.append(np.mean(decode_bits(pred_mlp_raw) != test_bits.flatten()))
        ber_ls_zf.append(np.mean(decode_symbols(pred_zf) != test_bits.flatten()))
        ber_mmse.append(np.mean(decode_symbols(pred_mmse) != test_bits.flatten()))
        
    # Convert parameters
    acc_ae_elm = [1 - b for b in ber_ae_elm]
    acc_elm = [1 - b for b in ber_elm]
    acc_mlp = [1 - b for b in ber_mlp]
    acc_ls_zf = [1 - b for b in ber_ls_zf]
    acc_mmse = [1 - b for b in ber_mmse]
    
    # Throughput (Effective Bits per Transmission)
    # Total Bits = NumSubcarriers * 2 (for QPSK)
    max_bits = n_fft * 2
    th_ae_elm = [max_bits * (1 - b) for b in ber_ae_elm]
    th_elm = [max_bits * (1 - b) for b in ber_elm]
    th_ls_zf = [max_bits * (1 - b) for b in ber_ls_zf]
    th_mmse = [max_bits * (1 - b) for b in ber_mmse]

    # Plot 1: Multi-Parameter Summary
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # BER Plot
    axs[0, 0].semilogy(snr_range, ber_ae_elm, 'o-', lw=2, label='Proposed AE-ELM')
    axs[0, 0].semilogy(snr_range, ber_elm, 's--', label='Standard ELM')
    axs[0, 0].semilogy(snr_range, ber_mmse, 'd-.', label='MMSE')
    axs[0, 0].semilogy(snr_range, ber_ls_zf, 'x-', alpha=0.4, label='LS-ZF')
    axs[0, 0].set_xlabel('SNR (dB)'); axs[0, 0].set_ylabel('BER'); axs[0, 0].set_title('Bit Error Rate vs SNR')
    axs[0, 0].grid(True, which='both'); axs[0, 0].legend()

    # Accuracy Plot
    axs[0, 1].plot(snr_range, acc_ae_elm, 'o-', lw=2, label='AE-ELM', color='green')
    axs[0, 1].plot(snr_range, acc_elm, 's--', label='Standard ELM', color='blue')
    axs[0, 1].plot(snr_range, acc_mmse, 'd-.', label='MMSE', color='orange')
    axs[0, 1].set_xlabel('SNR (dB)'); axs[0, 1].set_ylabel('Accuracy'); axs[0, 1].set_title('Detection Accuracy vs SNR')
    axs[0, 1].grid(True); axs[0, 1].legend()

    # Throughput Plot
    axs[1, 0].plot(snr_range, th_ae_elm, 'o-', lw=2, label='AE-ELM', color='green')
    axs[1, 0].plot(snr_range, th_mmse, 'd-.', label='MMSE', color='orange')
    axs[1, 0].plot(snr_range, th_ls_zf, 'x-', alpha=0.4, label='LS-ZF', color='gray')
    axs[1, 0].set_xlabel('SNR (dB)'); axs[1, 0].set_ylabel('Throughput (Bits/Symbol)'); axs[1, 0].set_title('Throughput vs SNR')
    axs[1, 0].grid(True); axs[1, 0].legend()

    # Training Time Bar
    models = ['AE-ELM', 'ELM', 'MLP-BP']
    times = [runtimes[m] for m in models]
    axs[1, 1].bar(models, times, color=['green', 'blue', 'red'])
    axs[1, 1].set_ylabel('Time (s)'); axs[1, 1].set_title('Training Computational Complexity'); axs[1, 1].set_yscale('log')
    
from mpl_toolkits.mplot3d import Axes3D

def generate_3d_analysis():
    print("Generating 3D Sensitivity Analysis...")
    
    # 1. Hyper-parameter Sensitivity: Regularization (C) and Hidden Nodes (L)
    # Axes: C (log), L (linear), Performance
    c_values = np.logspace(-10, 10, 10)
    l_values = np.arange(50, 501, 50)
    C, L = np.meshgrid(c_values, l_values)
    Z_perf = np.zeros(C.shape)
    
    # Simulate a fixed environment for this analysis
    test_snr = 15
    train_bits, train_mod = generate_data(1000, n_fft)
    tx_train = ofdm_tx(train_mod, n_fft, n_cp)
    rx_train, _ = channel_effect(tx_train, test_snr, num_receivers)
    fft_train = ofdm_rx_fft(rx_train, n_fft, n_cp)
    x_train_3d = prepare_nn_input(fft_train)
    y_train_3d = prepare_nn_output(train_mod)
    
    test_bits, test_mod = generate_data(500, n_fft)
    tx_test = ofdm_tx(test_mod, n_fft, n_cp)
    rx_test, _ = channel_effect(tx_test, test_snr, num_receivers)
    fft_test = ofdm_rx_fft(rx_test, n_fft, n_cp)
    x_test_3d = prepare_nn_input(fft_test)
    
    input_size = num_receivers * n_fft * 2
    output_size = n_fft * 2
    
    for i in range(len(l_values)):
        for j in range(len(c_values)):
            # Modified ELM with C and L
            l = l_values[i]
            c = c_values[j]
            w_rand = np.random.randn(input_size, l)
            h = 1 / (1 + np.exp(-np.dot(x_train_3d, w_rand)))
            # Regularized Least Squares: beta = (H^T H + I/C)^-1 H^T T
            beta = np.dot(inv(h.T @ h + (1/c) * np.eye(l)), h.T @ y_train_3d)
            
            # Predict
            h_test = 1 / (1 + np.exp(-np.dot(x_test_3d, w_rand)))
            pred = np.dot(h_test, beta)
            
            # Record Accuracy (Relative to best possible for Z-axis effect)
            res = pred[:, :n_fft] + 1j * pred[:, n_fft:]
            bits_extracted = []
            for s in range(500):
                bits_extracted.append(qpsk_demod(res[s]))
            ber = np.mean(np.array(bits_extracted).flatten() != test_bits.flatten())
            Z_perf[i, j] = -np.log10(ber + 1e-6) # Log performance for visualization

    # 2. BER vs SNR and Training Set Size (Figure 13)
    snr_vals = np.arange(0, 16, 3)
    train_sizes = np.linspace(500, 3000, 6, dtype=int)
    SNR_grid, SIZE_grid = np.meshgrid(snr_vals, train_sizes)
    Z_ber = np.zeros(SNR_grid.shape)
    
    for i in range(len(train_sizes)):
        for j in range(len(snr_vals)):
            sz = train_sizes[i]
            snr = snr_vals[j]
            
            # Subset of training data
            train_bits_s, train_mod_s = generate_data(sz, n_fft)
            tx_train_s = ofdm_tx(train_mod_s, n_fft, n_cp)
            rx_train_s, _ = channel_effect(tx_train_s, 20, num_receivers) # high snr train
            fft_train_s = ofdm_rx_fft(rx_train_s, n_fft, n_cp)
            x_s = prepare_nn_input(fft_train_s)
            y_s = prepare_nn_output(train_mod_s)
            
            # Train model
            w_s = np.random.randn(input_size, 512)
            h_s = 1 / (1 + np.exp(-np.dot(x_s, w_s)))
            beta_s = np.dot(pinv(h_s), y_s)
            
            # Test at current SNR
            t_bits, t_mod = generate_data(500, n_fft)
            tx_t = ofdm_tx(t_mod, n_fft, n_cp)
            rx_t, _ = channel_effect(tx_t, snr, num_receivers)
            fft_t = ofdm_rx_fft(rx_t, n_fft, n_cp)
            x_t = prepare_nn_input(fft_t)
            h_t = 1 / (1 + np.exp(-np.dot(x_t, w_s)))
            pred_t = np.dot(h_t, beta_s)
            
            res_t = pred_t[:, :n_fft] + 1j * pred_t[:, n_fft:]
            bits_ext = []
            for s in range(500):
                bits_ext.append(qpsk_demod(res_t[s]))
            ber_val = np.mean(np.array(bits_ext).flatten() != t_bits.flatten())
            Z_ber[i, j] = -np.log10(ber_val + 1e-4) # Plotted as log scale like image

    # Plotting both
    fig = plt.figure(figsize=(20, 8))
    
    # Plot (a)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(np.log10(C), L, Z_perf, cmap='viridis', edgecolor='k', alpha=0.8)
    ax1.set_xlabel('log10(C)'); ax1.set_ylabel('Hidden Nodes (L)'); ax1.set_zlabel('Performance Rate')
    ax1.set_title('(a) Performance vs (C, L) conditions')

    # Plot (b)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(SNR_grid, SIZE_grid, Z_ber, cmap='Greens', edgecolor='k', alpha=0.8)
    ax2.set_xlabel('SNR'); ax2.set_ylabel('Training Set Size'); ax2.set_zlabel('-log10(BER)')
    ax2.set_title('Figure 13. BER vs Training Size & SNR')
    
    plt.tight_layout()
    plt.savefig('/Users/yeshnavya/Desktop/Wireless/sensitivity_analysis_3d.png')
    print("3D Analysis Plots saved to /Users/yeshnavya/Desktop/Wireless/sensitivity_analysis_3d.png")
    plt.show()

if __name__ == "__main__":
    run_simulation()
    generate_3d_analysis()

