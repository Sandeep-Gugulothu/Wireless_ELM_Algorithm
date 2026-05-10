import numpy as np
import matplotlib.pyplot as plt
from ae_elm_system import MIMO_OFDM_System, ELM_AE, ELM_Classifier
from sklearn.preprocessing import StandardScaler

def run_simulation():
    # 1. System Initialization
    n_tx, n_rx, n_fft = 2, 2, 64
    system = MIMO_OFDM_System(n_tx=n_tx, n_rx=n_rx, n_fft=n_fft)
    
    # Generate a fixed channel for this run
    H_fixed = (np.random.randn(n_rx, n_tx) + 1j * np.random.randn(n_rx, n_tx)) / np.sqrt(2)
    
    # 2. Data Preparation
    n_train_blocks = 500 # -> 500 * 64 = 32000 samples
    train_snr = 20
    
    print(f"Generating Training Data (SNR={train_snr}dB)...")
    X_train, T_train, _ = system.generate_data(n_train_blocks, train_snr, H_fixed)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    # 3. Model Initialization (Following Fig 7)
    input_dim = X_train.shape[1]
    ae_hidden = 128
    elm_hidden = 512
    
    print("Building ELM-AE for Feature Mapping...")
    ae = ELM_AE(input_dim, ae_hidden)
    ae.train(X_train_s, C=1.0)
    
    ae_feats_train = ae.extract_features(X_train_s)
    
    print("Building ELM Classifier for Signal Detection...")
    detector = ELM_Classifier(ae_feats_train.shape[1], elm_hidden, T_train.shape[1], C=1.0)
    detector.train(ae_feats_train, T_train)
    
    # 4. Evaluation
    snr_range = [0, 5, 10, 15, 20]
    ae_elm_bers = []
    zf_bers = []
    
    print(f"{'SNR(dB)':>10} | {'ZF BER':>10} | {'AE-ELM BER':>10}")
    print("-" * 40)
    
    H_inv = np.linalg.pinv(H_fixed)

    for snr in snr_range:
        n_test_blocks = 200
        X_test, T_test, _ = system.generate_data(n_test_blocks, snr, H_fixed)
        X_test_s = scaler.transform(X_test)
        
        # AE-ELM Path
        ae_feats_test = ae.extract_features(X_test_s)
        preds = (detector.predict(ae_feats_test) > 0.5).astype(int)
        ae_elm_ber = np.mean(preds != T_test)
        ae_elm_bers.append(ae_elm_ber)
        
        # ZF Baseline (Calculated for the subcarrier-reshaped X_test)
        y_complex = X_test[:, :n_rx] + 1j * X_test[:, n_rx:]
        x_hat = np.dot(y_complex, H_inv.T)
        
        zf_preds = np.zeros(T_test.shape)
        zf_preds[:, 0::2] = (x_hat.real > 0).astype(int)
        zf_preds[:, 1::2] = (x_hat.imag > 0).astype(int)
        zf_ber = np.mean(zf_preds != T_test)
        zf_bers.append(zf_ber)
        
        print(f"{snr:10d} | {zf_ber:10.6f} | {ae_elm_ber:10.6f}")

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, zf_bers, 'o-', label='Zero Forcing (Baseline)')
    plt.semilogy(snr_range, ae_elm_bers, 's-', label='AE-ELM Detector')
    plt.grid(True, which="both", ls="-")
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('MIMO-OFDM Signal Detection: Subcarrier-level AE-ELM')
    plt.legend()
    plt.savefig('ber_performance_v2.png')
    print("Simulation complete. Results saved to ber_performance_v2.png")

if __name__ == "__main__":
    run_simulation()
