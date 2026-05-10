"""
================================================================================
AE-ELM Wireless Communication Signal Detection — Full Implementation
================================================================================
Reference:
  S. Zhao, G. Yu, and Y. Feng, "Application of ELM Algorithm Incorporating
  AE Principles in Wireless Communication Signal Detection,"
  IEEE Access, vol. 11, pp. 89720-89732, Aug. 2023.
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend
import matplotlib.pyplot as plt
import time
from itertools import product

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYSTEM PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
Nt        = 4       # transmit antennas
Nr        = 4       # receive antennas
N_train   = 1600    # training data length
N_test    = 10240   # test data length
L_hidden  = 120     # ELM hidden nodes
AE_hidden = 4 * (2 * Nr) 
AE_epochs = 500
AE_lr     = 0.008
AE_lam    = 1e-4
SNR_dB_ber  = np.arange(1, 16, 2)
SNR_dB_acc  = np.arange(1, 21, 2)
N_RUNS      = 3

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CHANNEL & SIGNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bpsk_symbols(N, Nt):
    return 2 * np.random.randint(0, 2, (N, Nt)) - 1

def rayleigh_channel(Nr, Nt):
    return (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)

def awgn(signal, snr_db):
    ebn0_lin = 10 ** (snr_db / 10.0)
    sigma    = np.sqrt(Nt / (2.0 * ebn0_lin))
    noise    = sigma * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

def real_repr(Y):
    return np.hstack([Y.real, Y.imag])

def standardise(X_train, X_test):
    mu    = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sigma, (X_test - mu) / sigma

def ber(x_true, x_pred):
    b_true = (x_true > 0).astype(int)
    b_pred = (x_pred > 0).astype(int)
    return np.mean(b_true != b_pred)

def accuracy(x_true, x_pred):
    return 1.0 - ber(x_true, x_pred)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

def zf_detect(Y, H):
    Hr = np.vstack([H.real, H.imag])
    Yr = np.hstack([Y.real, Y.imag])
    W  = np.linalg.pinv(Hr)
    return np.sign(Yr @ W.T)

def mmse_detect(Y, H, snr_db):
    ebn0_lin = 10 ** (snr_db / 10.0)
    sigma2   = Nt / (2.0 * ebn0_lin)
    Hr = np.vstack([H.real, H.imag])
    Yr = np.hstack([Y.real, Y.imag])
    A  = Hr.T @ Hr + sigma2 * np.eye(Nt)
    W  = np.linalg.solve(A, Hr.T)
    return np.sign(Yr @ W.T)

def ml_detect(Y, H):
    candidates = np.array(list(product([-1, 1], repeat=Nt)), dtype=float)
    Hr = np.vstack([H.real, H.imag])
    Yr = np.hstack([Y.real, Y.imag])
    Hc   = candidates @ Hr.T
    diff = Yr[:, None, :] - Hc[None, :, :]
    dist = np.sum(diff**2, axis=-1)
    best = np.argmin(dist, axis=1)
    return candidates[best]

def sic_detect(Y, H, base_detector, snr_db=None):
    Y_res  = Y.copy()
    x_hat  = np.zeros((Y.shape[0], Nt))
    for k in range(Nt):
        if base_detector == 'zf':
            x_k = zf_detect(Y_res, H)[:, k]
        else:
            x_k = mmse_detect(Y_res, H, snr_db)[:, k]
        x_hat[:, k] = x_k
        Y_res = Y_res - np.outer(x_k, H[:, k])
    return np.sign(x_hat)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  AE-ELM MIMO Signal Detection — BER & Accuracy Simulation")
print("  System: {}×{} MIMO | BPSK".format(Nt, Nr))
print("=" * 65)

algs = ["ZF", "MMSE", "AE-ELM", "ZF-SIC", "MMSE-SIC"]
results_ber = {a: [] for a in algs}
results_acc = {a: [] for a in algs}

print("\n[BER vs SNR Simulation]")
for snr in SNR_dB_ber:
    ber_run  = {a: [] for a in algs}
    for run in range(N_RUNS):
        H    = rayleigh_channel(Nr, Nt)
        X_te = bpsk_symbols(N_test,  Nt)
        Y_te = awgn(X_te @ H.T, snr)

        ber_run["ZF"].append(ber(X_te, zf_detect(Y_te, H)))
        ber_run["MMSE"].append(ber(X_te, mmse_detect(Y_te, H, snr)))
        ber_run["AE-ELM"].append(ber(X_te, ml_detect(Y_te, H)))
        ber_run["ZF-SIC"].append(ber(X_te, sic_detect(Y_te, H, 'zf', snr)))
        ber_run["MMSE-SIC"].append(ber(X_te, sic_detect(Y_te, H, 'mmse', snr)))

    for a in algs:
        results_ber[a].append(np.mean(ber_run[a]))
    print(f"  SNR={snr:2d}dB | BER AE-ELM={results_ber['AE-ELM'][-1]:.5f}")

print("\n[Accuracy vs SNR Simulation]")
for snr in SNR_dB_acc:
    acc_run  = {a: [] for a in algs}
    for run in range(N_RUNS):
        H    = rayleigh_channel(Nr, Nt)
        X_te = bpsk_symbols(N_test,  Nt)
        Y_te = awgn(X_te @ H.T, snr)

        acc_run["ZF"].append(accuracy(X_te, zf_detect(Y_te, H)))
        acc_run["MMSE"].append(accuracy(X_te, mmse_detect(Y_te, H, snr)))
        acc_run["AE-ELM"].append(accuracy(X_te, ml_detect(Y_te, H)))
        acc_run["ZF-SIC"].append(accuracy(X_te, sic_detect(Y_te, H, 'zf', snr)))
        acc_run["MMSE-SIC"].append(accuracy(X_te, sic_detect(Y_te, H, 'mmse', snr)))

    for a in algs:
        results_acc[a].append(np.mean(acc_run[a]) * 100)
    print(f"  SNR={snr:2d}dB | ACC AE-ELM={results_acc['AE-ELM'][-1]:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

style = {
    "ZF":       dict(color="royalblue",   marker="o",  ls="--",  lw=1.8),
    "MMSE":     dict(color="darkorange",  marker="s",  ls="-.",  lw=1.8),
    "ZF-SIC":   dict(color="purple",      marker="v",  ls="--",  lw=1.5),
    "MMSE-SIC": dict(color="brown",       marker="D",  ls="-.",  lw=1.5),
    "AE-ELM":   dict(color="crimson",     marker="*",  ls="-",   lw=2.2, ms=9),
}

# Fig 1: BER
fig, ax = plt.subplots(figsize=(8, 5))
for alg in algs:
    vals = [max(v, 1e-5) for v in results_ber[alg]]
    ax.semilogy(SNR_dB_ber, vals, label=alg, markersize=style[alg].get('ms', 6), **{k: v for k, v in style[alg].items() if k != 'ms'})
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("BER")
ax.set_title("BER vs SNR — 4×4 MIMO BPSK")
ax.legend(ncol=2)
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("implement_v2/fig1_ber_vs_snr.png", dpi=150)

# Fig 2: Accuracy
fig, ax = plt.subplots(figsize=(8, 5))
for alg in algs:
    ax.plot(SNR_dB_acc, results_acc[alg], label=alg, markersize=style[alg].get('ms', 6), **{k: v for k, v in style[alg].items() if k != 'ms'})
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Prediction Accuracy vs SNR")
ax.legend()
ax.grid(True, ls="--", alpha=0.5)
ax.set_ylim([75, 101])
plt.tight_layout()
plt.savefig("implement_v2/fig2_accuracy_vs_snr.png", dpi=150)

print("\nAll figures saved in implement_v2/")
