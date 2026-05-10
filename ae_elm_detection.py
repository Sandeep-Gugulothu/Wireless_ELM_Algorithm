"""
================================================================================
AE-ELM Wireless Communication Signal Detection — Full Implementation
================================================================================
Reference:
  S. Zhao, G. Yu, and Y. Feng, "Application of ELM Algorithm Incorporating
  AE Principles in Wireless Communication Signal Detection,"
  IEEE Access, vol. 11, pp. 89720-89732, Aug. 2023.
  DOI: 10.1109/ACCESS.2023.3306470

THEORETICAL BACKGROUND
----------------------
The core problem is MIMO signal detection:
    y = H·x + n
where:
  y ∈ C^(Nr×1)  — received signal vector
  H ∈ C^(Nr×Nt) — channel matrix (Rayleigh flat-fading)
  x ∈ {-1,+1}^Nt — transmitted BPSK symbols
  n ~ CN(0, σ²I) — complex AWGN

The AE-ELM algorithm solves this in two stages:

STAGE 1 — Autoencoder (AE) Feature Extraction [Paper Sec. III-A, Eq. 1-5]
  The AE is a 3-layer neural network (input → hidden → output).
  Encoder:  y = f_θ(x) = sigmoid(W·x + b)          [Eq. 1]
  Decoder:  z = g_θ(y) = sigmoid(W'·x + b')         [Eq. 2]
  Cost:     J_AE(θ) = Σ L(xi, g_θ(f_θ(xi)))        [Eq. 3]
  where L(x,y) = Σ ||xi - zi||²                     [Eq. 5]
  Parameters θ = {W, b, W', b'} optimised by gradient descent [Eq. 4]

  The AE learns a compact, noise-robust representation of the received
  signal WITHOUT labels (unsupervised). The encoder output (hidden layer)
  captures the essential structure of the MIMO channel.

STAGE 2 — Extreme Learning Machine (ELM) Classification [Paper Sec. III-A, Eq. 6-10]
  ELM is a single hidden-layer feedforward network where:
  - Input→hidden weights a_{i,j} and biases b_j are RANDOMLY initialised
    and NEVER updated. This is the key innovation — no iterative training.
  - Hidden layer output: h_j(Xn) = g(Σ_i Xn(i)·a_{i,j} + b_j)  [Eq. 8]
  - Output weights β solved analytically via ridge regression:
    β = T · H^T · (1/C + H·H^T)^{-1}                            [Eq. 9]
  - Final output: f(x) = T·H^T·(1/C + H·H^T)^{-1·g(Wx+b)       [Eq. 10]
  where C is the activation (regularisation) factor.

  Ridge regression (Eq. 9) is used instead of plain least squares to avoid
  ill-conditioned ("sick") matrices, improving stability [Paper p.89724].

AE-ELM INTEGRATION [Paper Sec. III-A, Fig. 3, Eq. 11-14]
  1. Train AE unsupervised on received signals → learn feature mapping
  2. Use AE encoder output as ELM input features
  3. Train ELM supervised with transmitted symbol labels
  4. At test time: received signal → AE encode → ELM classify → detected symbol

MIMO SYSTEM MODEL [Paper Sec. III-B, Fig. 4-6]
  Large-scale MIMO with multipath effect. At low SNR, signal clusters from
  different antennas overlap (Fig. 6b in paper), making linear detection fail.
  AE-ELM learns the nonlinear decision boundary directly from data.

COMPARISON ALGORITHMS [Paper Sec. IV-A]
  - ZF   (Zero Forcing):          x̂ = (H^H·H)^{-1}·H^H·y  — nulls interference
  - MMSE (Min Mean Square Error): x̂ = (H^H·H + σ²I)^{-1}·H^H·y — noise-aware
  - ML   (Maximum Likelihood):    x̂ = argmin_x ||y - H·x||² — optimal, high complexity
  - QGA-RBF: Quantum Genetic Algorithm + Radial Basis Function network
  - ZF-SIC:  ZF + Successive Interference Cancellation
  - MMSE-SIC: MMSE + Successive Interference Cancellation

PAPER'S KEY RESULTS (Table/Fig. in Sec. IV)
  - AE-ELM BER = 0.0004 (best among all)
  - AE-ELM avg relative time complexity = 0.5292s
  - AE-ELM detection error rate = 0.0031 (lowest)
  - AE-ELM avg detection time = 0.219s (fastest)
  - AE-ELM prediction accuracy at 20dB SNR = 95.3%
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves files without display
import matplotlib.pyplot as plt
import time
from itertools import product

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYSTEM PARAMETERS  (matching paper Sec. IV exactly)
# ─────────────────────────────────────────────────────────────────────────────
Nt        = 4       # transmit antennas  (paper: 4×4 MIMO)
Nr        = 4       # receive antennas
N_train   = 1600    # training data length  [paper p.89728]
N_test    = 10240   # test data length      [paper p.89728]
L_hidden  = 120     # ELM hidden nodes, L=120 [paper p.89729, Fig.12a]
# AE hidden nodes: "2 to 4 times the length of the input vector" [paper p.89728]
# Input vector length = 2·Nr (real + imag parts) = 8
AE_hidden = 3 * (2 * Nr)   # 3× → 24 nodes (mid-range of 2–4×)
AE_epochs = 400             # gradient descent epochs for AE
AE_lr     = 0.003           # learning rate
AE_lam    = 1e-4            # L2 regularisation λ  [paper Eq. 13]
# Activation factor C = 1/SNR  [paper p.89729: "set C to 1/snr"]
# SNR range: paper tests 1–15 dB (Fig. 9a) and 1–20 dB (Fig. 11)
SNR_dB_ber  = np.arange(1, 16, 2)    # for BER plot  (1,3,5,...,15)
SNR_dB_acc  = np.arange(1, 21, 2)    # for accuracy plot (1,3,...,19)
N_RUNS      = 3     # average over multiple channel realisations for stability

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CHANNEL & SIGNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bpsk_symbols(N, Nt):
    """
    Generate N rows of Nt BPSK symbols ∈ {-1, +1}.
    BPSK maps bit 0 → -1, bit 1 → +1.
    Used because paper uses BPSK modulation for the 4×4 MIMO system.
    """
    return 2 * np.random.randint(0, 2, (N, Nt)) - 1


def rayleigh_channel(Nr, Nt):
    """
    i.i.d. Rayleigh flat-fading channel matrix H ∈ C^(Nr×Nt).
    Each entry ~ CN(0,1): real and imag parts each ~ N(0, 1/√2).
    Models the multipath wireless channel in the paper's MIMO system
    (Fig. 4-5 in paper). The multipath effect causes inter-stream
    interference that linear detectors struggle to handle.
    """
    return (np.random.randn(Nr, Nt) +
            1j * np.random.randn(Nr, Nt)) / np.sqrt(2)


def awgn(signal, snr_db):
    """
    Add complex AWGN to signal.
    SNR_lin = 10^(SNR_dB/10) = signal_power / noise_power
    Since BPSK symbols have unit power, σ² = 1/SNR_lin.
    Complex noise: n ~ CN(0, σ²I), so each component ~ N(0, σ²/2).
    """
    snr_lin = 10 ** (snr_db / 10.0)
    sigma   = np.sqrt(1.0 / (2.0 * snr_lin))
    noise   = sigma * (np.random.randn(*signal.shape) +
                       1j * np.random.randn(*signal.shape))
    return signal + noise


def real_repr(Y):
    """
    Convert complex received signal Y ∈ C^(N×Nr) to real representation
    by stacking real and imaginary parts → R^(N × 2Nr).
    This is necessary because neural networks operate on real numbers.
    The paper uses this as the AE input vector x ∈ R^d [Eq. 1].
    """
    return np.hstack([Y.real, Y.imag])


def standardise(X_train, X_test):
    """
    Zero-mean, unit-variance standardisation.
    Applied before feeding data to AE to stabilise gradient descent.
    μ and σ computed on training set only (no data leakage).
    """
    mu    = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def ber(x_true, x_pred):
    """
    Bit Error Rate for BPSK: compare sign of true vs predicted.
    BER = (number of wrong bits) / (total bits)
    Paper reports BER as primary performance metric [Sec. IV-A, Fig. 9].
    """
    b_true = (x_true > 0).astype(int)
    b_pred = (x_pred > 0).astype(int)
    return np.mean(b_true != b_pred)


def accuracy(x_true, x_pred):
    """
    Symbol detection accuracy = 1 - BER (for BPSK).
    Paper reports prediction accuracy in Fig. 11 [Sec. IV-A].
    """
    return 1.0 - ber(x_true, x_pred)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  AUTOENCODER  [Paper Sec. III-A, Eq. 1-5, 11-13, Fig. 1]
# ─────────────────────────────────────────────────────────────────────────────

class Autoencoder:
    """
    Three-layer Autoencoder: Input → Hidden (Encoder) → Output (Decoder).

    Architecture [Paper Fig. 1]:
      Input layer  x ∈ R^d  (d = 2·Nr = 8 for 4×4 MIMO)
      Hidden layer y ∈ R^p  (p = AE_hidden, the compressed representation)
      Output layer z ∈ R^d  (reconstruction of input)

    Encoder [Paper Eq. 1, 11]:
      y = f_θ(x) = tanh(W·x + b)
      W ∈ R^(p×d): encoding weight matrix
      b ∈ R^p:     encoding bias

    Decoder [Paper Eq. 2, 12]:
      z = g_θ(y) = tanh(W'·y + b')
      W' ∈ R^(d×p): decoding weight matrix (W' = W^T in tied-weights AE)
      b' ∈ R^d:     decoding bias

    Loss function [Paper Eq. 3, 5, 13]:
      J = Σ||x - z||² + λ·Σ(W² + W'²)
      The first term is reconstruction error (squared Euclidean distance).
      The second term is L2 regularisation with factor λ to prevent overfitting.

    Training [Paper Eq. 4]:
      Parameters θ = {W, b, W', b'} updated by gradient descent.
      After training, ONLY the encoder (W, b) is kept for feature extraction.
      The decoder is discarded — its purpose was only to guide training.

    Why AE for signal detection?
      The AE learns to extract the most informative features of the received
      signal in an UNSUPERVISED way (no labels needed). This is powerful
      because it can work with unlabelled channel data and automatically
      discovers the structure imposed by the MIMO channel H.
    """
    def __init__(self, input_dim, hidden_dim, lam=1e-4):
        # Xavier initialisation: scale = sqrt(2/fan_in) for tanh
        scale_e = np.sqrt(2.0 / input_dim)
        scale_d = np.sqrt(2.0 / hidden_dim)
        self.We  = np.random.randn(input_dim,  hidden_dim) * scale_e  # W
        self.be  = np.zeros(hidden_dim)                                # b
        self.Wd  = np.random.randn(hidden_dim, input_dim)  * scale_d  # W'
        self.bd  = np.zeros(input_dim)                                 # b'
        self.lam = lam   # regularisation factor λ [Paper Eq. 13]
        self.train_errors = []   # track reconstruction error per epoch

    def encode(self, X):
        """Encoder: y = tanh(X·W + b)  [Paper Eq. 11]"""
        return np.tanh(X @ self.We + self.be)

    def decode(self, H):
        """Decoder: z = tanh(H·W' + b')  [Paper Eq. 12]"""
        return np.tanh(H @ self.Wd + self.bd)

    def fit(self, X, lr=0.003, epochs=400, batch=128):
        """
        Mini-batch gradient descent to minimise J [Paper Eq. 13].
        Gradients derived via chain rule (backpropagation):
          ∂J/∂W'  = H^T · δ_dec  +  λ·W'
          ∂J/∂W   = X^T · δ_enc  +  λ·W
        where δ_dec = (z - x) ⊙ (1 - z²)   [tanh derivative]
              δ_enc = (δ_dec · W'^T) ⊙ (1 - y²)
        """
        N = X.shape[0]
        for ep in range(epochs):
            idx      = np.random.permutation(N)
            ep_error = 0.0
            for start in range(0, N, batch):
                xb = X[idx[start:start + batch]]   # mini-batch

                # ── forward pass ──
                y  = self.encode(xb)               # hidden representation
                z  = self.decode(y)                # reconstruction

                # ── reconstruction error ──
                rec_err = z - xb                   # (B, d)
                ep_error += np.sum(rec_err**2)

                # ── decoder gradients ──
                # δ_dec = rec_err ⊙ tanh'(z) = rec_err ⊙ (1 - z²)
                delta_d = rec_err * (1.0 - z**2)
                dWd = y.T @ delta_d / len(xb) + self.lam * self.Wd
                dbd = delta_d.mean(axis=0)

                # ── encoder gradients ──
                # δ_enc = (δ_dec · W'^T) ⊙ (1 - y²)
                # delta_d: (B, input_dim=8), self.Wd: (hidden=24, input_dim=8)
                # Need to propagate back: delta_d @ self.Wd.T → (B, hidden=24)
                delta_e = (delta_d @ self.Wd.T) * (1.0 - y**2)
                dWe = xb.T @ delta_e / len(xb) + self.lam * self.We
                dbe = delta_e.mean(axis=0)

                # ── parameter update ──
                self.Wd -= lr * dWd
                self.bd -= lr * dbd
                self.We -= lr * dWe
                self.be -= lr * dbe

            self.train_errors.append(ep_error / N)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EXTREME LEARNING MACHINE  [Paper Sec. III-A, Eq. 6-10, Fig. 2]
# ─────────────────────────────────────────────────────────────────────────────

class ELM:
    """
    Single Hidden-Layer Feedforward Network with Extreme Learning.

    Architecture [Paper Fig. 2]:
      Input layer  x ∈ R^d
      Hidden layer h ∈ R^L  (L = number of hidden nodes)
      Output layer o ∈ R^m  (m = number of output classes)

    Key innovation: Input weights a_{i,j} and biases b_j are RANDOMLY
    INITIALISED and NEVER UPDATED. Only output weights β are learned.
    This eliminates iterative backpropagation entirely.

    Hidden layer output [Paper Eq. 7, 8]:
      H = [h(x1); h(x2); ...; h(xN)]  ∈ R^(N×L)
      h_j(Xn) = g(Σ_i Xn(i)·a_{i,j} + b_j)
      where g(·) = tanh (chosen as best activation in paper Fig. 12b)

    Output weight solution [Paper Eq. 9] — Ridge Regression:
      β = T · H^T · (1/C + H·H^T)^{-1}
    Equivalently (more numerically stable for L < N):
      β = (1/C·I + H^T·H)^{-1} · H^T · T

    Ridge regression adds 1/C to the diagonal of H^T·H, preventing
    the "sick matrix" (ill-conditioned) problem [Paper p.89724].
    C is the activation factor — paper sets C = 1/SNR [p.89729].

    Output function [Paper Eq. 10]:
      f(x) = T·H^T·(1/C + H·H^T)^{-1}·g(Wx + b)

    Why ELM is fast:
      Training = one matrix inversion (O(L³)) vs backprop (O(epochs·N·L)).
      For L=120, N=1600: ELM solves in milliseconds.
    """
    def __init__(self, n_input, n_hidden, C=1.0):
        # Random input weights a_{i,j} [Paper: "randomly initialised"]
        self.W    = np.random.randn(n_input, n_hidden) * np.sqrt(1.0/n_input)
        self.b    = np.random.randn(n_hidden)           # random biases b_j
        self.C    = C       # activation factor [Paper Eq. 9]
        self.beta = None    # output weights β (learned analytically)

    def _hidden_output(self, X):
        """
        Compute hidden layer matrix H [Paper Eq. 7, 8].
        H[n,j] = tanh(Σ_i X[n,i]·W[i,j] + b[j])
        """
        return np.tanh(X @ self.W + self.b)   # (N, L)

    def fit(self, X, T):
        """
        Solve for output weights β analytically [Paper Eq. 9].
        Using the numerically stable form: β = (I/C + H^T·H)^{-1}·H^T·T
        This is equivalent to the paper's formula but avoids inverting
        the larger (N×N) matrix when L < N (which is our case: L=120 < N=1600).
        """
        H = self._hidden_output(X)                      # (N, L)
        A = np.eye(H.shape[1]) / self.C + H.T @ H      # (L, L)
        t = T if T.ndim > 1 else T[:, None]             # ensure 2D
        self.beta = np.linalg.solve(A, H.T @ t)         # (L, m)

    def predict(self, X):
        """Forward pass at test time: f(x) = H·β"""
        H = self._hidden_output(X)
        return (H @ self.beta).squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BASELINE DETECTORS  [Paper Sec. IV-A, comparison algorithms]
# ─────────────────────────────────────────────────────────────────────────────

def zf_detect(Y, H):
    """
    Zero-Forcing (ZF) Detector.
    Inverts the channel effect by multiplying with the pseudo-inverse of H:
      x̂ = H^† · y = (H^H·H)^{-1}·H^H · y
    Completely nulls inter-stream interference but amplifies noise,
    especially at low SNR. This is why it performs poorly at low SNR.
    [Paper: traditional linear algorithm, comparison baseline]
    """
    # Work in real-valued domain: stack [Re(H); Im(H)] and [Re(y), Im(y)]
    Hr = np.vstack([H.real, H.imag])          # (2Nr, Nt)
    Yr = np.hstack([Y.real, Y.imag])          # (N, 2Nr)
    W  = np.linalg.pinv(Hr)                   # (Nt, 2Nr) — pseudo-inverse
    return np.sign(Yr @ W.T)                  # hard decision


def mmse_detect(Y, H, snr_db):
    """
    Minimum Mean Square Error (MMSE) Detector.
    Balances interference cancellation with noise enhancement:
      x̂ = (H^H·H + σ²·I)^{-1}·H^H · y
    The σ²·I term (= I/SNR) prevents noise amplification.
    Better than ZF at low SNR but still linear — cannot handle
    the nonlinear decision boundaries in complex MIMO channels.
    [Paper: traditional linear algorithm, comparison baseline]
    """
    snr_lin = 10 ** (snr_db / 10.0)
    sigma2  = 1.0 / snr_lin
    Hr = np.vstack([H.real, H.imag])          # (2Nr, Nt)
    Yr = np.hstack([Y.real, Y.imag])          # (N, 2Nr)
    A  = Hr.T @ Hr + sigma2 * np.eye(Nt)      # (Nt, Nt)
    W  = np.linalg.solve(A, Hr.T)             # (Nt, 2Nr)
    return np.sign(Yr @ W.T)


def ml_detect(Y, H):
    """
    Maximum Likelihood (ML) Detector — optimal but exponential complexity.
    Exhaustive search over all 2^Nt BPSK symbol combinations:
      x̂ = argmin_{x ∈ {-1,+1}^Nt} ||y - H·x||²
    For Nt=4: only 2^4=16 candidates → feasible.
    For large Nt (e.g., 64 in massive MIMO): completely intractable.
    ML gives the theoretical BER lower bound.
    [Paper: comparison baseline, Fig. 8a, 9a]
    """
    candidates = np.array(list(product([-1, 1], repeat=Nt)),
                          dtype=float)                    # (16, 4)
    Hr = np.vstack([H.real, H.imag])                     # (2Nr, Nt)
    Yr = np.hstack([Y.real, Y.imag])                     # (N, 2Nr)
    # Compute ||y - H·c||² for all N samples and 16 candidates
    Hc   = candidates @ Hr.T                             # (16, 2Nr)
    diff = Yr[:, None, :] - Hc[None, :, :]              # (N, 16, 2Nr)
    dist = np.sum(diff**2, axis=-1)                      # (N, 16)
    best = np.argmin(dist, axis=1)                       # (N,)
    return candidates[best]                              # (N, Nt)


def sic_detect(Y, H, base_detector, snr_db=None):
    """
    Successive Interference Cancellation (SIC).
    Detects one stream at a time, subtracts its contribution, then
    detects the next stream on the residual signal.
    Used with ZF (ZF-SIC) and MMSE (MMSE-SIC) in the paper.

    Algorithm:
      For k = 1 to Nt:
        1. Detect x_k using base detector on residual y_res
        2. Subtract: y_res = y_res - H[:,k]·x̂_k
    SIC improves over plain ZF/MMSE by removing already-detected
    interference, but error propagation can degrade performance.
    [Paper: comparison baselines ZF-SIC and MMSE-SIC, Fig. 8b, 9b]
    """
    Y_res  = Y.copy()
    x_hat  = np.zeros((Y.shape[0], Nt))
    order  = list(range(Nt))   # detection order (can be optimised)

    for k in order:
        # Detect stream k using base detector on residual
        if base_detector == 'zf':
            x_k = zf_detect(Y_res, H)[:, k]
        else:  # mmse
            x_k = mmse_detect(Y_res, H, snr_db)[:, k]
        x_hat[:, k] = x_k
        # Subtract detected stream k from received signal
        # y_res = y_res - H[:,k] · x̂_k  (for each sample)
        Y_res = Y_res - np.outer(x_k, H[:, k])

    return np.sign(x_hat)


def qga_rbf_detect(Y_train_r, X_train, Y_test_r, snr_db):
    """
    QGA-RBF: Quantum Genetic Algorithm + Radial Basis Function network.
    Approximation: RBF network trained with standard gradient descent.
    The QGA in the paper optimises RBF centres/widths; here we use
    k-means initialisation as a practical substitute.
    [Paper: comparison baseline, Fig. 8b, 9b]
    """
    from scipy.spatial.distance import cdist

    N, d = Y_train_r.shape
    n_centres = min(L_hidden, N // 10)

    # K-means initialisation for RBF centres
    idx     = np.random.choice(N, n_centres, replace=False)
    centres = Y_train_r[idx].copy()
    sigma   = np.median(cdist(centres, centres)) + 1e-8

    def rbf_features(X):
        D = cdist(X, centres)          # (N, K)
        return np.exp(-D**2 / (2 * sigma**2))

    # Train output weights via ridge regression (same as ELM)
    C_val = 1.0 / (10 ** (snr_db / 10.0))
    Phi   = rbf_features(Y_train_r)    # (N_train, K)
    A     = np.eye(n_centres) / C_val + Phi.T @ Phi
    T     = X_train.astype(float)
    beta  = np.linalg.solve(A, Phi.T @ T)

    Phi_test = rbf_features(Y_test_r)
    return np.sign(Phi_test @ beta)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  AE-ELM DETECTOR  [Paper Sec. III-A, Fig. 3, Eq. 11-14]
# ─────────────────────────────────────────────────────────────────────────────

def ae_elm_detect(Y_train_r, X_train, Y_test_r, snr_db, return_ae=False):
    """
    Full AE-ELM detection pipeline [Paper Fig. 3]:

    TRAINING PHASE:
      Step 1 [Paper Eq. 11-13]: Train AE on received signals (unsupervised).
        - AE learns to reconstruct Y_train from its compressed representation.
        - Gradient descent minimises J = ||Y - Ŷ||² + λ||W||²
        - After convergence, encoder weights (W, b) are frozen.

      Step 2 [Paper Eq. 14]: Train ELM on AE features (supervised).
        - Extract features: F_train = AE_encode(Y_train)
        - Solve: β = (I/C + F^T·F)^{-1}·F^T·X_train
        - C = 1/SNR as recommended in paper [p.89729]

    DETECTION PHASE:
      Step 3: For test signal Y_test:
        - Extract features: F_test = AE_encode(Y_test)
        - Detect: x̂ = sign(F_test · β)

    One ELM per transmit antenna (Nt=4 ELMs total).
    This is consistent with the paper's multi-class detection approach.
    """
    t0 = time.time()

    # ── Step 1: AE feature extraction (unsupervised) ──
    ae = Autoencoder(Y_train_r.shape[1], AE_hidden, lam=AE_lam)
    ae.fit(Y_train_r, lr=AE_lr, epochs=AE_epochs, batch=128)

    feat_train = ae.encode(Y_train_r)   # (N_train, AE_hidden)
    feat_test  = ae.encode(Y_test_r)    # (N_test,  AE_hidden)

    # ── Step 2: ELM classification (supervised, per antenna) ──
    C_val = 1.0 / (10 ** (snr_db / 10.0))   # C = 1/SNR [paper p.89729]
    preds = np.zeros((Y_test_r.shape[0], Nt))

    for ant in range(Nt):
        elm = ELM(feat_train.shape[1], L_hidden, C=C_val)
        elm.fit(feat_train, X_train[:, ant].astype(float))
        preds[:, ant] = np.sign(elm.predict(feat_test))

    elapsed = time.time() - t0

    if return_ae:
        return preds, elapsed, ae
    return preds, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAINING ERROR COMPARISON  [Paper Sec. IV-A, Fig. 8]
# ─────────────────────────────────────────────────────────────────────────────

def compute_training_errors(Y_train_r, X_train, n_iter=200):
    """
    Simulate iterative training error curves for all algorithms.
    Paper Fig. 8 shows training error vs number of iterations (1-1000).
    We track MSE between predicted and true symbols over training steps.

    For AE-ELM: track AE reconstruction error per epoch.
    For ZF/MMSE/ML: these are non-iterative, so we simulate their
    equivalent 'error' as the detection MSE at each data subset size.
    """
    print("\n[Training Error Curves]")
    iters = np.linspace(1, n_iter, 50, dtype=int)

    # AE-ELM: track reconstruction error during AE training
    ae_errors = []
    ae = Autoencoder(Y_train_r.shape[1], AE_hidden, lam=AE_lam)
    for ep in range(n_iter):
        ae.fit(Y_train_r, lr=AE_lr, epochs=1, batch=128)
        rec = ae.decode(ae.encode(Y_train_r))
        err = np.mean((rec - Y_train_r)**2)
        ae_errors.append(err)

    # Baselines: compute detection MSE on growing subsets
    H = rayleigh_channel(Nr, Nt)
    zf_errors, mmse_errors, ml_errors = [], [], []
    snr_fixed = 10  # fixed SNR for training error comparison

    for it in iters:
        n = max(10, int(it / n_iter * len(Y_train_r)))
        Ys = Y_train_r[:n]
        Xs = X_train[:n]
        # Reconstruct complex Y for baselines
        Yc = Ys[:, :Nr] + 1j * Ys[:, Nr:]

        xzf  = zf_detect(Yc, H)
        xmm  = mmse_detect(Yc, H, snr_fixed)
        xml  = ml_detect(Yc, H)

        zf_errors.append(np.mean((xzf  - Xs)**2))
        mmse_errors.append(np.mean((xmm - Xs)**2))
        ml_errors.append(np.mean((xml  - Xs)**2))

    return (list(range(1, n_iter+1)), ae_errors,
            iters, zf_errors, mmse_errors, ml_errors)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  AE-ELM MIMO Signal Detection — Full Simulation")
print("  Paper: Zhao et al., IEEE Access 2023")
print("  System: {}×{} MIMO | BPSK | N_train={} | N_test={}".format(
      Nt, Nr, N_train, N_test))
print("=" * 65)

# ── Storage ──
algs_ber = ["ZF", "MMSE", "ML", "ZF-SIC", "MMSE-SIC", "QGA-RBF", "AE-ELM"]
results_ber = {a: [] for a in algs_ber}
results_acc = {a: [] for a in algs_ber}
ae_elm_times = []

# ── BER vs SNR (1–15 dB, paper Fig. 9) ──
print("\n[BER vs SNR Simulation]")
for snr in SNR_dB_ber:
    ber_run  = {a: [] for a in algs_ber}
    time_run = []

    for run in range(N_RUNS):
        H    = rayleigh_channel(Nr, Nt)
        X_tr = bpsk_symbols(N_train, Nt)
        Y_tr = awgn(X_tr @ H.T, snr)
        X_te = bpsk_symbols(N_test,  Nt)
        Y_te = awgn(X_te @ H.T, snr)

        Y_tr_r, Y_te_r = standardise(real_repr(Y_tr), real_repr(Y_te))

        ber_run["ZF"].append(ber(X_te, zf_detect(Y_te, H)))
        ber_run["MMSE"].append(ber(X_te, mmse_detect(Y_te, H, snr)))
        ber_run["ML"].append(ber(X_te, ml_detect(Y_te, H)))
        ber_run["ZF-SIC"].append(ber(X_te, sic_detect(Y_te, H, 'zf', snr)))
        ber_run["MMSE-SIC"].append(ber(X_te, sic_detect(Y_te, H, 'mmse', snr)))
        ber_run["QGA-RBF"].append(ber(X_te, qga_rbf_detect(Y_tr_r, X_tr, Y_te_r, snr)))

        x_ae, t_ae = ae_elm_detect(Y_tr_r, X_tr, Y_te_r, snr)
        ber_run["AE-ELM"].append(ber(X_te, x_ae))
        time_run.append(t_ae)

    for a in algs_ber:
        results_ber[a].append(np.mean(ber_run[a]))
    ae_elm_times.append(np.mean(time_run))

    print(f"  SNR={snr:2d}dB | "
          f"ZF={results_ber['ZF'][-1]:.4f}  "
          f"MMSE={results_ber['MMSE'][-1]:.4f}  "
          f"ML={results_ber['ML'][-1]:.5f}  "
          f"AE-ELM={results_ber['AE-ELM'][-1]:.4f}  "
          f"(t={ae_elm_times[-1]:.2f}s)")

avg_time = np.mean(ae_elm_times)
print(f"\n  Avg AE-ELM training time: {avg_time:.4f} s")
print(f"  Paper reports: 0.5292 s  (our result: {avg_time:.4f} s)")

# ── Accuracy vs SNR (1–20 dB, paper Fig. 11) ──
print("\n[Prediction Accuracy vs SNR (1-20 dB)]")
for snr in SNR_dB_acc:
    acc_run = {a: [] for a in algs_ber}
    for run in range(N_RUNS):
        H    = rayleigh_channel(Nr, Nt)
        X_tr = bpsk_symbols(N_train, Nt)
        Y_tr = awgn(X_tr @ H.T, snr)
        X_te = bpsk_symbols(N_test,  Nt)
        Y_te = awgn(X_te @ H.T, snr)
        Y_tr_r, Y_te_r = standardise(real_repr(Y_tr), real_repr(Y_te))

        acc_run["ZF"].append(accuracy(X_te, zf_detect(Y_te, H)))
        acc_run["MMSE"].append(accuracy(X_te, mmse_detect(Y_te, H, snr)))
        acc_run["ML"].append(accuracy(X_te, ml_detect(Y_te, H)))
        acc_run["ZF-SIC"].append(accuracy(X_te, sic_detect(Y_te, H, 'zf', snr)))
        acc_run["MMSE-SIC"].append(accuracy(X_te, sic_detect(Y_te, H, 'mmse', snr)))
        acc_run["QGA-RBF"].append(accuracy(X_te, qga_rbf_detect(Y_tr_r, X_tr, Y_te_r, snr)))
        x_ae, _ = ae_elm_detect(Y_tr_r, X_tr, Y_te_r, snr)
        acc_run["AE-ELM"].append(accuracy(X_te, x_ae))

    for a in algs_ber:
        results_acc[a].append(np.mean(acc_run[a]) * 100)

    print(f"  SNR={snr:2d}dB | "
          f"ZF={results_acc['ZF'][-1]:.1f}%  "
          f"MMSE={results_acc['MMSE'][-1]:.1f}%  "
          f"AE-ELM={results_acc['AE-ELM'][-1]:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PLOTS  [Reproducing Paper Fig. 9, 10, 11]
# ─────────────────────────────────────────────────────────────────────────────

style = {
    "ZF":       dict(color="royalblue",   marker="o",  ls="--",  lw=1.8),
    "MMSE":     dict(color="darkorange",  marker="s",  ls="-.",  lw=1.8),
    "ML":       dict(color="green",       marker="^",  ls=":",   lw=1.8),
    "ZF-SIC":   dict(color="purple",      marker="v",  ls="--",  lw=1.5),
    "MMSE-SIC": dict(color="brown",       marker="D",  ls="-.",  lw=1.5),
    "QGA-RBF":  dict(color="gray",        marker="x",  ls=":",   lw=1.5),
    "AE-ELM":   dict(color="crimson",     marker="*",  ls="-",   lw=2.2, ms=9),
}

# ── Fig 1: BER vs SNR ──
fig, ax = plt.subplots(figsize=(8, 5))
for alg in algs_ber:
    vals = [max(v, 1e-5) for v in results_ber[alg]]
    ax.semilogy(SNR_dB_ber, vals, label=alg,
                markersize=style[alg].get('ms', 6), **{k: v for k, v in style[alg].items() if k != 'ms'})
ax.set_xlabel("SNR (dB)", fontsize=12)
ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
ax.set_title("BER vs SNR — 4×4 MIMO BPSK\n"
             "Reproducing Zhao et al. IEEE Access 2023, Fig. 9", fontsize=11)
ax.legend(fontsize=10, ncol=2)
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("fig1_ber_vs_snr.png", dpi=150)
print("\nSaved: fig1_ber_vs_snr.png")

# ── Fig 2: Prediction Accuracy vs SNR ──
fig, ax = plt.subplots(figsize=(8, 5))
for alg in ["ZF", "MMSE", "ZF-SIC", "MMSE-SIC", "AE-ELM"]:
    ax.plot(SNR_dB_acc, results_acc[alg], label=alg,
            markersize=style[alg].get('ms', 6), **{k: v for k, v in style[alg].items() if k != 'ms'})
ax.set_xlabel("SNR (dB)", fontsize=12)
ax.set_ylabel("Prediction Accuracy (%)", fontsize=12)
ax.set_title("Prediction Accuracy vs SNR — 4×4 MIMO BPSK\n"
             "Reproducing Zhao et al. IEEE Access 2023, Fig. 11", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, ls="--", alpha=0.5)
ax.set_ylim([75, 101])
plt.tight_layout()
plt.savefig("fig2_accuracy_vs_snr.png", dpi=150)
print("Saved: fig2_accuracy_vs_snr.png")

# ── Fig 3: Relative Time Complexity ──
# Paper Fig. 10: relative time complexity vs number of operations (1-10)
# We simulate by running each detector 10 times and recording time
print("\n[Time Complexity Comparison]")
n_ops   = 10
H_fixed = rayleigh_channel(Nr, Nt)
X_fixed = bpsk_symbols(N_test, Nt)
Y_fixed = awgn(X_fixed @ H_fixed.T, 10)
Y_fixed_r, _ = standardise(real_repr(Y_fixed), real_repr(Y_fixed))
X_tr_f  = bpsk_symbols(N_train, Nt)
Y_tr_f  = awgn(X_tr_f @ H_fixed.T, 10)
Y_tr_r_f, _ = standardise(real_repr(Y_tr_f), real_repr(Y_tr_f))

time_records = {a: [] for a in ["MMSE", "ZF-SIC", "MMSE-SIC", "AE-ELM"]}
for op in range(n_ops):
    t0 = time.time(); mmse_detect(Y_fixed, H_fixed, 10);  time_records["MMSE"].append(time.time()-t0)
    t0 = time.time(); sic_detect(Y_fixed, H_fixed, 'zf', 10); time_records["ZF-SIC"].append(time.time()-t0)
    t0 = time.time(); sic_detect(Y_fixed, H_fixed, 'mmse', 10); time_records["MMSE-SIC"].append(time.time()-t0)
    _, t_ae = ae_elm_detect(Y_tr_r_f, X_tr_f, Y_fixed_r, 10)
    time_records["AE-ELM"].append(t_ae)

fig, ax = plt.subplots(figsize=(8, 4))
ops_x = np.arange(1, n_ops+1)
tc_style = {
    "MMSE":     dict(color="darkorange", marker="s", ls="-.", lw=1.8),
    "ZF-SIC":   dict(color="purple",     marker="v", ls="--", lw=1.5),
    "MMSE-SIC": dict(color="brown",      marker="D", ls="-.", lw=1.5),
    "AE-ELM":   dict(color="crimson",    marker="*", ls="-",  lw=2.2, ms=9),
}
for alg, times in time_records.items():
    ax.plot(ops_x, times, label=alg,
            markersize=tc_style[alg].get('ms', 6),
            **{k: v for k, v in tc_style[alg].items() if k != 'ms'})
ax.set_xlabel("Number of Operations", fontsize=12)
ax.set_ylabel("Time (s)", fontsize=12)
ax.set_title("Relative Time Complexity per Operation\n"
             "Reproducing Zhao et al. IEEE Access 2023, Fig. 10", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("fig3_time_complexity.png", dpi=150)
print("Saved: fig3_time_complexity.png")

# ── Summary Table ──
print("\n" + "=" * 65)
print("  SUMMARY TABLE")
print("=" * 65)
print(f"  {'Algorithm':<12} {'BER@15dB':>10} {'Acc@20dB':>10} {'Avg Time(s)':>12}")
print("  " + "-" * 50)
for alg in algs_ber:
    b_val = results_ber[alg][-1] if results_ber[alg] else float('nan')
    a_val = results_acc[alg][-1] if results_acc[alg] else float('nan')
    t_val = avg_time if alg == "AE-ELM" else float('nan')
    t_str = f"{t_val:.4f}" if not np.isnan(t_val) else "  N/A"
    print(f"  {alg:<12} {b_val:>10.5f} {a_val:>9.1f}% {t_str:>12}")
print("=" * 65)
print("\nAll figures saved. Simulation complete.")

