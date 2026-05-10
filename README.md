# Application of AE-ELM Algorithm in Wireless Signal Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

This repository contains the complete implementation and simulation framework for a high-performance wireless signal detection system utilizing a hybrid **Autoencoder (AE) and Extreme Learning Machine (ELM)** architecture.

## 📌 Project Overview
The research focuses on the challenges of signal detection in Multiple-Input Multiple-Output (MIMO) systems, especially under low Signal-to-Noise Ratio (SNR) and nonlinear channel conditions. Traditional linear detectors like Zero-Forcing (ZF) and Minimum Mean Square Error (MMSE) often struggle with noise enhancement and interference in these regimes.

**Proposed Solution:** A two-stage hybrid model:
1.  **Autoencoder (AE):** Performs unsupervised feature extraction to learn a noise-robust representation of the received MIMO signal.
2.  **Extreme Learning Machine (ELM):** Provides ultra-fast supervised classification with analytical weight calculation (Ridge Regression), bypassing the need for slow iterative backpropagation.

## 🚀 Key Features
- **MIMO-OFDM Support**: Simulation of 4x4 MIMO channels with Rayleigh flat-fading and wideband OFDM effects.
- **Hybrid DL Architecture**: Seamless integration of AE-based dimensionality reduction and ELM-based signal estimation.
- **Comprehensive Benchmarking**: Comparative analysis against ZF, MMSE, ML, ZF-SIC, and MMSE-SIC.
- **Visual Analytics**: Automated generation of BER vs SNR and Prediction Accuracy vs SNR plots.

## 📊 Performance Summary
| Algorithm | BER @ 15dB | Prediction Accuracy | Relative Complexity |
| :--- | :--- | :--- | :--- |
| **AE-ELM** | **0.0004** | **95.3%** | **Low (Analytical)** |
| MMSE-SIC | 0.0012 | 91.2% | Medium |
| ZF-SIC | 0.0054 | 86.5% | Medium |
| ML (Optimal) | 0.0001 | 98.0% | High (Exponential) |

## 📁 Repository Structure
- `ae_elm_detection.py`: Primary simulation script (600+ lines) including the full AE-ELM pipeline.
- `ofdm_ae_elm_comparison.py`: Specialized script for OFDM-integrated signal detection.
- `planning.md`: Project roadmap and theoretical foundations.
- `Images/`: Generated plots and system architecture diagrams.
- `implement_v2/`: Optimized implementations of the signal detection engine.

## 🛠 Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sandeep-Gugulothu/Wireless_ELM_Algorithm.git
   ```
2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib scipy
   ```
3. **Run the main simulation:**
   ```bash
   python ae_elm_detection.py
   ```

## 📚 References
Based on the research paper:
> S. Zhao, G. Yu, and Y. Feng, "Application of ELM Algorithm Incorporating AE Principles in Wireless Communication Signal Detection," in *IEEE Access*, vol. 11, pp. 89720-89732, 2023.

---
**Developed by Sandeep Gugulothu as part of the Wireless Communication Research Project.**
