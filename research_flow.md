# Wireless Research Workflow & Technical Progression

This document outlines the systematic research flow from basic signal detection to advanced integrated MIMO-OFDM architectures.

## Phase 1: MIMO Signal Detection Foundation
**Source**: `ae_elm_detection.py`
- **Objective**: Implement and benchmark the core AE-ELM architecture.
- **Environment**: 4x4 Rayleigh flat-fading channel with BPSK modulation.
- **Key Milestones**:
    - Developed unsupervised Autoencoder for robust feature extraction from noisy complex signals.
    - Implemented ELM for fast, non-iterative classification.
    - Benchmarked against ZF, MMSE, ML, and SIC detectors, confirming near-optimal performance at low complexity.

## Phase 2: Transition to Wideband OFDM
**Source**: `ofdm_ae_elm_comparison.py`
- **Objective**: Extend the detection framework to frequency-selective channels.
- **Implementation**:
    - Integrated OFDM subcarrier mapping and IFFT/FFT processing.
    - Added Cyclic Prefix (CP) management to handle multi-path delay spread.
    - Evaluated AE-ELM performance across multiple subcarriers.
- **Outcome**: Validated that AE-ELM effectively learns decision boundaries even in multi-carrier interference scenarios.

## Phase 3: Full System Integration & 6G Extensions
**Source**: `implement_v2/ae_elm_full.py`
- **Objective**: Finalize a robust, adaptive receiver model for high-mobility scenarios.
- **Advanced Features**:
    - **Sliding Window Detection**: Handling non-stationary channel conditions.
    - **Sensitivity Analysis**: Stress-testing the model against SNR variations and channel estimation errors.
    - **Proposed Extension**: Initial roadmap for Attention-Assisted and Online Sparse ELM (OS-ELM) to support 6G massive MIMO.

## Phase 4: Final Manuscript & Documentation
- Integration of all simulation results into the IEEE term paper.
- Documenting the Python implementation and repository structure for reproducibility.
