# Research Project Planning & Core Topics

This document defines the roadmap and core technologies explored during the development of the AE-ELM Wireless Signal Detection system.

## I. Core Research Topics & Technologies

### 1. Extreme Learning Machine (ELM)
- **Concept**: Single hidden-layer feedforward network with random hidden nodes and analytical output weight solution.
- **Research Goal**: Evaluate computational speed and robustness against traditional iterative neural networks (DNNs).


### 2. Autoencoders (AE) for Wireless Features
- **Concept**: Unsupervised learning of compressed, noise-resistant signal representations.
- **Research Goal**: Extract essential channel state information (CSI) from raw IQ samples without labeled data.
- **Architecture**: Encoders for dimensionality reduction and Decoders for reconstruction validation.

### 3. MIMO-OFDM System Modelling
- **Concept**: Combining Multiple-Input Multiple-Output (MIMO) with Orthogonal Frequency Division Multiplexing (OFDM).
- **Research Goal**: Combat multipath interference and frequency-selective fading in 5G/6G environments.
- **Parameters**: 4x4 MIMO, Cyclic Prefix (CP), and 64-subcarrier OFDM symbols.

### 4. Hybrid AE-ELM Signal Detection
- **Concept**: Two-stage detection (AE Encode -> ELM Classify).
- **Research Goal**: Prove that hybrid data-driven models can outperform linear detectors (ZF/MMSE) in non-linear channel regimes.

---

## II. Development Roadmap

### Phase 1: Literature Review 
- Study Zhao et al. (AE-ELM) and Gündüz (ML in the Air).
- Map mathematical foundations of MIMO signal detection.

### Phase 2: Algorithm Implementation
- Develop core ELM and AE modules in Python.
- Simulate Rayleigh fading channels and AWGN noise models.

### Phase 3: System Extension & OFDM 
- Integrate OFDM processing.
- Perform comparative benchmarking against ZF-SIC and MMSE-SIC.

### Phase 4: Final Evaluation & Extension
- Finalize Bit Error Rate (BER) and Accuracy curves.
- Draft the "Proposed Novel Extension" for adaptive 6G receivers.
