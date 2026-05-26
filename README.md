# HyTAM-Net: Hybrid Topology-Aware Network for Coastal Wind Field Reconstruction

This repository contains the official implementation of the paper: **"Reconstruction of High-Precision Coastal Wind Fields via Topography-Aware Error Correction of ERA5 Data"**.

---

###  Important Notice 

 **Current Status:** To comply with the double-blind review process and protect ongoing research, **only partial data preprocessing scripts and baseline model implementations are currently provided** in this repository. 

The full production code (including the complete parallel dual-stream Mamba-Transformer architecture, direction-aware cross-attention mechanism, and anisotropic fetch dictionary modules) along with the matched verification datasets will be fully released immediately upon the formal acceptance of the paper.


---

## 1. Introduction

Formulating high-precision nearshore ocean wind fields is vital for offshore wind resource assessment and marine meteorological warning systems. However, coarse reanalysis products like the European Centre for Medium-Range Weather Forecasts Reanalysis 5 (ERA5) frequently suffer from topographic contamination in complex coastal zones, exhibiting systematic biases in the overall wind background and dynamic variations.

**HyTAM-Net** addresses these challenges through three core designs:
1. **Spatiotemporal Decoupling:** A parallel dual-stream Mamba-Transformer architecture achieving physical frequency-division decoupling of low-frequency climatic evolution and high-frequency gusts.
2. **Topography-Aware Interaction:** A direction-aware cross-attention mechanism extracting topographic blocking constraints from an anisotropic fetch dictionary.
3. **Physical Consistency:** A dual-track affine modulation pathway incorporating positive-bounded time-varying scaling factors and localized baseline residuals to eliminate non-physical predictions (e.g., negative wind speeds).

---

## 2. Repository Structure (Current vs. Full Release)

```text
HyTAM-Net/
├── data_preprocessing/             
│   └── partial_preprocess_demo.py   <-- [Available Now] Basic coordinate alignment and missing value QC
├── models/                         
│   ├── baselines.py                 <-- [Available Now] Standard MLP and CNN baseline calibration models
│   └── hytam_net.py                 <-- [Post-Acceptance] Complete Mamba-Transformer dual-stream network
├── utils/                          
│   └── fetch_calculators.py         <-- [Post-Acceptance] Anisotropic fetch rose dictionary generators
├── requirements.txt                 <-- [Available Now] Environment dependencies
└── README.md                        <-- [Available Now] This file
