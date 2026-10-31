# HyTAM-Net: Topography-Aware Deep-Learning Correction of ERA5 Coastal Wind Speeds

This repository accompanies the paper **"Topography-Aware Deep-Learning Correction of ERA5 Coastal Wind Speeds"**.

## Release status

Part of the source code is available now: the preprocessing chain that builds the station dataset from NDBC buoy records and ERA5, and the evaluation metrics used in the paper. The model, the training pipeline and the experiment configurations will be added after the review process.

## Overview

Hourly ERA5 10-m wind speeds carry systematic, site-dependent errors near coasts, where coastline geometry, fetch and shallow water modify the marine surface layer below the resolution of the reanalysis. HyTAM-Net corrects ERA5 at buoy stations by predicting a positive, time-varying scaling factor α, so that the corrected wind speed is U_cor = α · U_ERA5.

- **Dynamic branch.** A Mamba stream and a Transformer stream encode the same 72-h window of ERA5 variables in parallel. Their outputs are concatenated and linearly projected.
- **Static coastal descriptors.** A 36-direction fetch dictionary, distance to coast and bathymetry describe the coastal geometry of each station.
- **Dual-path modulation.** The static descriptors modulate the dynamic features through a latent-feature affine transform. A Softplus output keeps the corrected wind speed positive.
- **Evaluation.** Errors are reported in two dimensions: hourly tracking at each station (T_ metrics, median over stations) and station-mean climatology (C_ metrics, across stations).

## Repository structure

```text
HyTAM-Net/
├── data_preprocessing/
│   ├── QC_NDBC_3.py                  [available]  quality control of NDBC records
│   ├── QC_NDBC_station.py            [available]  station screening
│   ├── clean_out_of_bounds_ndbc.py   [available]  physical-range filtering
│   ├── Get_NDBC_Sta_High.py          [available]  anemometer heights from NDBC metadata
│   ├── MOST_Extrapolation_NDBC.py    [available]  MOST height correction to 10 m
│   ├── Extract_Bathymetry.py         [available]  station bathymetry
│   └── Match_ERA5_NDBC.py            [available]  ERA5-buoy collocation
├── evaluation/
│   └── metrics.py                    [available]  T_ and C_ metrics
├── data/
│   └── station tables                [available]  static descriptors, anemometer heights, data split
├── model/                            [after review]
├── training/                         [after review]
├── requirements.txt
└── README.md
```

## Data sources

- NDBC buoy observations: https://www.ndbc.noaa.gov
- ERA5 hourly single-level data: Copernicus Climate Data Store
- GSHHG shoreline database: https://www.soest.hawaii.edu/pwessel/gshhg/

## Preprocessing order

1. Quality control and physical-range filtering of the NDBC records (`QC_NDBC_3.py`, `QC_NDBC_station.py`, `clean_out_of_bounds_ndbc.py`).
2. Anemometer heights (`Get_NDBC_Sta_High.py`) and MOST height correction to 10 m (`MOST_Extrapolation_NDBC.py`).
3. Station bathymetry (`Extract_Bathymetry.py`).
4. Collocation of ERA5 with the buoy records (`Match_ERA5_NDBC.py`).

Paths are set in the configuration block at the top of each script.

## Data split

Training 2008–2023, validation 2024, test 2025. Normalization statistics are computed from the training years only.

## Citation

The citation will be added after publication.
