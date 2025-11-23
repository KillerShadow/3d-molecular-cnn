# 3D-CNN for Molecular Property Prediction

This repository contains a complete workflow for generating quantum chemical data, processing volumetric electronic densities, and training a 3D Convolutional Neural Network (CNN).

## Overview

The workflow consists of three stages:
1.  **Simulation:** Running QM/MD simulations and calculating electronic properties (HOMO, LUMO, Density) using ORCA.
2.  **Processing:** Extracting volumetric cube files and scalar descriptors into a compressed HDF5 dataset.
3.  **Training:** Training a custom 3D CNN on the volumetric data using TensorFlow.

## Project Structure

```text
├── scripts/
│   ├── run_simulation.py   # Automates ORCA MD and Single-Point calculations
│   └── process_data.py     # Parses output files into HDF5 format
├── src/
│   ├── generator.py        # TensorFlow Data Generator for HDF5
│   └── model.py            # 3D CNN Model definition and training loop
├── README.md
└── requirements.txt
