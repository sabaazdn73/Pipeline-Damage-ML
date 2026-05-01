# Pipeline Damage Detection using ML

## Overview
This project develops ML-based surrogate models for pipeline damage detection, combining simplified FEM simulations with machine learning and uncertainty quantification.

## Research Motivation
Inspired by real-world pipeline integrity assessment (similar to Enbridge pipeline monitoring systems).

## Methodology
1. **FEM Simulation**: Thin-wall pressure vessel model
   - Hoop stress, axial stress, von Mises stress
   - Damage modeled as wall thickness reduction
   
2. **Uncertainty Quantification**: 
   - Material properties sampled from distributions
   - Simulates real-world measurement uncertainty

3. **ML Models Compared**:
   - Linear Regression (R²=0.640)
   - Random Forest (R²=0.675)  
   - **Gaussian Process (R²=0.997) ✓ Best**

## Key Results
Gaussian Process Regressor achieves R²=0.997 with uncertainty bounds — critical for safety-critical pipeline decisions.

## Tech Stack
Python, NumPy, Scikit-learn, Matplotlib, Pandas

## 📚 Theoretical Foundation (FEM Assignments)
To demonstrate my deep understanding of the Finite Element Method, I have included a collection of my graduate-level assignments covering:
* **Shape Function Derivations:** ISO-Q4, ISO-Q8, ISO-Q9, CST, and LST elements using Characteristic and Cross-Beam methods.
* **Numerical Integration:** Extensive work on Gauss-Quadrature rules and their impact on element convergence.
* **Stiffness Matrix Formulation:** Manual and computational derivation of element stiffness matrices.
* **Error Analysis:** Comparative study of CST vs. Q4 elements in bending scenarios (Shear Locking phenomenon).

## Project Structure
```text
pipeline-damage-ml/
├── src/
│   ├── fem_model.py       # FEM simulation
│   ├── dataset.py         # Data generation
│   └── ml_models.py       # ML training & comparison
├── data/
│   └── model_comparison.png
└── README.md
