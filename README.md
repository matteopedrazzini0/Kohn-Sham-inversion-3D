# Kohn-Sham-inversion-3D
Numerical inversion of Kohn–Sham equations on a 3D Cartesian grid for nuclei without imposed symmetry

## Overview
This project implements a numerical method to reconstruct effective nuclear potentials and single-particle wave functions from a given density by solving an inverse Kohn–Sham problem.
The formulation is fully three-dimensional and does not assume any spatial symmetry.

## Main features
- Three-dimensional Cartesian grid formulation
- Kinetic-energy minimization solved using IPOPT
- Fully numerical implementation developed from scratch
- Parallelized code designed for large-scale simulations
- Execution on HPC clusters using SLURM batch scheduling

## Requirements

Python 3.10+ recommended.

Install dependencies:
```bash
pip install -r requirements.txt
```
Notes:
-`cyipopt` may require an IPOPT installation on your system (or via your HPC environment modules).
-For more details see: https://coin-or.github.io/Ipopt/
