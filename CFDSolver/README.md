# CFD Solver: Lid-Driven Cavity Flow

## Overview

This solver computes steady-state solutions for 2D incompressible lid-driven cavity flow using the **streamfunction-vorticity formulation**. The implementation eliminates pressure from the Navier-Stokes equations and uses explicit time-stepping with vectorized NumPy operations for efficiency.

## Method

### Streamfunction-Vorticity Formulation

**Governing Equations:**
```
∂ω/∂t + u·∂ω/∂x + v·∂ω/∂y = (1/Re)·∇²ω    (Vorticity transport)
∇²ψ = -ω                                      (Streamfunction Poisson equation)
u = ∂ψ/∂y,  v = -∂ψ/∂x                       (Velocity recovery)
```

### Numerical Scheme

- **Time Integration:** FTCS (Forward-Time Central-Space) explicit
- **Spatial Discretization:** 2nd-order central differences
- **Poisson Solver:** Jacobi iteration (100 sweeps per time step)
- **Stability:** Adaptive time step based on CFL conditions
- **Convergence Criterion:** ||Δω||∞ < 10⁻⁶

## Installation

### Requirements

- Python 3.7+
- NumPy ≥ 1.21.0
- Matplotlib ≥ 3.4.0

### Setup

```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface

```bash
python run_solver.py --Re <Reynolds_numbers> --N <grid_size> [--save y/n] [--output <dir>]
```

**Required Arguments:**
- `--Re`: Reynolds numbers (space-separated list)
- `--N`: Grid size (creates N×N mesh)

**Optional Arguments:**
- `--save`: Save solution arrays as .npy files (`y` or `n`, default: `n`)
- `--output`: Output directory (default: `results`)

### Examples

**Single Reynolds number:**
```bash
python run_solver.py --Re 100 --N 129
```

**Multiple Reynolds numbers:**
```bash
python run_solver.py --Re 100 400 1000 --N 129
```

**With solution saving:**
```bash
python run_solver.py --Re 100 400 1000 3200 --N 129 --save y
```

**Custom output directory:**
```bash
python run_solver.py --Re 100 --N 65 --output my_results
```

## Outputs

After running, all results are saved to the output directory:

### Directory Structure
```
results/
├── plots/
│   ├── streamlines_Re{Re}_N{N}.svg      # Individual streamline plots
│   ├── u_profile_N{N}.svg                # u-velocity comparison
│   ├── v_profile_N{N}.svg                # v-velocity comparison
│   └── convergence.svg                   # Convergence history
├── solutions/                            # (if --save y)
│   ├── psi_Re{Re}_N{N}.npy
│   ├── omega_Re{Re}_N{N}.npy
│   ├── u_Re{Re}_N{N}.npy
│   └── v_Re{Re}_N{N}.npy
└── simulation_summary.txt                # Text summary
```

### Generated Plots

1. **Streamline Plots** - Individual plots for each (Re, N) combination showing:
   - Fine-resolution streamlines (81 contour levels)
   - Primary and secondary/tertiary vortices
   - Convergence information

2. **Velocity Profiles** - Comparison plots across Reynolds numbers:
   - u-velocity along vertical centerline (x=0.5)
   - v-velocity along horizontal centerline (y=0.5)

3. **Convergence History** - Semi-log plot of residual vs iteration

## Code Structure

### Core Modules

- **`mesh.py`** - Uniform collocated grid mesh class
- **`solver.py`** - Streamfunction-vorticity solver with vectorized operations
- **`boundary_conditions.py`** - Boundary conditions for ψ and ω
- **`operators.py`** - Finite difference operators
- **`run_solver.py`** - Main driver script (use this!)

### Legacy Scripts

- `main.py` - Old interface (deprecated, use `run_solver.py`)
- `plot_detailed_streamlines.py` - Standalone plotting utility

## Physical Features

### Vortex Structures

The lid-driven cavity flow exhibits multiple vortex structures:

1. **Primary Vortex** (ψ < 0)
   - Large clockwise recirculation
   - Driven by moving lid
   - Dominates flow field

2. **Secondary Vortices** (ψ > 0, ~10⁻³ to 10⁻⁴)
   - Counter-clockwise rotation in bottom corners
   - Form due to flow separation at sharp corners
   - Bottom-right typically stronger than bottom-left

3. **Tertiary Vortices** (ψ > 0, ~10⁻⁴)
   - Appear at high Re (≥ 3200)
   - Located in top-left corner
   - Result from primary vortex/wall interaction

### Reynolds Number Effects

| Re   | Primary Vortex | Secondary Vortices | Tertiary Vortex |
|------|----------------|-------------------|-----------------|
| 100  | ✓              | Weak              | ✗               |
| 400  | ✓              | Moderate          | ✗               |
| 1000 | ✓              | Strong            | ✗               |
| 3200 | ✓              | Strong            | ✓               |

## Validation

Results validated against benchmark literature:
- **Ghia et al. (1982)** - Multigrid streamfunction-vorticity
- **Botella & Peyret (1998)** - Spectral methods

Primary vortex center locations match published data to within 1%.

## Performance

Typical run times on 129×129 grid (M1 Mac):
- Re = 100: ~5 minutes
- Re = 400: ~10 minutes
- Re = 1000: ~15 minutes
- Re = 3200: ~40 minutes

## Troubleshooting

**Slow convergence at high Re:**
- Use finer grid (increase N)
- Be patient - Re ≥ 3200 may need 50,000+ iterations

**Numerical instability:**
- Check that grid is fine enough for Reynolds number
- Solver automatically adjusts time step for stability

**Memory issues:**
- Reduce grid size (N)
- Don't save solutions for many cases simultaneously

## References

1. Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *J. Comput. Phys.*, 48(3), 387-411.

2. Kulkarni, T. (2016). Numerical simulations of lid driven cavity with ADI and streamfunction vorticity formulations. Technical Report.

## License

Academic use - CSE397 Final Project, Fall 2025
