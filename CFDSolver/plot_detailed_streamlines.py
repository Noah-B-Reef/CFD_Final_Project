#!/usr/bin/env python
"""
Generate detailed streamline plot with better visualization of secondary/tertiary vortices.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mesh import Mesh
from solver import StreamFunctionSolver

def plot_detailed_streamlines(Re, nx, ny, output_file='streamlines_detailed.svg'):
    """
    Run simulation and create detailed streamline plot showing all vortices.
    
    Args:
        Re: Reynolds number
        nx: Grid points in x
        ny: Grid points in y
        output_file: Output filename
    """
    print(f"\nRunning case: Re={Re}, grid={nx}x{ny}")
    print("="*60)
    
    # Create mesh and solver
    mesh = Mesh(nx, ny)
    solver = StreamFunctionSolver(mesh, Re, lid_velocity=1.0)
    
    # Solve
    solver.solve(max_steps=60000, tol=1e-6, print_interval=2000)
    
    # Create figure with better size
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Find streamfunction range
    psi_min = np.min(mesh.psi)
    psi_max = np.max(mesh.psi)
    print(f"\nStreamfunction range: [{psi_min:.6e}, {psi_max:.6e}]")
    
    # Create multi-scale levels to capture ALL vortices (primary, secondary, tertiary)
    # The key is to have many levels near zero where weak vortices exist

    # Strategy: Use multiple logarithmic/linear regions for maximum resolution

    # 1. Primary vortex (large negative values): linear spacing
    if psi_min < -1e-4:
        primary_levels = np.linspace(psi_min, -1e-4, 25)
    else:
        primary_levels = np.array([psi_min])

    # 2. Transition region (small negative): logarithmic spacing for better resolution
    if psi_min < -1e-7:
        transition_neg = -np.logspace(np.log10(max(-psi_min, 1e-4)), np.log10(1e-7), 30)[::-1]
    else:
        transition_neg = np.array([])

    # 3. Near-zero region (crucial for weak vortices): very fine logarithmic spacing
    if psi_max > 1e-7:
        # Fine positive levels using multiple logarithmic segments
        weak_pos = np.logspace(np.log10(1e-7), np.log10(min(1e-5, psi_max)), 20)
        if psi_max > 1e-5:
            medium_pos = np.logspace(np.log10(1e-5), np.log10(min(1e-3, psi_max)), 20)
        else:
            medium_pos = np.array([])
        if psi_max > 1e-3:
            strong_pos = np.logspace(np.log10(1e-3), np.log10(psi_max), 15)
        else:
            strong_pos = np.array([])
        pos_levels = np.concatenate([weak_pos, medium_pos, strong_pos])
    else:
        pos_levels = np.array([])

    # Combine all levels
    level_components = [primary_levels, transition_neg]
    if len(pos_levels) > 0:
        level_components.append(pos_levels)

    levels = np.sort(np.concatenate(level_components))
    levels = np.unique(levels)  # Remove duplicates

    print(f"Using {len(levels)} contour levels across {np.log10(-psi_min):.1f} to {np.log10(max(psi_max, 1e-7)):.1f} log range")
    
    # Plot all streamlines with uniform blue color
    contour = ax.contour(mesh.X, mesh.Y, mesh.psi,
                        levels=levels,
                        colors='blue',
                        linewidths=0.8,
                        linestyles='solid',
                        alpha=0.85)
    
    # Set limits and aspect
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    # Labels and title
    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('y', fontsize=14, fontweight='bold')
    ax.set_title(f'Streamlines: Re = {Re}, Grid = {nx} × {ny}',
                 fontsize=16, fontweight='bold')

    # Add subtle grid
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
    
    # Add convergence info box
    textstr = f'Converged at step {len(solver.convergence_history)}\n'
    textstr += f'Final residual: {solver.convergence_history[-1]:.3e}\n'
    textstr += f'ψ_min = {psi_min:.6f}\n'
    textstr += f'ψ_max = {psi_max:.6e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.35, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDetailed streamline plot saved to: {output_file}")
    
    # Analyze vortices
    print("\n" + "="*60)
    print("VORTEX ANALYSIS")
    print("="*60)
    
    # Find local extrema
    psi_interior = mesh.psi[1:-1, 1:-1]
    
    # Primary vortex (minimum)
    min_idx = np.unravel_index(np.argmin(psi_interior), psi_interior.shape)
    min_idx = (min_idx[0] + 1, min_idx[1] + 1)  # Adjust for interior indexing
    vortex_x = mesh.x[min_idx[0]]
    vortex_y = mesh.y[min_idx[1]]
    print(f"\nPrimary vortex (ψ minimum):")
    print(f"  Location: ({vortex_x:.4f}, {vortex_y:.4f})")
    print(f"  ψ value: {mesh.psi[min_idx]:.6f}")
    
    # Look for secondary vortices (positive local maxima in corners)
    print(f"\nSearching for secondary vortices (ψ > 0)...")
    
    # Bottom-left corner
    corner_size = nx // 8
    bl_corner = mesh.psi[1:corner_size, 1:corner_size]
    if np.max(bl_corner) > 1e-7:
        bl_max_idx = np.unravel_index(np.argmax(bl_corner), bl_corner.shape)
        bl_max_idx = (bl_max_idx[0] + 1, bl_max_idx[1] + 1)
        print(f"  Bottom-left vortex:")
        print(f"    Location: ({mesh.x[bl_max_idx[0]]:.4f}, {mesh.y[bl_max_idx[1]]:.4f})")
        print(f"    ψ value: {mesh.psi[bl_max_idx]:.6e}")
    
    # Bottom-right corner
    br_corner = mesh.psi[-corner_size:-1, 1:corner_size]
    if np.max(br_corner) > 1e-7:
        br_max_idx = np.unravel_index(np.argmax(br_corner), br_corner.shape)
        br_max_idx = (br_max_idx[0] + nx - corner_size, br_max_idx[1] + 1)
        print(f"  Bottom-right vortex:")
        print(f"    Location: ({mesh.x[br_max_idx[0]]:.4f}, {mesh.y[br_max_idx[1]]:.4f})")
        print(f"    ψ value: {mesh.psi[br_max_idx]:.6e}")
    
    # Top-left corner (for higher Re)
    tl_corner = mesh.psi[1:corner_size, -corner_size:-1]
    if np.max(tl_corner) > 1e-7:
        tl_max_idx = np.unravel_index(np.argmax(tl_corner), tl_corner.shape)
        tl_max_idx = (tl_max_idx[0] + 1, tl_max_idx[1] + ny - corner_size)
        print(f"  Top-left vortex:")
        print(f"    Location: ({mesh.x[tl_max_idx[0]]:.4f}, {mesh.y[tl_max_idx[1]]:.4f})")
        print(f"    ψ value: {mesh.psi[tl_max_idx]:.6e}")
    
    # Flow statistics
    print(f"\n" + "="*60)
    print("FLOW STATISTICS")
    print("="*60)
    print(f"  Max |ω|: {np.max(np.abs(mesh.omega)):.3f}")
    print(f"  Max |u|: {np.max(np.abs(mesh.u)):.6f}")
    print(f"  Max |v|: {np.max(np.abs(mesh.v)):.6f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate detailed streamline plot')
    parser.add_argument('--Re', type=int, default=1000, help='Reynolds number')
    parser.add_argument('--nx', type=int, default=129, help='Grid points in x')
    parser.add_argument('--ny', type=int, default=129, help='Grid points in y')
    parser.add_argument('--output', type=str, default='plots/streamlines_detailed.svg',
                       help='Output filename')
    
    args = parser.parse_args()
    
    plot_detailed_streamlines(args.Re, args.nx, args.ny, args.output)
