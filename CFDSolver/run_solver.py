#!/usr/bin/env python
"""
Lid-Driven Cavity CFD Solver using Streamfunction-Vorticity Formulation
Main script for running simulations and generating visualizations
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from mesh import Mesh
from solver import StreamFunctionSolver


def run_simulation(Re, N, save_solution=False, output_dir='results'):
    """
    Run a single simulation case.
    
    Args:
        Re: Reynolds number
        N: Grid size (NxN)
        save_solution: Whether to save .npy files
        output_dir: Output directory
        
    Returns:
        mesh: Solved mesh object
        solver: Solver with convergence history
    """
    print(f"\n{'='*70}")
    print(f"Running: Re={Re}, Grid={N}x{N}")
    print(f"{'='*70}")
    
    # Create mesh and solver
    mesh = Mesh(N, N)
    solver = StreamFunctionSolver(mesh, Re, lid_velocity=1.0)
    
    # Solve
    max_steps = 100000 if Re >= 3200 else 60000
    converged = solver.solve(max_steps=max_steps, tol=1e-6, print_interval=5000)
    
    if converged:
        print(f"✓ Converged at step {len(solver.convergence_history)}")
    else:
        print(f"⚠ Did not fully converge (residual: {solver.convergence_history[-1]:.3e})")
    
    # Save solution if requested
    if save_solution:
        save_dir = Path(output_dir) / 'solutions'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(save_dir / f'psi_Re{Re}_N{N}.npy', mesh.psi)
        np.save(save_dir / f'omega_Re{Re}_N{N}.npy', mesh.omega)
        np.save(save_dir / f'u_Re{Re}_N{N}.npy', mesh.u)
        np.save(save_dir / f'v_Re{Re}_N{N}.npy', mesh.v)
        print(f"✓ Solution saved to {save_dir}/")
    
    return mesh, solver


def plot_streamlines(results_dict, output_dir='results'):
    """
    Generate streamline plots for all cases.
    
    Args:
        results_dict: Dictionary {(Re, N): (mesh, solver)}
        output_dir: Output directory for plots
    """
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Generating Streamline Plots")
    print(f"{'='*70}")
    
    for (Re, N), (mesh, solver) in results_dict.items():
        print(f"  Creating plot for Re={Re}, N={N}...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get streamfunction range
        psi_min = np.min(mesh.psi)
        psi_max = np.max(mesh.psi)
        
        # Create fine contour levels (logarithmic for positive values)
        neg_levels = np.linspace(psi_min, -1e-7, 40)
        if psi_max > 1e-7:
            pos_levels = np.logspace(np.log10(1e-7), np.log10(psi_max), 40)
            levels = np.sort(np.concatenate([neg_levels, [0], pos_levels]))
        else:
            levels = np.sort(np.concatenate([neg_levels, [0]]))
        
        # Plot streamlines
        contour = ax.contour(mesh.X, mesh.Y, mesh.psi, 
                            levels=levels, 
                            colors='blue', 
                            linewidths=0.8,
                            alpha=0.9)
        
        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=12, fontweight='bold')
        ax.set_ylabel('y', fontsize=12, fontweight='bold')
        ax.set_title(f'Streamlines: Re = {Re}, Grid = {N}×{N}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
        
        # Add info box
        textstr = f'Steps: {len(solver.convergence_history)}\n'
        textstr += f'Residual: {solver.convergence_history[-1]:.2e}\n'
        textstr += f'ψ_min: {psi_min:.5f}\n'
        textstr += f'ψ_max: {psi_max:.2e}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
        ax.text(0.02, 0.35, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')
        
        plt.tight_layout()
        filename = plots_dir / f'streamlines_Re{Re}_N{N}.svg'
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {filename}")


def plot_velocity_profiles(results_dict, N, output_dir='results'):
    """
    Generate velocity profile plots for comparison across Reynolds numbers.
    
    Args:
        results_dict: Dictionary {(Re, N): (mesh, solver)}
        N: Grid size to plot
        output_dir: Output directory
    """
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating velocity profile comparisons...")
    
    # Filter for specific grid size
    Re_list = sorted([Re for (Re, n) in results_dict.keys() if n == N])
    
    if len(Re_list) < 2:
        print(f"  ⚠ Need at least 2 Reynolds numbers for comparison plot")
        return
    
    # u-velocity along vertical centerline
    fig, ax = plt.subplots(figsize=(8, 10))
    for Re in Re_list:
        mesh, _ = results_dict[(Re, N)]
        u_centerline, _ = mesh.get_centerline_profiles()
        ax.plot(u_centerline, mesh.y, label=f'Re={Re}', linewidth=2, marker='o', 
                markersize=3, markevery=N//10)
    
    ax.set_xlabel('u', fontsize=12, fontweight='bold')
    ax.set_ylabel('y', fontsize=12, fontweight='bold')
    ax.set_title(f'u-velocity at vertical centerline (x=0.5)\nGrid: {N}×{N}', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = plots_dir / f'u_profile_N{N}.svg'
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")
    
    # v-velocity along horizontal centerline
    fig, ax = plt.subplots(figsize=(10, 6))
    for Re in Re_list:
        mesh, _ = results_dict[(Re, N)]
        _, v_centerline = mesh.get_centerline_profiles()
        ax.plot(mesh.x, v_centerline, label=f'Re={Re}', linewidth=2, marker='o',
                markersize=3, markevery=N//10)
    
    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('v', fontsize=12, fontweight='bold')
    ax.set_title(f'v-velocity at horizontal centerline (y=0.5)\nGrid: {N}×{N}', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = plots_dir / f'v_profile_N{N}.svg'
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_convergence(results_dict, output_dir='results'):
    """
    Generate convergence history plots.
    
    Args:
        results_dict: Dictionary {(Re, N): (mesh, solver)}
        output_dir: Output directory
    """
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating convergence plots...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for (Re, N), (mesh, solver) in sorted(results_dict.items()):
        history = solver.convergence_history
        steps = np.arange(len(history))
        ax.semilogy(steps, history, linewidth=2, label=f'Re={Re}, N={N}')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$|\Delta\omega|_\infty$', fontsize=12, fontweight='bold')
    ax.set_title('Convergence History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    plt.tight_layout()
    
    filename = plots_dir / 'convergence.svg'
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def generate_summary_report(results_dict, output_dir='results'):
    """
    Generate summary report of all simulations.
    
    Args:
        results_dict: Dictionary {(Re, N): (mesh, solver)}
        output_dir: Output directory
    """
    report_file = Path(output_dir) / 'simulation_summary.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CFD Simulation Summary: Lid-Driven Cavity Flow\n")
        f.write("Streamfunction-Vorticity Formulation\n")
        f.write("="*70 + "\n\n")
        
        for (Re, N), (mesh, solver) in sorted(results_dict.items()):
            f.write(f"\nRe = {Re}, Grid = {N}×{N}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Iterations: {len(solver.convergence_history)}\n")
            f.write(f"  Final residual: {solver.convergence_history[-1]:.6e}\n")
            f.write(f"  ψ_min: {np.min(mesh.psi):.6f}\n")
            f.write(f"  ψ_max: {np.max(mesh.psi):.6e}\n")
            f.write(f"  Max |ω|: {np.max(np.abs(mesh.omega)):.3f}\n")
            f.write(f"  Max |u|: {np.max(np.abs(mesh.u)):.6f}\n")
            f.write(f"  Max |v|: {np.max(np.abs(mesh.v)):.6f}\n")
            
            # Find primary vortex
            psi_interior = mesh.psi[1:-1, 1:-1]
            min_idx = np.unravel_index(np.argmin(psi_interior), psi_interior.shape)
            min_idx = (min_idx[0] + 1, min_idx[1] + 1)
            vortex_x = mesh.x[min_idx[0]]
            vortex_y = mesh.y[min_idx[1]]
            f.write(f"  Primary vortex center: ({vortex_x:.4f}, {vortex_y:.4f})\n")
    
    print(f"\n✓ Summary report saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='CFD Solver for Lid-Driven Cavity Flow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_solver.py --Re 100 400 1000 --N 129
  python run_solver.py --Re 100 --N 65 --save y
  python run_solver.py --Re 100 400 1000 3200 --N 129 --save y --output results
        """
    )
    
    parser.add_argument('--Re', type=int, nargs='+', required=True,
                       help='Reynolds numbers to simulate (space-separated list)')
    parser.add_argument('--N', type=int, required=True,
                       help='Grid size (NxN mesh)')
    parser.add_argument('--save', type=str, choices=['y', 'n'], default='n',
                       help='Save solution arrays (.npy files): y or n')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    # Convert save option
    save_solution = (args.save == 'y')
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Run all simulations
    print("\n" + "="*70)
    print("STARTING SIMULATIONS")
    print("="*70)
    print(f"Reynolds numbers: {args.Re}")
    print(f"Grid size: {args.N}×{args.N}")
    print(f"Save solutions: {save_solution}")
    print(f"Output directory: {args.output}")
    
    results_dict = {}
    for Re in args.Re:
        mesh, solver = run_simulation(Re, args.N, save_solution, args.output)
        results_dict[(Re, args.N)] = (mesh, solver)
    
    # Generate all plots
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    plot_streamlines(results_dict, args.output)
    plot_velocity_profiles(results_dict, args.N, args.output)
    plot_convergence(results_dict, args.output)
    
    # Generate summary
    generate_summary_report(results_dict, args.output)
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE")
    print(f"{'='*70}")
    print(f"All results saved to: {args.output}/")
    print(f"  - Plots: {args.output}/plots/")
    if save_solution:
        print(f"  - Solutions: {args.output}/solutions/")
    print(f"  - Summary: {args.output}/simulation_summary.txt")


if __name__ == '__main__':
    main()
