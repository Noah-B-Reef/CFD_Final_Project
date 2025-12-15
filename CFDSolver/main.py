"""
Main driver for stream function-vorticity lid-driven cavity solver.
Runs simulations for varying Reynolds numbers and grid sizes, and generates plots.
"""
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mesh import Mesh
from solver import StreamFunctionSolver


def save_solution(mesh, solver, Re, nx, ny, output_dir='solutions'):
    """
    Save steady-state solution to NPZ file for later plotting.

    Args:
        mesh: Mesh object with solution
        solver: Solver object with convergence history
        Re: Reynolds number
        nx: Grid points in x
        ny: Grid points in y
        output_dir: Output directory for solution files
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'solution_Re{Re}_grid{nx}x{ny}.npz')

    np.savez(filename,
             psi=mesh.psi,
             omega=mesh.omega,
             u=mesh.u,
             v=mesh.v,
             x=mesh.x,
             y=mesh.y,
             X=mesh.X,
             Y=mesh.Y,
             Re=Re,
             nx=nx,
             ny=ny,
             convergence_history=np.array(solver.convergence_history))

    print(f"Saved solution to: {filename}")


def run_case(Re, nx, ny, max_steps=8000, tol=1e-6, lid_velocity=1.0, print_interval=200, save=False, save_dir='solutions'):
    """
    Run a single simulation case.

    Args:
        Re: Reynolds number
        nx: Grid points in x
        ny: Grid points in y
        max_steps: Maximum time steps
        tol: Convergence tolerance
        lid_velocity: Lid velocity
        print_interval: Print interval
        save: If True, save solution to file
        save_dir: Directory for saved solutions

    Returns:
        mesh: Mesh object with solution
        solver: Solver object with convergence history
    """
    print(f"\n{'='*60}")
    print(f"Running case: Re={Re}, grid={nx}x{ny}")
    print(f"{'='*60}")

    mesh = Mesh(nx, ny)
    solver = StreamFunctionSolver(mesh, Re, lid_velocity)
    solver.solve(max_steps=max_steps, tol=tol, print_interval=print_interval)

    if save:
        save_solution(mesh, solver, Re, nx, ny, save_dir)

    return mesh, solver


def plot_convergence_for_grids(results, reynolds_number, grid_sizes, output_dir='plots'):
    """
    Generate convergence plot for fixed Re with varying grid sizes.

    Args:
        results: Dictionary with keys (Re, nx, ny) and values (mesh, solver)
        reynolds_number: Fixed Reynolds number
        grid_sizes: List of (nx, ny) tuples
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for nx, ny in grid_sizes:
        if (reynolds_number, nx, ny) in results:
            mesh, solver = results[(reynolds_number, nx, ny)]
            history = solver.convergence_history
            steps = np.arange(len(history))
            ax.semilogy(steps, history, linewidth=2, label=f'{nx} × {ny}')

    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$|\Delta\omega|_\infty$', fontsize=12, fontweight='bold')
    ax.set_title(f'Convergence History: Re={reynolds_number}, Varying Grid Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    filepath = os.path.join(output_dir, f'convergence_Re{reynolds_number}_varying_grids.svg')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plot: {filepath}")


def plot_convergence_for_reynolds(results, reynolds_numbers, grid_size, output_dir='plots'):
    """
    Generate convergence plot for varying Re at fixed grid size.

    Args:
        results: Dictionary with keys (Re, nx, ny) and values (mesh, solver)
        reynolds_numbers: List of Reynolds numbers
        grid_size: Tuple (nx, ny) for fixed grid
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    nx, ny = grid_size
    fig, ax = plt.subplots(figsize=(8, 6))

    for Re in reynolds_numbers:
        if (Re, nx, ny) in results:
            mesh, solver = results[(Re, nx, ny)]
            history = solver.convergence_history
            steps = np.arange(len(history))
            ax.semilogy(steps, history, linewidth=2, label=f'Re = {Re}')

    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$|\Delta\omega|_\infty$', fontsize=12, fontweight='bold')
    ax.set_title(f'Convergence History: Grid {nx}×{ny}, Varying Re', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    filepath = os.path.join(output_dir, f'convergence_grid{nx}x{ny}_varying_Re.svg')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plot: {filepath}")


def plot_velocity_profiles_re_variation(results, grid_size, reynolds_numbers, output_dir='plots'):
    """
    Plot velocity profiles for varying Reynolds numbers at fixed grid size.
    
    Args:
        results: Dictionary with keys (Re, nx, ny) and values (mesh, solver)
        grid_size: Tuple (nx, ny) for the fixed grid
        reynolds_numbers: List of Reynolds numbers to plot
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    nx, ny = grid_size
    
    # Plot u-velocity along vertical centerline
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for Re in reynolds_numbers:
        if (Re, nx, ny) in results:
            mesh, _ = results[(Re, nx, ny)]
            u_centerline, _ = mesh.get_centerline_profiles()
            ax.plot(u_centerline, mesh.y, label=f'Re={Re}', linewidth=2)
    
    ax.set_xlabel(r'$u$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_title(f'u-velocity along vertical centerline (x=L/2)\nGrid: {nx}×{ny}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'u_profile_grid{nx}x{ny}.svg')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved u-velocity profile: {filepath}")
    
    # Plot v-velocity along horizontal centerline
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for Re in reynolds_numbers:
        if (Re, nx, ny) in results:
            mesh, _ = results[(Re, nx, ny)]
            _, v_centerline = mesh.get_centerline_profiles()
            ax.plot(mesh.x, v_centerline, label=f'Re={Re}', linewidth=2)
    
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$v$', fontsize=12)
    ax.set_title(f'v-velocity along horizontal centerline (y=H/2)\nGrid: {nx}×{ny}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'v_profile_grid{nx}x{ny}.svg')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved v-velocity profile: {filepath}")


def plot_streamlines(results, reynolds_numbers, grid_size, output_dir='plots'):
    """
    Plot streamlines for varying Reynolds numbers at fixed grid size in a 2x2 subplot.
    Uses fine multi-scale logarithmic spacing to resolve all vortices.

    Args:
        results: Dictionary with keys (Re, nx, ny) and values (mesh, solver)
        reynolds_numbers: List of Reynolds numbers to plot
        grid_size: Tuple (nx, ny) for the fixed grid
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    nx, ny = grid_size

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, Re in enumerate(reynolds_numbers):
        if (Re, nx, ny) in results:
            mesh, solver = results[(Re, nx, ny)]
            ax = axes[idx]

            # Generate fine multi-scale levels to resolve all vortices
            psi_min = np.min(mesh.psi)
            psi_max = np.max(mesh.psi)

            # 1. Primary vortex (large negative values): linear spacing
            if psi_min < -1e-4:
                primary_levels = np.linspace(psi_min, -1e-4, 25)
            else:
                primary_levels = np.array([psi_min])

            # 2. Transition region (small negative): logarithmic spacing
            if psi_min < -1e-7:
                transition_neg = -np.logspace(np.log10(max(-psi_min, 1e-4)), np.log10(1e-7), 30)[::-1]
            else:
                transition_neg = np.array([])

            # 3. Near-zero region (crucial for weak vortices): fine logarithmic spacing
            if psi_max > 1e-7:
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
            levels = np.unique(levels)

            # Create contour plot with fine resolution
            contour = ax.contour(mesh.X, mesh.Y, mesh.psi, levels=levels,
                                linewidths=0.7, colors='blue', alpha=0.8)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xlabel('x', fontsize=10, fontweight='bold')
            ax.set_ylabel('y', fontsize=10, fontweight='bold')
            ax.set_title(f'Re = {Re}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'streamlines_grid{nx}x{ny}.svg')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved streamline plot: {filepath}")


def plot_grid_convergence(results, reynolds_number, grid_sizes, output_dir='plots'):
    """
    Plot velocity profiles for varying grid sizes at fixed Reynolds number.
    
    Args:
        results: Dictionary with keys (Re, nx, ny) and values (mesh, solver)
        reynolds_number: Fixed Reynolds number
        grid_sizes: List of (nx, ny) tuples
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot u-velocity along vertical centerline
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for nx, ny in grid_sizes:
        if (reynolds_number, nx, ny) in results:
            mesh, _ = results[(reynolds_number, nx, ny)]
            u_centerline, _ = mesh.get_centerline_profiles()
            ax.plot(u_centerline, mesh.y, label=f'{nx}×{ny}', linewidth=2)
    
    ax.set_xlabel(r'$u$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_title(f'u-velocity: Grid Convergence at Re={reynolds_number}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'u_grid_convergence_Re{reynolds_number}.svg')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved grid convergence plot (u): {filepath}")
    
    # Plot v-velocity along horizontal centerline
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for nx, ny in grid_sizes:
        if (reynolds_number, nx, ny) in results:
            mesh, _ = results[(reynolds_number, nx, ny)]
            _, v_centerline = mesh.get_centerline_profiles()
            ax.plot(mesh.x, v_centerline, label=f'{nx}×{ny}', linewidth=2)
    
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$v$', fontsize=12)
    ax.set_title(f'v-velocity: Grid Convergence at Re={reynolds_number}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'v_grid_convergence_Re{reynolds_number}.svg')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved grid convergence plot (v): {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Stream function-vorticity solver for lid-driven cavity'
    )
    parser.add_argument('--Re', type=int, nargs='+', 
                       default=[100, 400, 1000, 3200],
                       help='Reynolds numbers to simulate')
    parser.add_argument('--grids', type=int, nargs='+',
                       default=[32, 65, 129],
                       help='Grid sizes (assumes square grids)')
    parser.add_argument('--max-steps', type=int, default=8000,
                       help='Maximum time steps')
    parser.add_argument('--tol', type=float, default=1e-6,
                       help='Convergence tolerance')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--save', action='store_true',
                       help='Save steady-state solutions to NPZ files')
    parser.add_argument('--save-dir', type=str, default='solutions',
                       help='Output directory for saved solutions')

    args = parser.parse_args()

    # Dictionary to store all results: key = (Re, nx, ny), value = (mesh, solver)
    results = {}

    # Run all simulations
    print("\n" + "="*60)
    print("RUNNING SIMULATIONS")
    print("="*60)

    for Re in args.Re:
        for grid_size in args.grids:
            nx = ny = grid_size
            mesh, solver = run_case(Re, nx, ny,
                                   max_steps=args.max_steps,
                                   tol=args.tol,
                                   save=args.save,
                                   save_dir=args.save_dir)
            results[(Re, nx, ny)] = (mesh, solver)
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    # Determine finest grid
    finest_grid = max(args.grids)

    # Convergence plots for varying grids at Re=100
    if 100 in args.Re and len(args.grids) > 1:
        print("\nGenerating convergence plot for Re=100 with varying grids...")
        grid_sizes = [(g, g) for g in args.grids]
        plot_convergence_for_grids(results, 100, grid_sizes, args.output_dir)

    # Convergence plots for varying grids at Re=1000
    if 1000 in args.Re and len(args.grids) > 1:
        print("\nGenerating convergence plot for Re=1000 with varying grids...")
        grid_sizes = [(g, g) for g in args.grids]
        plot_convergence_for_grids(results, 1000, grid_sizes, args.output_dir)

    # Convergence plot for varying Re at fixed grid
    if len(args.Re) > 1:
        print(f"\nGenerating convergence plot for varying Re at {finest_grid}x{finest_grid} grid...")
        plot_convergence_for_reynolds(results, args.Re, (finest_grid, finest_grid), args.output_dir)

    # Grid convergence plots for Re=100
    if 100 in args.Re and len(args.grids) > 1:
        print("\nGenerating velocity profile grid convergence plots at Re=100...")
        grid_sizes = [(g, g) for g in args.grids]
        plot_grid_convergence(results, 100, grid_sizes, args.output_dir)

    # Velocity profiles for varying Re at finest grid
    if len(args.Re) > 1:
        print(f"\nGenerating velocity profiles for varying Re at {finest_grid}x{finest_grid} grid...")
        plot_velocity_profiles_re_variation(
            results, 
            (finest_grid, finest_grid), 
            args.Re, 
            args.output_dir
        )
    
    # Streamline plots for varying Re at finest grid
    if len(args.Re) >= 4:
        print(f"\nGenerating streamline plots for varying Re at {finest_grid}x{finest_grid} grid...")
        plot_streamlines(
            results,
            args.Re[:4],  # Take first 4 Re values for 2x2 grid
            (finest_grid, finest_grid),
            args.output_dir
        )
    
    print("\n" + "="*60)
    print("COMPLETE")
    print(f"All plots saved to: {args.output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()
