"""
Boundary conditions for stream function-vorticity formulation.
"""


def apply_streamfunction_bc(psi):
    """
    Apply boundary conditions for stream function.
    psi = 0 on all walls (no-slip condition).
    
    Args:
        psi: Stream function field (modified in place)
    """
    psi[0, :] = 0.0   # Left wall
    psi[-1, :] = 0.0  # Right wall
    psi[:, 0] = 0.0   # Bottom wall
    psi[:, -1] = 0.0  # Top wall (moving lid)


def apply_vorticity_bc(omega, psi, lid_velocity, dx, dy):
    """
    Apply boundary conditions for vorticity using Equation 26 from PDF.
    Derived from no-slip condition and Poisson equation for psi.
    
    At bottom wall (y=0): ω = 2(ψ_{i,1} - ψ_{i,2}) / dy^2
    Using ghost cell elimination with dpsi/dy|_{y=0} = u|_{y=0} = 0
    
    Args:
        omega: Vorticity field (modified in place)
        psi: Stream function field
        lid_velocity: Velocity of moving lid (top wall)
        dx: Grid spacing in x
        dy: Grid spacing in y
    """
    # Bottom wall (y=0, j=0): no-slip, u=0, v=0
    # From PDF Eq. 26: ω_{i,1} = 2(ψ_{i,1} - ψ_{i,2}) / dy^2
    # Note: psi[:, 0] = 0 (boundary condition)
    omega[:, 0] = 2.0 * (psi[:, 0] - psi[:, 1]) / (dy**2)
    
    # Top wall (y=H, j=-1): moving lid with u=lid_velocity, v=0
    # dpsi/dy = u = lid_velocity at top wall
    # Using ghost cell: ψ_{i,ny} = ψ_{i,ny-2} + 2*dy*lid_velocity
    # ω_{i,ny-1} = 2(ψ_{i,ny-1} - ψ_{i,ny-2}) / dy^2 - 2*lid_velocity/dy
    omega[:, -1] = 2.0 * (psi[:, -1] - psi[:, -2]) / (dy**2) - 2.0 * lid_velocity / dy
    
    # Left wall (x=0, i=0): no-slip, u=0, v=0
    # Similar to bottom wall
    omega[0, :] = 2.0 * (psi[0, :] - psi[1, :]) / (dx**2)
    
    # Right wall (x=L, i=-1): no-slip, u=0, v=0
    omega[-1, :] = 2.0 * (psi[-1, :] - psi[-2, :]) / (dx**2)
