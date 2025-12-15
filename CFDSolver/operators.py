"""
Finite difference operators for stream function-vorticity formulation.
"""
import numpy as np


def laplacian(f, dx, dy):
    """
    Compute Laplacian using central differences with periodic-like roll.
    
    Args:
        f: 2D field array
        dx: Grid spacing in x
        dy: Grid spacing in y
        
    Returns:
        Laplacian of f
    """
    lap_x = (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / (dx**2)
    lap_y = (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / (dy**2)
    return lap_x + lap_y


def gradient_x(f, dx):
    """
    Compute gradient in x-direction using central differences.
    
    Args:
        f: 2D field array
        dx: Grid spacing in x
        
    Returns:
        df/dx
    """
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)


def gradient_y(f, dy):
    """
    Compute gradient in y-direction using central differences.
    
    Args:
        f: 2D field array
        dy: Grid spacing in y
        
    Returns:
        df/dy
    """
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dy)


def velocity_from_streamfunction(psi, dx, dy):
    """
    Compute velocity components from stream function.
    u = dpsi/dy, v = -dpsi/dx
    
    Args:
        psi: Stream function
        dx: Grid spacing in x
        dy: Grid spacing in y
        
    Returns:
        u, v: Velocity components
    """
    u = gradient_y(psi, dy)
    v = -gradient_x(psi, dx)
    return u, v


def advection_term(omega, u, v, dx, dy):
    """
    Compute advection term for vorticity transport: u * domega/dx + v * domega/dy
    
    Args:
        omega: Vorticity field
        u: x-velocity
        v: y-velocity
        dx: Grid spacing in x
        dy: Grid spacing in y
        
    Returns:
        Advection term
    """
    domega_dx = gradient_x(omega, dx)
    domega_dy = gradient_y(omega, dy)
    return u * domega_dx + v * domega_dy
