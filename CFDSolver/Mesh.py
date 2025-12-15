"""
Uniform collocated grid mesh for stream function-vorticity solver.
"""
import numpy as np


class Mesh:
    """
    Uniform collocated grid for the lid-driven cavity problem.
    All variables (psi, omega, u, v) are stored at cell centers.
    """

    def __init__(self, nx, ny, length=1.0, height=1.0):
        """
        Initialize uniform collocated grid.
        
        Args:
            nx: Number of grid points in x-direction
            ny: Number of grid points in y-direction
            length: Domain length in x-direction (default: 1.0)
            height: Domain height in y-direction (default: 1.0)
        """
        self.nx = nx
        self.ny = ny
        self.length = length
        self.height = height
        
        # Grid spacing
        self.dx = length / (nx - 1)
        self.dy = height / (ny - 1)
        
        # Coordinate arrays
        self.x = np.linspace(0, length, nx)
        self.y = np.linspace(0, height, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Field variables
        self.psi = np.zeros((nx, ny))    # Stream function
        self.omega = np.zeros((nx, ny))  # Vorticity
        self.u = np.zeros((nx, ny))      # x-velocity
        self.v = np.zeros((nx, ny))      # y-velocity
        
    def get_centerline_profiles(self):
        """
        Extract velocity profiles along the centerlines.
        
        Returns:
            u_centerline: u-velocity along vertical centerline (x = L/2)
            v_centerline: v-velocity along horizontal centerline (y = H/2)
        """
        i_mid = self.nx // 2
        j_mid = self.ny // 2
        
        u_centerline = self.u[i_mid, :]
        v_centerline = self.v[:, j_mid]
        
        return u_centerline, v_centerline
