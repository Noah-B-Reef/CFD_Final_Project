#!/usr/bin/env python3
"""
Generate a meshgrid with explicit tags for left, right, top, bottom and
internal points and plot it using matplotlib.

The script creates a simple rectangular grid defined by the coordinates
``x_min, x_max, y_min, y_max`` and a number of points in each direction.
Each grid point is classified as one of the following tags:

* ``left``   – points on the left boundary (x == x_min)
* ``right``  – points on the right boundary (x == x_max)
* ``bottom`` – points on the bottom boundary (y == y_min)
* ``top``    – points on the top boundary (y == y_max)
* ``internal`` – all other points

The grid is visualised in a scatter plot where each tag uses a distinct
colour.  A legend identifies the tags.

The script can be run directly:

    python meshgrid_plot.py

It will open a window with the plot and also save ``meshgrid.png`` in the
current working directory.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_meshgrid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
):
    """Return a 2‑D meshgrid and an array of tags.

    Parameters
    ----------
    x_min, x_max: float
        Horizontal extent of the grid.
    y_min, y_max: float
        Vertical extent of the grid.
    nx, ny: int
        Number of points in the x and y directions (including boundaries).

    Returns
    -------
    X, Y: np.ndarray
        Meshgrid coordinates of shape (ny, nx).
    tags: list[str]
        List of tag strings for each grid point in row‑major order.
    """

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    tags: list[str] = []
    for j in range(ny):
        for i in range(nx):
            if np.isclose(X[i, j], x_min):
                tags.append("left")
            elif np.isclose(X[i, j], x_max):
                tags.append("right")
            elif np.isclose(Y[i, j], y_min):
                tags.append("bottom")
            elif np.isclose(Y[i, j], y_max):
                tags.append("top")
            else:
                tags.append("internal")
    return X, Y, tags


def plot_meshgrid(X: np.ndarray, Y: np.ndarray, tags: list[str]):
    """Plot the meshgrid with a colour for each tag.

    The function uses a colormap that assigns a distinct colour to each
    unique tag.  ``scatter`` is used because it allows individual point
    colours.
    """

    unique_tags = sorted(set(tags))
    color_map = {
        "left": "tab:blue",
        "right": "tab:red",
        "bottom": "tab:green",
        "top": "tab:orange",
        "internal": "gray",
    }

    fig, ax = plt.subplots(figsize=(6, 5))
    for tag in unique_tags:
        mask = np.array(tags) == tag
        ax.scatter(X.ravel()[mask], Y.ravel()[mask], c=color_map[tag], label=tag, s=30)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save and show
    out_path = Path("meshgrid.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    # Example parameters – a 10×8 grid in the unit square
    X, Y, tags = create_meshgrid(0.0, 1.0, 0.0, 1.0, nx=10, ny=8)
    plot_meshgrid(X, Y, tags)

