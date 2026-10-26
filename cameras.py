# Pranav Minasandra
# 17 Nov 2025
# pminasandra.github.io

import numpy as np
from collections.abc import Iterable


def run_cameras(cam_positions, cam_radii, positions, *, assume_unique=True):
    """For each individual, return the nearest camera that detects it, or -1."""
    cam_positions = np.asarray(cam_positions, dtype=float)
    cam_radii = np.asarray(cam_radii, dtype=float)
    positions = np.asarray(positions, dtype=float)
    assert cam_positions.ndim == 2 and cam_positions.shape[1] == 2
    assert cam_radii.shape == (len(cam_positions),)
    assert positions.ndim == 2 and positions.shape[1] == 2
    if len(cam_positions) == 0:
        return -np.ones(len(positions), dtype=int)
    diff = positions[:, None, :] - cam_positions[None, :, :]
    dist2 = np.einsum('nmk,nmk->nm', diff, diff)
    dist2[dist2 > cam_radii[None, :] ** 2] = np.inf
    spotted_by = np.argmin(dist2, axis=1)
    spotted_by[~np.isfinite(np.min(dist2, axis=1))] = -1
    return spotted_by


def make_camera_grid(k, radius=0.5, box_size=100., offset_x=0., offset_y=0.):
    """Place k cameras on a regular square grid."""
    if k <= 0:
        raise ValueError('k must be a positive integer')
    n_side = int(round(np.sqrt(k)))
    if n_side * n_side != k:
        raise ValueError(f'k must be a perfect square, got {k}')
    coords_1d = (np.arange(n_side) + 0.5) * (box_size / n_side)
    xs, ys = np.meshgrid(coords_1d, coords_1d, indexing='xy')
    cam_positions = np.column_stack((xs.ravel() + offset_x, ys.ravel() + offset_y))
    if np.isscalar(radius):
        cam_radii = np.full(k, radius, dtype=float)
    else:
        cam_radii = np.asarray(radius, dtype=float)
        if cam_radii.shape != (k,):
            raise ValueError('Expected one radius per camera.')
    return cam_positions, cam_radii


def create_cameras(k, radius=0.5, box_size=100., rng=None):
    """Create a square camera grid with a reproducible random global offset."""
    if rng is None:
        rng = np.random.default_rng()
    spacing = box_size / np.sqrt(k)
    offset_x, offset_y = rng.uniform(-spacing / 2, spacing / 2, 2)
    return make_camera_grid(k, radius, box_size, offset_x, offset_y)
