from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

import cameras
import config


def run_cameras_on_trajectory(
    traj_path: str | Path,
    cams: Tuple | None = None,
    m: int = 20,
    radius: float = 0.5,
    box_size: float = config.BOX_SIZE,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Given a saved predator-prey trajectory (.npy), return camera sightings.

    Columns: ['timestamp', 'camera_id', 'n_prey', 'n_predators'].
    One row per *triggered camera* per timestep. Predators occupy the first
    config.N_PREDATORS trajectory columns; prey occupy the remaining columns.
    If cameras overlap, an animal is counted by the nearest camera only.
    """
    traj_pos = np.load(Path(traj_path))
    if traj_pos.ndim != 3 or traj_pos.shape[2] != 2:
        raise ValueError('Expected traj_pos with shape (T+1, N, 2).')
    T1, N, _ = traj_pos.shape
    if N != config.N_PREDATORS + config.N_PREY:
        raise ValueError('Trajectory abundance does not match config.py.')
    rng = np.random.default_rng(seed)
    if cams is None:
        cam_positions, cam_radii = cameras.create_cameras(
            k=m, radius=radius, box_size=box_size, rng=rng)
    else:
        cam_positions, cam_radii = cams
    cam_positions = np.asarray(cam_positions, dtype=float)
    cam_radii = np.asarray(cam_radii, dtype=float)

    records = []
    for t in range(T1):
        spotted_by = cameras.run_cameras(cam_positions, cam_radii, traj_pos[t])
        for cam_id in np.unique(spotted_by[spotted_by >= 0]):
            inds = np.flatnonzero(spotted_by == cam_id)
            records.append({
                'timestamp': t,
                'camera_id': int(cam_id),
                'n_prey': int(np.sum(inds >= config.N_PREDATORS)),
                'n_predators': int(np.sum(inds < config.N_PREDATORS)),
            })
    df = pd.DataFrame.from_records(records,
                                   columns=['timestamp', 'camera_id', 'n_prey', 'n_predators'])
    return df, cam_positions, cam_radii
