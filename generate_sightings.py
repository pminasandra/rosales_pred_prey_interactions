# generate_sightings.py
from __future__ import annotations
from typing import Iterable, Tuple

from pathlib import Path
import numpy as np
import pandas as pd

import config
import cameras
from detections import run_cameras_on_trajectory


def make_camsets(camera_sizes=None):
    if camera_sizes is None:
        camera_sizes = config.CAMERA_SIZES

    camsets = {}

    for s in camera_sizes:
        cam_radii = np.full(s, config.CAMERA_RADIUS)

        camsets[s] = cameras.make_camera_grid(
            s, cam_radii, box_size=config.BOX_SIZE
        )

    return camsets


CAMSETS = make_camsets()


def generate_all_sightings(
    *,
    cams: dict | None = None,
    camera_sizes=None,
    radius: float | Iterable = 0.5,
    box_size: float = config.BOX_SIZE,
    seed_base: int = config.CAMERA_SEED,
):
    """Generate aggregate predator/prey counts for all saved trajectories."""
    if camera_sizes is None:
        camera_sizes = config.CAMERA_SIZES
    traj_root = Path(config.DATA) / 'trajectories'
    sightings_root = Path(config.DATA) / 'sightings'
    cameras_root = Path(config.DATA) / 'cameras'
    if not traj_root.exists():
        raise FileNotFoundError(f'No trajectories found in {traj_root}')
    if cams is not None:
        cameras_root.mkdir(parents=True, exist_ok=True)
        for s in camera_sizes:
            file = np.column_stack((cams[s][0], cams[s][1]))
            fname = cameras_root / f'global_ncams_{s}.parquet'
            pd.DataFrame(file, columns=['x', 'y', 'r']).to_parquet(fname, index=False)
            print('Saved', fname)
    for N_dir in sorted(traj_root.iterdir()):
        if not N_dir.is_dir():
            continue
        for cond_dir in sorted(N_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            sight_dir = sightings_root / N_dir.name / cond_dir.name
            cam_dir = cameras_root / N_dir.name / cond_dir.name
            sight_dir.mkdir(parents=True, exist_ok=True)
            cam_dir.mkdir(parents=True, exist_ok=True)
            for traj_file in sorted(cond_dir.glob('sim_*.parquet')):
                base = traj_file.stem
                seed_for_this_sim = seed_base + int(base.split('_')[1])
                for k in camera_sizes:
                    df_sight, cam_positions, cam_radii = run_cameras_on_trajectory(
                        traj_path=traj_file, cams=cams[k] if cams is not None else None,
                        m=k, radius=radius, box_size=box_size,
                        seed=seed_for_this_sim)
                    sight_out = sight_dir / f'{base}_ncams_{k}.parquet'
                    df_sight.to_parquet(sight_out, index=False)
                    print(f'Saved: {sight_out}')
                    if cams is None:
                        cam_out = cam_dir / f'{base}_ncams_{k}.parquet'
                        pd.DataFrame(np.column_stack((cam_positions, cam_radii)),
                                     columns=['x', 'y', 'r']).to_parquet(cam_out, index=False)
                        print(f'Saved: {cam_out}')


if __name__ == '__main__':
    generate_all_sightings(cams=make_camsets())
