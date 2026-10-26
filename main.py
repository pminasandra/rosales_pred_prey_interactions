# Pranav Minasandra
# pminasandra.github.io
# Dec 5 2025

import multiprocessing as mp
import os

import config
import trajectories
import generate_sightings


if __name__ == '__main__':
    print('Creating trajectories. (This might take a while)')
    os.makedirs(trajectories.base_dir, exist_ok=True)

    # Four scenarios correspond to (predator_chase, prey_evade) booleans.
    with mp.Pool() as pool:
        pool.map(trajectories.make_traj, trajectories.conditions)

    print('Generating pseudocameratrap datasets.')
    generate_sightings.generate_all_sightings(
        cams=generate_sightings.make_camsets())
