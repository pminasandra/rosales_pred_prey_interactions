
# Pranav Minasandra
# 17 Nov 2025
# pminasandra.github.io

from __future__ import annotations
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

import config


# All scenarios use the same model with two switches.
conditions = ['independent', 'predator_chase', 'prey_evade', 'both']
SCENARIOS = {
    'independent': (False, False),
    'predator_chase': (True, False),
    'prey_evade': (False, True),
    'both': (True, True),
}

n_reps = config.N_REPS
T = config.T
box_size = config.BOX_SIZE
base_dir = os.path.join(config.DATA, 'trajectories')


def _wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def _turn_towards(theta, target, strength):
    return _wrap_angle(theta + strength * _wrap_angle(target - theta))


def _reflect(pos, headings, low, high):
    """Reflect coordinates and headings at rectangular boundaries, including overshoots."""
    for axis in range(2):
        width = high[axis] - low[axis]
        phase = np.mod(pos[:, axis] - low[axis], 2.0 * width)
        flip = phase > width
        pos[:, axis] = low[axis] + np.where(flip, 2 * width - phase, phase)
        if axis == 0:
            headings[flip] = np.pi - headings[flip]
        else:
            headings[flip] = -headings[flip]
    return pos, _wrap_angle(headings)


def _validate_parameters():
    if config.N_PREDATORS < 0 or config.N_PREY < 0:
        raise ValueError('Species abundances must be nonnegative.')
    if config.DT <= 0 or config.BOX_SIZE <= 0:
        raise ValueError('DT and BOX_SIZE must be positive.')
    if not (0 < config.PREDATOR_SPEED < config.PREY_SPEED):
        raise ValueError('Require 0 < PREDATOR_SPEED < PREY_SPEED.')
    for name in ('PREDATOR_TURNING_RATE', 'PREY_TURNING_RATE', 'PREY_RETURN_TURNING_RATE'):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f'{name} must be in [0, 1].')
    for name in ('PREDATOR_DETECTION_RADIUS', 'PREY_DETECTION_RADIUS', 'PREDATOR_NOISE', 'PREY_NOISE'):
        if getattr(config, name) < 0:
            raise ValueError(f'{name} must be nonnegative.')
    if config.N_PREY and not config.PREY_HOME_RANGES:
        raise ValueError('At least one prey home range is required.')
    if config.PREY_DETECTION_RADIUS <= config.PREDATOR_DETECTION_RADIUS:
        pass
        #raise ValueError('PREY_DETECTION_RADIUS must exceed PREDATOR_DETECTION_RADIUS.')
    for home in config.PREY_HOME_RANGES:
        x0, y0, x1, y1 = home
        if not (0 <= x0 < x1 <= box_size and 0 <= y0 < y1 <= box_size):
            raise ValueError(f'Invalid home range: {home}')


def trajectory_stream(interaction=None, seed=None):
    """Yield positions indefinitely, with shape (N_PREDATORS + N_PREY, 2).

    `interaction` is a mutable dictionary containing `predator_chase` and
    `prey_evade`. Updating either entry takes effect on the next time step.
    """
    if interaction is None:
        interaction = {'predator_chase': True, 'prey_evade': True}
    _validate_parameters()
    rng = np.random.default_rng(seed)
    npred, nprey = config.N_PREDATORS, config.N_PREY
    N = npred + nprey
    homes = np.asarray(config.PREY_HOME_RANGES, dtype=float)

    if len(homes) != nprey:
        raise ValueError(
            "Exactly one home range must be defined per prey individual."
        )

    prey_homes = homes.copy()

    predators = rng.uniform(0, box_size, (npred, 2))
    prey = np.empty((nprey, 2), dtype=float)
    if nprey:
        prey[:, 0] = rng.uniform(prey_homes[:, 0], prey_homes[:, 2])
        prey[:, 1] = rng.uniform(prey_homes[:, 1], prey_homes[:, 3])
    pred_th = rng.uniform(-np.pi, np.pi, npred)
    prey_th = rng.uniform(-np.pi, np.pi, nprey)
    yield np.concatenate((predators, prey), axis=0).copy()

    while True:
        # All interactions are calculated using positions at the *start* of the step.
        if npred and nprey:
            delta = prey[None, :, :] - predators[:, None, :]
            dist2 = np.sum(delta ** 2, axis=2)
        else:
            dist2 = np.empty((npred, nprey))

        if interaction['predator_chase'] and nprey:
            # Pursue the nearest visible prey, rather than the group centroid.
            nearest = np.argmin(dist2, axis=1) if npred else np.empty(0, int)
            visible = dist2[np.arange(npred), nearest] <= config.PREDATOR_DETECTION_RADIUS ** 2
            desired = np.arctan2(delta[np.arange(npred), nearest, 1],
                                 delta[np.arange(npred), nearest, 0]) if npred else np.empty(0)
            pred_th[visible] = _turn_towards(pred_th[visible], desired[visible],
                                             config.PREDATOR_TURNING_RATE)

        threatened = np.zeros(nprey, dtype=bool)
        if interaction['prey_evade'] and npred and nprey:
            nearest = np.argmin(dist2, axis=0)
            threatened = dist2[nearest, np.arange(nprey)] <= config.PREY_DETECTION_RADIUS ** 2
            away = delta[nearest, np.arange(nprey)]
            desired = np.arctan2(away[:, 1], away[:, 0])
            prey_th[threatened] = _turn_towards(prey_th[threatened], desired[threatened],
                                                config.PREY_TURNING_RATE)

        # An escaped prey returns towards its assigned square if currently outside.
        # This operates even when prey_evade=False, but has no effect on confined prey.
        if nprey:
            nearest_home = np.column_stack((
                np.clip(prey[:, 0], prey_homes[:, 0], prey_homes[:, 2]),
                np.clip(prey[:, 1], prey_homes[:, 1], prey_homes[:, 3]),
            ))
            home_delta = nearest_home - prey
            outside = np.any(np.abs(home_delta) > config.PREY_RETURN_MARGIN, axis=1)
            returning = outside & ~threatened
            desired = np.arctan2(home_delta[:, 1], home_delta[:, 0])
            prey_th[returning] = _turn_towards(prey_th[returning], desired[returning],
                                               config.PREY_RETURN_TURNING_RATE)

        pred_th = _wrap_angle(pred_th + rng.uniform(-config.PREDATOR_NOISE / 2,
                                                     config.PREDATOR_NOISE / 2, npred))
        prey_th = _wrap_angle(prey_th + rng.uniform(-config.PREY_NOISE / 2,
                                                     config.PREY_NOISE / 2, nprey))
        previous_prey = prey.copy()
        predators += config.PREDATOR_SPEED * config.DT * np.column_stack((np.cos(pred_th), np.sin(pred_th)))
        prey += config.PREY_SPEED * config.DT * np.column_stack((np.cos(prey_th), np.sin(prey_th)))

        low = np.zeros(2)
        high = np.full(2, box_size)
        predators, pred_th = _reflect(predators, pred_th, low, high)
        prey, prey_th = _reflect(prey, prey_th, low, high)

        # Outside a home range, prey may travel freely while threatened or returning.
        # Inside the home range, reflecting home walls apply unless threatened.
        for i in range(nprey):
            home = prey_homes[i]
            previous = previous_prey[i]
            previously_inside = (home[0] <= previous[0] <= home[2] and
                                 home[1] <= previous[1] <= home[3])
            if not threatened[i] and previously_inside:
                xy, th = _reflect(prey[i:i+1].copy(), prey_th[i:i+1].copy(),
                                  home[:2], home[2:])
                prey[i], prey_th[i] = xy[0], th[0]

        yield np.concatenate((predators, prey), axis=0).copy()


def simulate(predator_chase=True, prey_evade=True, seed=None, T=None):
    """Simulate T steps and return an array of shape (T+1, N, 2).

    Predators occupy columns [0:N_PREDATORS], prey the remaining columns.
    """
    if T is None:
        T = config.T
    if T < 0:
        raise ValueError('T must be nonnegative.')
    stream = trajectory_stream(
        interaction={'predator_chase': predator_chase, 'prey_evade': prey_evade},
        seed=seed,
    )
    return np.stack([next(stream) for _ in range(T + 1)])


def make_traj(cond, n_reps=None):
    if cond not in SCENARIOS:
        raise ValueError(f'Unknown condition: {cond}')
    if n_reps is None:
        n_reps = config.N_REPS
    cond_dir = os.path.join(base_dir, f'P{config.N_PREDATORS}_H{config.N_PREY}', cond)
    os.makedirs(cond_dir, exist_ok=True)

    npred, nprey = config.N_PREDATORS, config.N_PREY

    columns = (
        [f'prey{i}_{axis}' for i in range(nprey) for axis in ('x', 'y')]
        + [f'predator{i}_{axis}' for i in range(npred) for axis in ('x', 'y')]
    )

    for rep in range(n_reps):
        # Same initial state and random-number stream across conditions.
        seed = config.SEED_BASE + rep
        traj_pos = simulate(*SCENARIOS[cond], seed=seed)

        # Convert from predator-first array to prey-first DataFrame.
        ordered = np.concatenate(
            (traj_pos[:, npred:], traj_pos[:, :npred]), axis=1
        )

        df = pd.DataFrame(
            ordered.reshape(len(traj_pos), -1),
            columns=columns
        )

        out_path = os.path.join(cond_dir, f'sim_{rep:03d}.parquet')
        df.to_parquet(out_path, index=False)
        print(f'Saved {out_path}')


if __name__ == '__main__':
    with mp.Pool() as pool:
        pool.map(make_traj, conditions)
