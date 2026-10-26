# Movement and camera trap simulations

Predator-prey movement simulations with camera trap detections. Adapted from
[pminasandra/camera_trap_sims](https://github.com/pminasandra/camera_trap_sims),
which was itself used in our work [Goldsborough et al., 2025](https://ecoevorxiv.org/repository/view/10025/)

## Model

Predators and prey perform correlated random walks with independent angular noise.
At the start of each timestep a predator may turn toward its **nearest prey** within
`PREDATOR_DETECTION_RADIUS`; prey may turn away from their **nearest predator** within
`PREY_DETECTION_RADIUS`. The response is a fraction of the angular difference
(`*_TURNING_RATE`), then random noise is added. Animals otherwise retain their
previous heading. All pairwise sensing uses the positions at the beginning of
the timestep. Predators have a lower constant speed than prey. No animals die,
reproduce, eat, or disappear.

Prey are assigned evenly (round robin) to the rectangles in `PREY_HOME_RANGES`.
They start uniformly inside their assigned range. Prey reflect from their own
home-range walls while not threatened. While a predator is within prey-detection
radius, a prey may cross those walls (but still reflects at the global arena
wall). Once outside and no longer threatened, it turns toward the closest point
of its home range, until it returns; thereafter the home walls reflect again.
An unresponsive prey (`prey_evade=False`) remains within its home range.

## Four scenarios

`trajectories.SCENARIOS` maps names to `(predator_chase, prey_evade)`:

- `independent`: `(False, False)`
- `predator_chase`: `(True, False)`
- `prey_evade`: `(False, True)`
- `both`: `(True, True)`

## Run

Requires Python >= 3.10, `numpy`, `pandas` and a parquet backend (`pyarrow`).
Set biological and experimental parameters in `config.py`, then run:

```bash
python main.py
```

Change `N_REPS` and `T` to smaller values for a test run. For a single test:

```python
import trajectories
positions = trajectories.simulate(predator_chase=True, prey_evade=True, seed=1, T=100)
```

## Saved outputs

- `Data/trajectories/P{N_PREDATORS}_H{N_PREY}/{condition}/sim_###.npy`
  shape `(T+1, N_PREDATORS+N_PREY, 2)`; **predators first, prey second**.
- `Data/sightings/.../sim_###_ncams_K.parquet`: only triggered cameras,
  columns `timestamp`, `camera_id`, `n_prey`, `n_predators`. Each individual
  is assigned to its nearest detecting camera if detection circles overlap.
  Counts may be zero for one species but cannot both be zero.
- `Data/cameras/global_ncams_K.parquet`: fixed grid positions and radii.

A fixed seed per replicate makes different scenarios comparable. Cameras use a
separate seeded RNG. Any changes to `N_PREDATORS` and `N_PREY` require regenerating
trajectories; detections validate the stored population size.

## Notes

Original `vicsek.py` is conceptually superseded: this model is **not** Vicsek
alignment but coupled correlated random walks. Original visualisation utilities
are not required to run the experiment.
