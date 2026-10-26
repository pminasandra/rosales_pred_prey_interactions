# Pranav Minasandra
# pminasandra.github.io
# November 06, 2025

import os
import os.path
import numpy as np

# Directories
PROJECTROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.isfile('.cw'):
    with open('.cw', 'r') as f:
        PROJECTROOT = os.path.abspath(f.read().strip())
DATA = os.path.join(PROJECTROOT, 'Data')
FIGURES = os.path.join(PROJECTROOT, 'Figures')
formats = ['png', 'pdf', 'svg']

# Miscellaneous
SUPPRESS_INFORMATIVE_PRINT = False

# Biological parameters (one time step is the movement unit)
T = 1500
BOX_SIZE = 100.0
DT = 1.0
N_REPS = 200
SEED_BASE = 0

# Coordinates are (xmin, ymin, xmax, ymax). Each prey is assigned one home range.
# The number of prey is distributed as evenly as possible across these squares.
N_PREDATORS = 1
N_PREY = 16

PREY_HOME_SIZE = 40.0

PREY_HOME_CENTERS = np.linspace(
    PREY_HOME_SIZE / 2,
    BOX_SIZE - PREY_HOME_SIZE / 2,
    4
)

PREY_HOME_RANGES = [
    (
        x - PREY_HOME_SIZE / 2,
        y - PREY_HOME_SIZE / 2,
        x + PREY_HOME_SIZE / 2,
        y + PREY_HOME_SIZE / 2
    )
    for x in PREY_HOME_CENTERS
    for y in PREY_HOME_CENTERS
]

PREDATOR_SPEED = 0.30
PREY_SPEED = 0.6
PREDATOR_NOISE = 0.30       # uniform angular noise, in radians (full width)
PREY_NOISE = 0.40
PREDATOR_TURNING_RATE = 0.35  # fraction of desired direction adopted each step
PREY_TURNING_RATE = 0.55
PREDATOR_DETECTION_RADIUS = 14.0
PREY_DETECTION_RADIUS = 8.0

# Once a prey is no longer threatened, it moves toward its home range.
# It is allowed to leave its square only while a predator is detected.
PREY_RETURN_TURNING_RATE = 0.35
PREY_RETURN_MARGIN = 0.0

# Camera configuration (independent of biological movement)
CAMERA_SIZES = [4, 9, 16, 25, 36, 49, 64, 81]
CAMERA_RADIUS = 0.75
CAMERA_SEED = 1234
