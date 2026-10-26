
# Pranav Minasandra
# Camera-trap movement simulation: interactive demo

from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.colors import to_rgb
from matplotlib.widgets import RadioButtons
import numpy as np

import config
import trajectories
import cameras


TRAIL_LENGTH = 20            # number of previous movement steps shown
FRAME_INTERVAL = 40         # milliseconds per frame
STEPS_PER_FRAME = 1

PREY_COLOR = '#258b50'
PREDATOR_COLOR = '#8b2026'
BOTH_COLOR = '#ca9a38'

# Demo-specific camera settings (increase radius to make flashes observable).
CAMERA_COUNT = 49             # must be a perfect square
CAMERA_RADIUS = 2.0           # sensing radius in arena units
CAMERA_IDLE_ALPHA = 0.04
CAMERA_ACTIVE_ALPHA = 0.50


def animate(seed=None):
    """Show a continuously running simulation with live interaction controls."""
    modes = list(trajectories.SCENARIOS)
    interaction = dict(zip(('predator_chase', 'prey_evade'),
                           trajectories.SCENARIOS['both']))
    stream = trajectories.trajectory_stream(interaction=interaction, seed=seed)
    history = deque([next(stream)], maxlen=TRAIL_LENGTH + 1)
    npred = config.N_PREDATORS
    nprey = config.N_PREY

    cam_positions, cam_radii = cameras.create_cameras(
        CAMERA_COUNT, radius=CAMERA_RADIUS, box_size=config.BOX_SIZE,
        rng=np.random.default_rng(config.CAMERA_SEED))

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(right=0.74)
    ax.set_xlim(0, config.BOX_SIZE)
    ax.set_ylim(0, config.BOX_SIZE)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Circles show the actual camera detection footprints.
    camera_circles = []
    for (x, y), radius in zip(cam_positions, cam_radii):
        circle = Circle((x, y), radius, facecolor=(*to_rgb('grey'), CAMERA_IDLE_ALPHA),
                        edgecolor='none', zorder=0)
        ax.add_patch(circle)
        camera_circles.append(circle)

    predator_points = ax.scatter([], [], s=70, c=PREDATOR_COLOR, marker='s',
                                 zorder=3, label='Predator')
    prey_points = ax.scatter([], [], s=15, c=PREY_COLOR, marker='o',
                             zorder=3, label='Prey')

    predator_trails = LineCollection([], linewidths=1.6, zorder=2)
    prey_trails = LineCollection([], linewidths=0.9, zorder=2)
    ax.add_collection(predator_trails)
    ax.add_collection(prey_trails)
    #ax.legend(loc='upper left')

    radio_ax = fig.add_axes((0.77, 0.61, 0.22, 0.22))
    radio_ax.set_title('Interaction', fontsize=10)
    radio = RadioButtons(radio_ax, modes, active=modes.index('both'))

    def change_mode(label):
        interaction['predator_chase'], interaction['prey_evade'] = \
            trajectories.SCENARIOS[label]

    radio.on_clicked(change_mode)

    def update(frame):
        for _ in range(STEPS_PER_FRAME):
            history.append(next(stream))
        current = history[-1]

        # Each detected individual is assigned to at most one camera.
        spotted_by = cameras.run_cameras(cam_positions, cam_radii, current)
        predator_counts = np.bincount(spotted_by[:npred][spotted_by[:npred] >= 0],
                                      minlength=len(cam_positions))
        prey_counts = np.bincount(spotted_by[npred:][spotted_by[npred:] >= 0],
                                  minlength=len(cam_positions))
        for cam_id, circle in enumerate(camera_circles):
            if predator_counts[cam_id] and prey_counts[cam_id]:
                color, alpha = BOTH_COLOR, CAMERA_ACTIVE_ALPHA
            elif predator_counts[cam_id]:
                color, alpha = PREDATOR_COLOR, CAMERA_ACTIVE_ALPHA
            elif prey_counts[cam_id]:
                color, alpha = PREY_COLOR, CAMERA_ACTIVE_ALPHA
            else:
                color, alpha = 'grey', CAMERA_IDLE_ALPHA
            circle.set_facecolor((*to_rgb(color), alpha))
        predator_points.set_offsets(current[:npred])
        prey_points.set_offsets(current[npred:])

        # Segment opacity decreases with age (oldest -> newest).
        old_to_new = list(history)
        pred_segments, prey_segments = [], []
        pred_colors, prey_colors = [], []
        for age, (before, after) in enumerate(zip(old_to_new[:-1], old_to_new[1:])):
            alpha = 0.12 + 0.70 * (age + 1) / TRAIL_LENGTH
            if npred:
                pred_segments.extend(np.stack((before[:npred], after[:npred]), axis=1))
                pred_colors.extend([(0.545, 0.125, 0.149, alpha)] * npred)
            if nprey:
                prey_segments.extend(np.stack((before[npred:], after[npred:]), axis=1))
                prey_colors.extend([(0.145, 0.545, 0.314, alpha)] * nprey)

        predator_trails.set_segments(pred_segments)
        predator_trails.set_color(pred_colors)
        prey_trails.set_segments(prey_segments)
        prey_trails.set_color(prey_colors)
        return (predator_points, prey_points, predator_trails, prey_trails,
                *camera_circles)

    ani = FuncAnimation(fig, update, interval=FRAME_INTERVAL, blit=False,
                        cache_frame_data=False)
    plt.show()
    return ani


if __name__ == '__main__':
    animate(seed=config.SEED_BASE)
