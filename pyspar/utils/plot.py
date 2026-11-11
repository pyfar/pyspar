"""Helpers of the maps of the sphere."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy


def _hammer_aitoff(azimuth, elevation):
    """Hammer-Aitoff projection seen from inside the sphere.

    Positive azimuths, to the left of the listener, map to negative x.
    """
    azimuth = np.mod(np.asarray(azimuth, dtype=float) + np.pi,
                     2 * np.pi) - np.pi
    elevation = np.asarray(elevation, dtype=float)
    scale = np.sqrt(1 + np.cos(elevation) * np.cos(azimuth / 2))
    return np.stack([-np.cos(elevation) * np.sin(azimuth / 2) / scale,
                     0.5 * np.sin(elevation) / scale], axis=-1)


def _draw_map_grid(ax):
    """Meridians and parallels of the map with their angles in degrees.

    The angles are written below the lines: the azimuths from -150 to 150 on
    the equator, the elevations on the meridian of azimuth zero.
    """
    gray = '0.7'
    below = offset_copy(ax.transData, fig=ax.figure, y=-4, units='points')
    elevations = np.linspace(-np.pi / 2, np.pi / 2, 50)
    for azimuth in (-179.9, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120,
                    150, 179.9):
        xy = _hammer_aitoff(np.full(50, np.deg2rad(azimuth)), elevations)
        ax.plot(*xy.T, '--', color=gray, linewidth=0.5)
        if abs(azimuth) == 179.9:
            # Both edges are the meridian behind the listener.
            continue
        ax.text(*_hammer_aitoff(np.deg2rad(azimuth), 0), f"{azimuth:.0f}",
                transform=below, fontsize=8, color=gray, ha='center',
                va='top')
    # 180 degrees would wrap to -180 and close each parallel across the map.
    azimuths = np.deg2rad(np.linspace(-179.9, 179.9, 100))
    for elevation in (-60, -30, 0, 30, 60):
        xy = _hammer_aitoff(azimuths, np.full(100, np.deg2rad(elevation)))
        ax.plot(*xy.T, '--', color=gray, linewidth=0.5)
        if elevation:
            ax.text(*_hammer_aitoff(0, np.deg2rad(elevation)),
                    f"{elevation}", transform=below, fontsize=8, color=gray,
                    ha='center', va='top')


def _project(points):
    """Map positions of Cartesian points by direction, shape (n_points, 2)."""
    return _hammer_aitoff(
        np.arctan2(points[:, 1], points[:, 0]),
        np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))


def _map_axes(ax):
    """Axes with the grid of the map, new if `ax` is None."""
    if ax is None:
        _, ax = plt.subplots()
    _draw_map_grid(ax)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax


def _draw_loudspeakers(ax, real, imaginary):
    xy = _project(real)
    ax.scatter(*xy.T, s=64, color='k', edgecolors='w', linewidths=0.5,
               zorder=3)
    if len(imaginary):
        ax.scatter(*_project(imaginary).T, s=64, facecolors='none',
                   edgecolors='k', zorder=3)
