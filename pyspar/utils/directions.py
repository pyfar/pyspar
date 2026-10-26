"""Directions on the sphere: matched by angle and filled in where missing."""
import warnings

import numpy as np
import pyfar as pf


def nearest_directions(directions, targets):
    """Index of the direction nearest each target, by angle alone.

    Directions given at different distances, such as an HRIR set and a
    loudspeaker layout, would bias a Euclidean match, so both are projected
    onto the unit sphere first.

    Parameters
    ----------
    directions : pyfar.Coordinates
        Directions to choose from, ``cshape (n_directions,)``.
    targets : pyfar.Coordinates
        Directions to match.

    Returns
    -------
    index : numpy.ndarray, shape (n_targets,)
        Index into `directions` per target.
    """
    def on_unit_sphere(coordinates):
        xyz = np.atleast_2d(coordinates.cartesian).reshape(-1, 3)
        return pf.Coordinates(
            *(xyz / np.linalg.norm(xyz, axis=1, keepdims=True)).T)

    index, _ = on_unit_sphere(directions).find_nearest(
        on_unit_sphere(targets), distance_measure='spherical_radians')
    return np.asarray(index).ravel()


def _replace_missing_doa(doa, pressure):
    """Return unit directions with undefined ones pointing forward.

    Parameters
    ----------
    doa : pyfar.Coordinates, cshape (n_samples,)
        Direction of arrival per sample.
    pressure : numpy.ndarray, shape (n_samples,)
        Pressure, used to tell silent samples from samples carrying energy.

    Returns
    -------
    directions : numpy.ndarray, shape (n_samples, 3)
        Unit Cartesian directions.

    Notes
    -----
    Samples without a direction are NaN. The pseudo-intensity vector has none
    where the signal is digitally silent within its window and where the
    omnidirectional channel is exactly zero, which recordings of finite
    resolution reach near some of its zero crossings. These samples are placed
    straight ahead, with a warning when they carry more than :math:`10^{-4}`
    (-40 dB) of the energy of the response.
    """
    directions = np.atleast_2d(doa.cartesian).reshape(-1, 3).copy()
    missing = np.isnan(directions).any(axis=1)

    energy = np.sum(pressure ** 2)
    share = np.sum(pressure[missing] ** 2) / energy if energy > 0 else 0.0
    if share > 1e-4:
        warnings.warn(
            f"{int(np.sum(missing & (pressure != 0)))} of {len(directions)} "
            "samples have no direction but carry pressure "
            f"({10 * np.log10(share):.1f} dB of the energy of the response); "
            "they are placed straight ahead.",
            stacklevel=3)

    directions[missing] = [1.0, 0.0, 0.0]
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)
