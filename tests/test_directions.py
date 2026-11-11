"""Tests of the helpers for directions on the sphere."""
import warnings

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest

from pyspar.utils.directions import _replace_missing_doa


def doa(rows):
    """Directions from Cartesian rows."""
    return pf.Coordinates(*np.asarray(rows, dtype=float).T)


def test_missing_directions_point_straight_ahead():
    directions = _replace_missing_doa(doa([[0, 2, 0], [np.nan] * 3]),
                                      np.array([1.0, 0.0]))

    npt.assert_array_equal(directions, [[0, 1, 0], [1, 0, 0]])


def test_warns_above_minus_40_db_of_the_energy():
    # The second sample carries 1.2e-4 of the energy.
    with pytest.warns(UserWarning, match='1 of 2 samples have no direction'):
        _replace_missing_doa(doa([[1, 0, 0], [np.nan] * 3]),
                             np.array([1.0, 0.011]))


def test_no_warning_below_minus_40_db_of_the_energy():
    # The second sample carries 9.8e-5 of the energy.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _replace_missing_doa(doa([[1, 0, 0], [np.nan] * 3]),
                             np.array([1.0, 0.0099]))


def test_silent_response_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        directions = _replace_missing_doa(doa([[np.nan] * 3] * 2),
                                          np.zeros(2))

    npt.assert_array_equal(directions, [[1, 0, 0], [1, 0, 0]])
