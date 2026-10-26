"""Fixtures shared by all pyspar tests."""
import numpy as np
import pyfar as pf
import pytest
from spharpy import SphericalHarmonicSignal
from spharpy.spherical import spherical_harmonic_basis_real


def sh_signal(data, sampling_rate=48000, normalization='N3D',
              channel_convention='ACN'):
    """Return ``(n_coefficients, n_samples)`` data as a signal."""
    return SphericalHarmonicSignal(
        np.asarray(data, dtype=float), sampling_rate, 'real', normalization,
        channel_convention, False)


def encode(pressure, direction, n_max=1, normalization='N3D'):
    """Encode `pressure` as a plane wave arriving from `direction`.

    Parameters
    ----------
    pressure : array_like, shape (n_samples,)
        Pressure at the center of the array.
    direction : pyfar.Coordinates, cshape (1,) or (n_samples,)
        Direction of arrival, one for the whole signal or one per sample.
    n_max : int, optional
        Spherical harmonic order of the encoding.
    normalization : str, optional
        Normalization of the returned signal.

    Returns
    -------
    signal : spharpy.SphericalHarmonicSignal
        Encoded signal of order `n_max`.
    """
    pressure = np.asarray(pressure, dtype=float)
    basis = spherical_harmonic_basis_real(n_max, direction,
                                          normalization=normalization)
    data = pressure[:, None] * basis / np.sqrt(4 * np.pi)
    return sh_signal(data.T, normalization=normalization)


@pytest.fixture
def sampling_rate():
    """Sampling rate used throughout the tests."""
    return 48000


@pytest.fixture
def frontal_impulse():
    """First-order impulse arriving from the front, 64 samples at 48 kHz."""
    pressure = np.zeros(64)
    pressure[8] = 1.0
    front = pf.Coordinates.from_spherical_elevation(0, 0, 1)
    return encode(pressure, front)


def make_impulse_response():
    """First-order impulse response with a direct sound, reflections and tail.

    Deterministic: a direct sound from the front, twelve reflections from
    scattered directions and an incoherent decaying tail, so that the direction
    changes from sample to sample as it does in a measurement.
    """
    rng = np.random.default_rng(0)
    n_samples = 2048

    times = np.concatenate([[32], np.sort(rng.integers(64, 1024, 12))])
    gains = np.concatenate([[1.0], rng.uniform(0.1, 0.5, 12)])
    azimuth = np.concatenate([[0.0], rng.uniform(-np.pi, np.pi, 12)])
    elevation = np.concatenate([[0.0], np.arcsin(rng.uniform(-1, 1, 12))])

    pressure = np.zeros(n_samples)
    pressure[times] = gains
    directions_azimuth = np.zeros(n_samples)
    directions_elevation = np.zeros(n_samples)
    directions_azimuth[times] = azimuth
    directions_elevation[times] = elevation

    tail = slice(1024, n_samples)
    n_tail = n_samples - 1024
    pressure[tail] = (rng.standard_normal(n_tail) * 0.05
                      * np.exp(-np.arange(n_tail) / 400))
    directions_azimuth[tail] = rng.uniform(-np.pi, np.pi, n_tail)
    directions_elevation[tail] = np.arcsin(rng.uniform(-1, 1, n_tail))

    directions = pf.Coordinates.from_spherical_elevation(
        directions_azimuth, directions_elevation, np.ones(n_samples))
    return encode(pressure, directions)


@pytest.fixture
def impulse_response():
    """First-order impulse response: direct sound, reflections, tail."""
    return make_impulse_response()


def layout(azimuth_deg, elevation_deg):
    """Loudspeaker layout from directions in degrees."""
    return pf.Coordinates.from_spherical_elevation(
        np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg),
        np.ones(len(azimuth_deg)))


# [azimuth, elevation] in degrees.
LAYOUT_HORIZONTAL = np.array([[0, 0], [30, 0], [-30, 0], [110, 0],
                              [-110, 0]], float)
LAYOUT_SURROUND = np.array([[0, 0], [60, 0], [120, 0], [180, 0], [-120, 0],
                            [-60, 0], [0, 90], [0, -90]], float)


@pytest.fixture
def layout_horizontal():
    """ITU-R BS.775 5.0 layout, all loudspeakers on the horizon."""
    return layout(*LAYOUT_HORIZONTAL.T)


@pytest.fixture
def layout_surround():
    """Layout enclosing the listener: a ring plus top and bottom."""
    return layout(*LAYOUT_SURROUND.T)


@pytest.fixture
def layout_hemisphere():
    """Upper-hemisphere layout that leaves the lower hemisphere uncovered."""
    azimuth = [0, 90, 180, -90, 45, -135, 0]
    elevation = [0, 0, 0, 0, 40, 40, 90]
    return layout(azimuth, elevation)


@pytest.fixture
def analysis(impulse_response):
    """Analysis of the impulse response with a 5-sample window."""
    from pyspar import parametric
    return parametric.PseudoIntensityVectorAnalyzer(window_length=5)(
        impulse_response)
