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


def delayed_impulses(delays, n_samples, sampling_rate=48000):
    """Unit impulses delayed by `delays` seconds, ``(n_samples, len(delays))``.

    Fractional delays are applied as a linear phase, so the impulses are band
    limited rather than rounded to the nearest sample.
    """
    frequencies = np.fft.rfftfreq(n_samples, 1 / sampling_rate)
    phase = np.exp(-2j * np.pi * frequencies[:, None]
                   * np.asarray(delays)[None, :])
    return np.fft.irfft(phase, n=n_samples, axis=0)


def octahedron(radius):
    """Six capsules on the axes at `radius`: +x, -x, +y, -y, +z, -z."""
    xyz = radius * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                             [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    return pf.Coordinates(*xyz.T)


# The time-difference-of-arrival analysis uses pyfar's speed of sound.
SPEED_OF_SOUND = pf.constants.reference_speed_of_sound

# Capsules 4 samples from the centre at 48 kHz, so that a plane wave along an
# axis arrives with whole-sample delays.
OCTAHEDRON_RADIUS = 4 * SPEED_OF_SOUND / 48000


@pytest.fixture
def microphones():
    """Octahedral array with whole-sample delays along the axes."""
    return octahedron(OCTAHEDRON_RADIUS)


def make_array_response(microphones):
    """Capsule signals of a direct sound, reflections and a decaying tail.

    Each arrival is a plane wave with exact fractional delays per capsule.
    """
    rng = np.random.default_rng(11)
    n_samples, fs, c = 600, 48000, SPEED_OF_SOUND
    positions = microphones.cartesian
    data = np.zeros((n_samples, len(positions)))

    times = np.concatenate([[80], rng.uniform(120, 380, 8)])
    gains = np.concatenate([[1.0], rng.uniform(0.2, 0.6, 8)])
    for time, gain in zip(times, gains, strict=True):
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        delays = time / fs - positions @ direction / c
        data += gain * delayed_impulses(delays, n_samples, fs)

    tail = slice(400, n_samples)
    data[tail] += (rng.standard_normal((n_samples - 400, len(positions)))
                   * 0.02 * np.exp(-np.arange(n_samples - 400) / 60)[:, None])
    return pf.Signal(data.T, fs)


@pytest.fixture
def array_response(microphones):
    """Capsule signals of a direct sound, reflections and a decaying tail."""
    return make_array_response(microphones)


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
# Layouts with LFE channels and the indices of those: 5.1 and 5.2.  The
# center loudspeaker comes before the LFE channels, and of the two LFE
# channels the left one comes first.
LAYOUT_LFE_ONE = np.array([[30, 0], [-30, 0], [0, 0], [0, -30], [110, 0],
                           [-110, 0]], float)
LFE_ONE = (3,)
LAYOUT_LFE_TWO = np.array([[30, 0], [-30, 0], [0, 0], [110, 0], [-110, 0],
                           [45, -30], [-45, -30]], float)
LFE_TWO = (5, 6)


@pytest.fixture
def noise_between_silences():
    """Analysis result of decaying noise between long silences.

    Noise from a new direction at every sample, 4000 silent samples before it
    and 3500 after.  Filtered forward and backward, the low frequencies of a
    sound begin before it and last after it, some 60 ms at 80 Hz; the
    silences keep that inside the signal, where the ends would cut it off.
    """
    from pyspar import parametric
    rng = np.random.default_rng(0)
    n_samples = 10000
    pressure = np.zeros(n_samples)
    pressure[4000:6500] = (rng.standard_normal(2500)
                           * np.exp(-np.arange(2500) / 300))
    doa = pf.Coordinates.from_spherical_elevation(
        rng.uniform(-np.pi, np.pi, n_samples),
        np.arcsin(rng.uniform(-1, 1, n_samples)), np.ones(n_samples))
    return parametric.TimeResult(
        pressure=pf.Signal(pressure, 48000), doa=doa,
        signal=pf.Signal(pressure, 48000))


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


# [azimuth, elevation] in degrees: eight azimuths on three elevations, and
# the poles.
HRIR_GRID = np.column_stack([
    np.concatenate([np.tile(np.arange(0, 360, 45), 3), [0, 0]]),
    np.concatenate([np.repeat([-45, 0, 45], 8), [90, -90]])]).astype(float)


def hrir_grid():
    """26 directions: eight azimuths on three elevations, and the poles."""
    return layout(*HRIR_GRID.T)


def make_hrir_set():
    """Seeded random head-related impulse responses on the 26-point grid."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal((26, 2, 32)) * np.exp(-np.arange(32) / 6)
    return pf.Signal(data, 48000), hrir_grid()


@pytest.fixture
def hrir_set():
    """Seeded random head-related impulse responses on a 26-point grid.

    Every direction and both ears differ, so a one-hot loudspeaker signal
    identifies exactly which response was used for which ear.
    """
    return make_hrir_set()


@pytest.fixture
def delta_hrirs():
    """Unit impulses for every direction and ear: a transparent head."""
    data = np.zeros((26, 2, 8))
    data[:, :, 0] = 1.0
    return pf.Signal(data, 48000), hrir_grid()
