"""Tests of the Ambisonic encoder."""
import warnings
from math import factorial

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
from conftest import encode
from scipy.special import lpmv
from spharpy import SphericalHarmonicSignal

from pyspar import parametric


def sh_basis_direct(order, azimuth, elevation):
    """Real spherical harmonics in ACN order, orthonormal on the sphere.

    Written out from the associated Legendre functions, as an independent
    specification of the basis. scipy's ``lpmv`` includes the
    Condon-Shortley phase, which ``(-1)^m`` removes.
    """
    basis = np.zeros((azimuth.size, (order + 1) ** 2))
    for n in range(order + 1):
        for m in range(-n, n + 1):
            k = abs(m)
            legendre = lpmv(k, n, np.sin(elevation))
            norm = np.sqrt((2 * n + 1) * factorial(n - k)
                           / (4 * np.pi * factorial(n + k)))
            if m < 0:
                azimuthal = np.sqrt(2) * np.sin(k * azimuth)
            elif m == 0:
                azimuthal = np.ones_like(azimuth)
            else:
                azimuthal = np.sqrt(2) * np.cos(k * azimuth)
            basis[:, n * n + n + m] = (-1) ** m * legendre * norm * azimuthal
    return basis


def encoded(result, order, **kwargs):
    """Ambisonic time data as ``(n_samples, n_coefficients)``."""
    signal = parametric.AmbisonicSynthesizer(order, **kwargs)(result)
    return np.asarray(signal.time)[0].T


def degree_of(n_coefficients):
    """Spherical harmonic degree of each ACN channel."""
    return np.floor(np.sqrt(np.arange(n_coefficients))).astype(int)


# ----- encoding --------------------------------------------------------------

@pytest.mark.parametrize('order', [1, 2, 4])
def test_matches_the_direct_computation(analysis, order):
    directions = analysis.doa.cartesian
    sounding = analysis.pressure.time[0] != 0
    azimuth = np.arctan2(directions[sounding, 1], directions[sounding, 0])
    elevation = np.arcsin(np.clip(directions[sounding, 2], -1, 1))
    pressure = analysis.pressure.time[0][sounding] * np.sqrt(4 * np.pi)

    expected = pressure[:, None] * sh_basis_direct(order, azimuth,
                                                   elevation)

    npt.assert_allclose(encoded(analysis, order)[sounding], expected,
                        atol=1e-13)


@pytest.mark.parametrize(('azimuth', 'elevation'), [
    (-135, 30), (10, 89.9)])
def test_first_order_reproduces_a_plane_wave(azimuth, elevation):
    direction = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(azimuth), np.deg2rad(elevation), 1)
    pressure = np.zeros(32)
    pressure[5] = 1.0
    signal = encode(pressure, direction)
    result = parametric.PseudoIntensityVectorAnalyzer(None)(signal)

    npt.assert_allclose(encoded(result, 1), np.asarray(signal.time)[0].T,
                        atol=1e-15)


def test_sn3d_is_n3d_scaled_per_degree(analysis):
    n3d = encoded(analysis, 3, normalization='N3D')
    sn3d = encoded(analysis, 3, normalization='SN3D')

    scale = np.sqrt(2 * degree_of(16) + 1)
    npt.assert_allclose(sn3d * scale, n3d, rtol=1e-13, atol=1e-30)


def test_output_is_a_tagged_spherical_harmonic_signal(analysis):
    signal = parametric.AmbisonicSynthesizer(3, normalization='SN3D')(analysis)

    assert isinstance(signal, SphericalHarmonicSignal)
    assert signal.n_max == 3
    assert signal.normalization == 'SN3D'
    assert signal.channel_convention == 'ACN'
    assert signal.basis_type == 'real'
    assert not signal.condon_shortley
    assert signal.n_samples == analysis.pressure.n_samples
    assert signal.sampling_rate == analysis.sampling_rate


# ----- samples without a direction -----------------------------------------

def test_silent_samples_are_encoded_silently(analysis):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        data = encoded(analysis, 2)

    npt.assert_array_equal(data[analysis.pressure.time[0] == 0], 0.0)


def test_sounding_sample_without_a_direction_is_encoded_in_front(analysis):
    directions = analysis.doa.cartesian.copy()
    loudest = np.argmax(np.abs(analysis.pressure.time[0]))
    directions[loudest] = np.nan
    result = parametric.TimeResult(
        pressure=analysis.pressure, doa=pf.Coordinates(*directions.T),
        signal=analysis.signal)

    with pytest.warns(UserWarning, match='placed straight ahead'):
        data = encoded(result, 2)

    front = sh_basis_direct(2, np.array([0.0]), np.array([0.0]))[0]
    npt.assert_allclose(
        data[loudest],
        analysis.pressure.time[0][loudest] * np.sqrt(4 * np.pi) * front,
        atol=1e-15)


@pytest.mark.filterwarnings('ignore:.*samples have no direction')
def test_tdoa_analysis_encodes_to_ambisonics(array_response, microphones):
    result = parametric.TimeDifferenceOfArrivalAnalyzer(
        microphones, reference_channel=4)(array_response)

    encoded = parametric.AmbisonicSynthesizer(3)(result)

    # The image-source distances of TDOA do not leak into the encoding.
    npt.assert_allclose(np.asarray(encoded.time)[0, 0],
                        result.pressure.time[0],
                        rtol=1e-14, atol=1e-30)


# ----- parameters and errors -----------------------------------------------

def test_default_parameters():
    synthesizer = parametric.AmbisonicSynthesizer(3)

    assert synthesizer.order == 3
    assert synthesizer.method == 'asdm'
    assert synthesizer.normalization == 'N3D'


def test_order_accepts_numpy_integers():
    assert parametric.AmbisonicSynthesizer(np.int64(2)).order == 2


def test_error_if_the_order_is_negative():
    with pytest.raises(ValueError, match='order is -1 but must be non-'):
        parametric.AmbisonicSynthesizer(-1)


@pytest.mark.parametrize('order', [1.5, '3', True, None])
def test_error_if_the_order_is_not_an_integer(order):
    with pytest.raises(TypeError, match='order must be an integer'):
        parametric.AmbisonicSynthesizer(order)


def test_error_if_the_method_is_unknown():
    with pytest.raises(ValueError, match="method is '4d_asdm' but must"):
        parametric.AmbisonicSynthesizer(3, method='4d_asdm')


@pytest.mark.parametrize('normalization', ['maxN', 'FuMa', 'n3d'])
def test_error_if_the_normalization_is_unknown(normalization):
    with pytest.raises(ValueError, match='normalization is .* but must'):
        parametric.AmbisonicSynthesizer(3, normalization=normalization)


def test_error_if_the_input_is_not_an_analysis_result():
    with pytest.raises(TypeError,
                       match='requires a pyspar.parametric.TimeResult'):
        parametric.AmbisonicSynthesizer(2)(np.zeros(16))
