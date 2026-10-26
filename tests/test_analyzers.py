"""Tests of the direction-of-arrival analyzer."""
import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
from conftest import encode, sh_signal
from spharpy import SphericalHarmonicSignal

from pyspar import parametric


def piv_direct(data, window_length):
    """Unit pseudo-intensity vector smoothed by a centred Hann window.

    The product of the omnidirectional channel with each dipole channel is
    convolved with a symmetric Hann window of `window_length` samples
    without zero end points, centred, and the vector is normalized without
    a guard, so silent samples come out as NaN.
    """
    vector = np.column_stack([data[:, 0] * data[:, 3],
                              data[:, 0] * data[:, 1],
                              data[:, 0] * data[:, 2]])
    window = np.hanning(window_length + 2)[1:-1]
    smoothed = np.column_stack([
        np.convolve(component, window, mode='same') for component in vector.T])
    with np.errstate(invalid='ignore'):
        return smoothed / np.linalg.norm(smoothed, axis=1, keepdims=True)


@pytest.mark.parametrize(('azimuth', 'elevation'), [
    (0, 0), (90, 0), (-90, 0), (180, 0), (37, 21), (-120, -45),
    (0, 89.9), (0, -89.9)])
def test_plane_wave_direction_is_recovered(azimuth, elevation):
    direction = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(azimuth), np.deg2rad(elevation), 1)
    pressure = np.zeros(32)
    pressure[4] = 1.0

    doa = parametric.PseudoIntensityVectorAnalyzer()(
        encode(pressure, direction)).doa

    npt.assert_allclose(doa.cartesian[4], direction.cartesian[0], atol=1e-12)


@pytest.mark.parametrize('window_length', [1, 3, 5, 15])
def test_matches_the_direct_computation(impulse_response, window_length):
    data = np.asarray(impulse_response.time)[0].T
    expected = piv_direct(data, window_length)

    doa = parametric.PseudoIntensityVectorAnalyzer(
        window_length=window_length)(impulse_response).doa.cartesian

    defined = np.all(np.isfinite(expected), axis=1)
    npt.assert_allclose(doa[defined], expected[defined], atol=1e-12)


def test_silent_samples_carry_no_direction(frontal_impulse):
    doa = parametric.PseudoIntensityVectorAnalyzer()(
        frontal_impulse).doa.cartesian

    assert np.isnan(doa[:8]).all()
    assert np.isnan(doa[9:]).all()


@pytest.mark.parametrize('window_length', [1, 3, 7, 15])
def test_window_spreads_an_impulse_over_its_length_centred(window_length):
    pressure = np.zeros(64)
    pressure[10] = 1.0
    left = pf.Coordinates.from_spherical_elevation(np.pi / 2, 0, 1)

    doa = parametric.PseudoIntensityVectorAnalyzer(
        window_length=window_length)(encode(pressure, left)).doa.cartesian

    half = window_length // 2
    npt.assert_allclose(doa[10 - half:11 + half],
                        [[0, 1, 0]] * window_length, atol=1e-12)
    assert np.isnan(doa[:10 - half]).all()
    assert np.isnan(doa[11 + half:]).all()


def test_window_smooths_the_vector_not_the_channels():
    # An arrival from the left at sample 10 and a weaker one from the right
    # at sample 12, energies 1 and 0.16.  The vector smoothed by the Hann
    # window [0.25, 0.75, 1, 0.75, 0.25] is their weighted mean: at sample
    # 12 it is 0.25 - 0.16 > 0, still left.  Smoothing the channels first
    # would give (w10 + 0.4 w12)(w10 - 0.4 w12) = 0.0625 - 0.16 < 0 there,
    # the right.
    left = pf.Coordinates.from_spherical_elevation(np.pi / 2, 0, 1)
    right = pf.Coordinates.from_spherical_elevation(-np.pi / 2, 0, 1)
    sample = np.arange(64)
    data = (np.asarray(encode(np.where(sample == 10, 1.0, 0.0), left).time)
            + np.asarray(encode(np.where(sample == 12, 0.4, 0.0), right).time))

    doa = parametric.PseudoIntensityVectorAnalyzer(window_length=5)(
        sh_signal(data[0])).doa.cartesian

    npt.assert_allclose(doa[8:13, 1], 1.0, atol=1e-12)
    npt.assert_allclose(doa[13:15, 1], -1.0, atol=1e-12)
    assert np.isnan(doa[:8]).all()
    assert np.isnan(doa[15:]).all()


def test_window_is_the_actual_window_length(impulse_response):
    data = np.asarray(impulse_response.time)[0].T

    doa = parametric.PseudoIntensityVectorAnalyzer(window_length=15)(
        impulse_response).doa.cartesian

    assert not np.allclose(doa[np.isfinite(doa).all(1)],
                           piv_direct(data, 7)[np.isfinite(doa).all(1)])


@pytest.mark.parametrize(('normalization', 'channel_convention'), [
    ('SN3D', 'ACN'), ('maxN', 'ACN'), ('N3D', 'FuMa'), ('SN3D', 'FuMa'),
    ('maxN', 'FuMa')])
def test_result_does_not_depend_on_the_input_convention(
        impulse_response, normalization, channel_convention):
    converted = impulse_response.copy()
    converted.normalization = normalization
    converted.channel_convention = channel_convention
    analyzer = parametric.PseudoIntensityVectorAnalyzer(window_length=5)

    reference = analyzer(impulse_response)
    result = analyzer(converted)

    npt.assert_allclose(result.doa.cartesian, reference.doa.cartesian,
                        atol=1e-12)
    npt.assert_allclose(result.pressure.time[0], reference.pressure.time[0],
                        atol=1e-15)


def test_pressure_is_the_omnidirectional_channel(impulse_response):
    result = parametric.PseudoIntensityVectorAnalyzer()(impulse_response)

    npt.assert_array_equal(result.pressure.time[0],
                           np.asarray(impulse_response.time)[0, 0])


def test_result_signal_is_first_order_acn_n3d(impulse_response):
    converted = impulse_response.copy()
    converted.normalization = 'SN3D'

    signal = parametric.PseudoIntensityVectorAnalyzer()(converted).signal

    assert signal.n_max == 1
    assert signal.normalization == 'N3D'
    assert signal.channel_convention == 'ACN'
    npt.assert_allclose(np.asarray(signal.time),
                        np.asarray(impulse_response.time), atol=1e-15)


def test_takes_the_sampling_rate_from_the_signal(impulse_response):
    signal = sh_signal(np.asarray(impulse_response.time)[0],
                       sampling_rate=44100)

    result = parametric.PseudoIntensityVectorAnalyzer()(signal)

    assert result.sampling_rate == 44100


def test_doa_has_one_direction_per_sample(impulse_response):
    doa = parametric.PseudoIntensityVectorAnalyzer()(impulse_response).doa

    assert isinstance(doa, pf.Coordinates)
    assert doa.cshape == (impulse_response.n_samples,)


def test_higher_order_input_is_truncated_with_a_warning(impulse_response):
    data = np.zeros((9, impulse_response.n_samples))
    data[:4] = np.asarray(impulse_response.time)[0]

    with pytest.warns(UserWarning, match='order 2.*first order only'):
        result = parametric.PseudoIntensityVectorAnalyzer()(sh_signal(data))

    reference = parametric.PseudoIntensityVectorAnalyzer()(impulse_response)
    npt.assert_array_equal(result.doa.cartesian, reference.doa.cartesian)
    assert result.signal.n_max == 1


def test_higher_order_input_is_converted_before_truncation(
        impulse_response):
    data = np.zeros((9, impulse_response.n_samples))
    data[:4] = np.asarray(impulse_response.time)[0]
    sn3d = sh_signal(data)
    sn3d.normalization = 'SN3D'

    with pytest.warns(UserWarning, match='order 2'):
        signal = parametric.PseudoIntensityVectorAnalyzer()(sn3d).signal

    npt.assert_allclose(np.asarray(signal.time),
                        np.asarray(impulse_response.time), atol=1e-15)


def test_default_parameters():
    analyzer = parametric.PseudoIntensityVectorAnalyzer()

    assert analyzer.window_length == 1


@pytest.mark.parametrize('window_length', [0, -3])
def test_error_if_the_window_is_shorter_than_one_sample(window_length):
    with pytest.raises(ValueError, match='window_length is .* at least 1'):
        parametric.PseudoIntensityVectorAnalyzer(window_length=window_length)


@pytest.mark.parametrize('window_length', [2, 10])
def test_error_if_the_window_is_even(window_length):
    with pytest.raises(ValueError, match='must be odd'):
        parametric.PseudoIntensityVectorAnalyzer(window_length=window_length)


def test_error_if_the_window_is_not_an_integer():
    with pytest.raises(TypeError,
                       match='window_length must be an integer; got float'):
        parametric.PseudoIntensityVectorAnalyzer(window_length=2.5)


def test_error_if_the_input_is_an_array(impulse_response):
    with pytest.raises(TypeError, match='requires a spharpy'):
        parametric.PseudoIntensityVectorAnalyzer()(
            np.asarray(impulse_response.time)[0].T)


def test_error_if_the_input_has_no_dipole_channels():
    with pytest.raises(ValueError, match='at least order 1, got order 0'):
        parametric.PseudoIntensityVectorAnalyzer()(
            sh_signal(np.zeros((1, 16))))


def test_error_if_the_basis_is_complex():
    signal = SphericalHarmonicSignal(
        np.zeros((4, 16), dtype=complex), 48000, 'complex', 'N3D', 'ACN', True,
        is_complex=True)

    with pytest.raises(ValueError, match='real spherical harmonic basis'):
        parametric.PseudoIntensityVectorAnalyzer()(signal)


def test_error_if_the_condon_shortley_phase_is_included():
    signal = SphericalHarmonicSignal(
        np.zeros((4, 16)), 48000, 'real', 'N3D', 'ACN', True)

    with pytest.raises(ValueError, match='Condon-Shortley'):
        parametric.PseudoIntensityVectorAnalyzer()(signal)


def test_error_if_more_than_one_signal_is_given():
    with pytest.raises(ValueError, match='a single spherical harmonic signal'):
        parametric.PseudoIntensityVectorAnalyzer()(
            sh_signal(np.zeros((2, 4, 16))))


def test_input_signal_is_not_modified(impulse_response):
    converted = impulse_response.copy()
    converted.normalization = 'SN3D'
    before = np.asarray(converted.time).copy()

    parametric.PseudoIntensityVectorAnalyzer()(converted)

    assert converted.normalization == 'SN3D'
    np.testing.assert_array_equal(np.asarray(converted.time), before)
