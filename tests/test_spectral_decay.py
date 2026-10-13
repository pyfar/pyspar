"""Tests of the spectral decay correction of the Ambisonic encoder."""
import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
from conftest import encode, make_dense_response, sh_signal
from scipy.signal import butter, sosfiltfilt
from spharpy import SphericalHarmonicSignal

from pyspar import parametric
from pyspar.parametric.postprocessors import _spectral_decay_correction


def analyze(signal, sampling_rate=48000):
    """Analysis with a 4-sample window of `signal` at `sampling_rate`."""
    signal = sh_signal(np.asarray(signal.time)[0], sampling_rate)
    return parametric.PseudoIntensityVectorAnalyzer(
        window_length=5)(signal)


def encoded(result, order, **kwargs):
    """Ambisonic time data as ``(n_samples, n_coefficients)``."""
    signal = parametric.AmbisonicSynthesizer(order, **kwargs)(result)
    return np.asarray(signal.time)[0].T


def degree_of(n_coefficients):
    """Spherical harmonic degree of each ACN channel."""
    return np.floor(np.sqrt(np.arange(n_coefficients))).astype(int)


def decaying_noise(n_samples, decay_samples, seed):
    """Seeded noise with an exponential envelope."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_samples) * np.exp(
        -np.arange(n_samples) / decay_samples)


def make_low_passed_field():
    """First-order diffuse field without high frequencies, 0.25 s.

    24 plane waves from scattered directions, each carrying its own decaying
    noise low-passed at 1 kHz.
    """
    rng = np.random.default_rng(7)
    n_samples = 12000
    sos = butter(8, 1000, fs=48000, output='sos')
    data = np.zeros((4, n_samples))
    for _ in range(24):
        noise = sosfiltfilt(sos, rng.standard_normal(n_samples)) * np.exp(
            -np.arange(n_samples) / 3000)
        direction = pf.Coordinates.from_spherical_elevation(
            rng.uniform(-np.pi, np.pi), np.arcsin(rng.uniform(-1, 1)), 1)
        data += np.asarray(encode(noise, direction).time)[0]
    return sh_signal(data)


# -- filter bank --------------------------------------------------------------

def test_bands_add_up_to_the_signal_in_any_transform_length():
    # All four channels and the reference carry the same signal, so the
    # gains are one and the correction returns the bands added up: the
    # signal, for lengths that are and are not a power of two.
    for n_samples in (2048, 3000, 8192):
        signal = np.tile(decaying_noise(n_samples, n_samples / 4, 1)[:, None],
                         (1, 4))

        corrected = _spectral_decay_correction(signal, signal, 48000)

        # Half a window before the transform length the circular smoothing
        # scales one sample to about zero, see the Notes of the class.
        keep = np.arange(n_samples) != 2 ** int(np.ceil(
            np.log2(n_samples))) - 512
        npt.assert_allclose(corrected[keep], signal[keep],
                            atol=2e-3 * np.max(np.abs(signal)))


# -- reference energy and gains -----------------------------------------------

def test_reference_is_half_the_omni_and_half_the_mean_dipole_energy():
    # The gains are proportional to the root of the reference energy
    # E_w / 2 + (E_x + E_y + E_z) / 6.  With all four channels equal it is
    # E_w, without the dipoles E_w / 2, and with one dipole 2 / 3 E_w.
    signal = np.column_stack(
        [decaying_noise(3000, 900, seed) for seed in range(9)])
    omni = decaying_noise(3000, 900, 20)
    silent = np.zeros(3000)

    equal = _spectral_decay_correction(
        signal, np.column_stack([omni, omni, omni, omni]), 48000)
    omni_only = _spectral_decay_correction(
        signal, np.column_stack([omni, silent, silent, silent]), 48000)
    one_dipole = _spectral_decay_correction(
        signal, np.column_stack([omni, silent, omni, silent]), 48000)

    npt.assert_allclose(omni_only, np.sqrt(1 / 2) * equal, atol=1e-14)
    npt.assert_allclose(one_dipole, np.sqrt(2 / 3) * equal, atol=1e-14)


def test_level_of_each_order_is_set_by_the_reference():
    # Scaling the orders of the input does not change the output, apart
    # from the regularization of the gains.
    signal = np.column_stack(
        [decaying_noise(3000, 900, seed) for seed in range(9)])
    reference = np.column_stack(
        [decaying_noise(3000, 900, seed) for seed in range(20, 24)])
    scale = np.array([1, 5, 0.2])[degree_of(9)]

    corrected = _spectral_decay_correction(signal, reference, 48000)
    rescaled = _spectral_decay_correction(signal * scale, reference, 48000)

    npt.assert_allclose(rescaled, corrected, rtol=0,
                        atol=1e-5 * np.max(np.abs(corrected)))


def test_each_order_carries_2n_plus_1_times_the_reference_energy():
    signal = make_low_passed_field()
    first_order = np.asarray(signal.time)[0]
    reference = (np.sum(first_order[0] ** 2) / 2
                 + np.sum(first_order[1:] ** 2) / 6)

    corrected = encoded(analyze(signal), 3)

    degree = degree_of(16)
    energy = [np.sum(corrected[:, degree == n] ** 2) for n in range(4)]
    npt.assert_allclose(energy, np.array([1, 3, 5, 7]) * reference,
                        rtol=0.05)


def test_removes_the_high_frequencies_added_to_the_higher_orders():
    # Encoding a field without high frequencies sample by sample adds them
    # to the higher orders; the correction takes them out again.
    signal = make_low_passed_field()
    result = analyze(signal)
    degree = degree_of(16)

    def high_frequency_share(data):
        """Energy above 4 kHz relative to all of it, in dB."""
        energy = np.abs(np.fft.rfft(data, axis=0)) ** 2
        above = np.fft.rfftfreq(len(data), 1 / 48000) > 4000
        return 10 * np.log10(np.sum(energy[above]) / np.sum(energy))

    measured = high_frequency_share(np.asarray(signal.time)[0].T)
    plain = encoded(result, 3, spectral_decay_correction=False)
    corrected = encoded(result, 3)

    for n in (2, 3):
        assert high_frequency_share(plain[:, degree == n]) > measured + 10
        assert abs(high_frequency_share(corrected[:, degree == n])
                   - measured) < 3


def test_sound_from_a_single_direction_is_left_unchanged():
    # All channels are then copies of the pressure, whose energy is the
    # reference and 1 / (2n + 1) of that of each order.
    direction = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(40), np.deg2rad(-20), 1)
    result = analyze(encode(decaying_noise(6000, 1500, 5), direction))

    plain = encoded(result, 3, spectral_decay_correction=False)
    corrected = encoded(result, 3)

    npt.assert_allclose(corrected, plain, rtol=0,
                        atol=1e-5 * np.max(np.abs(plain)))


# -- encoder ------------------------------------------------------------------

def test_omnidirectional_channel_is_corrected_too(dense_analysis):
    corrected = encoded(dense_analysis, 2)

    assert not np.allclose(corrected[:, 0], dense_analysis.pressure.time[0],
                           rtol=1e-3)


def test_pressure_is_the_reference_without_a_first_order_signal(
        dense_analysis):
    # The omnidirectional channel then keeps its envelope, up to the
    # regularization of the gains; the higher orders are corrected.
    pressure = dense_analysis.pressure.time[0]
    result = parametric.TimeResult(
        pressure=pf.Signal(pressure, 48000), doa=dense_analysis.doa,
        signal=pf.Signal(pressure, 48000))

    plain = encoded(result, 2, spectral_decay_correction=False)
    corrected = encoded(result, 2)

    peak = np.max(np.abs(pressure))
    npt.assert_allclose(corrected[:, 0], pressure, rtol=0, atol=1e-5 * peak)
    assert np.max(np.abs(corrected[:, 1:] - plain[:, 1:])) > 1e-2 * peak


@pytest.mark.parametrize(('normalization', 'channel_convention'), [
    ('SN3D', 'ACN'), ('maxN', 'FuMa')])
def test_reference_is_read_in_n3d(
        dense_analysis, normalization, channel_convention):
    signal = dense_analysis.signal.copy()
    signal.channel_convention = channel_convention
    signal.normalization = normalization
    result = parametric.TimeResult(
        pressure=dense_analysis.pressure, doa=dense_analysis.doa,
        signal=signal)

    npt.assert_array_equal(encoded(result, 2), encoded(dense_analysis, 2))


def test_window_has_32_samples_below_24_khz():
    # The rule round(fs / 48000) * 1024 would give no samples.  With 32
    # samples and a response of 32, each sample is weighted by the energy of
    # the whole response, so equal references give equal gains.
    signal = np.column_stack(
        [decaying_noise(32, 10, seed) for seed in range(4)])
    reference = np.column_stack(
        [decaying_noise(32, 10, seed) for seed in range(20, 24)])

    corrected = _spectral_decay_correction(signal, reference, 16000)

    assert corrected.shape == (32, 4)
    assert np.all(np.isfinite(corrected))
    npt.assert_allclose(
        _spectral_decay_correction(signal, 2 * reference, 16000),
        2 * corrected, atol=1e-14)


# -- parameters and errors ----------------------------------------------------

def test_correction_accepts_numpy_booleans():
    synthesizer = parametric.AmbisonicSynthesizer(
        2, spectral_decay_correction=np.bool_(False))

    assert synthesizer.spectral_decay_correction is False


@pytest.mark.parametrize('value', [1, 0, 'yes', None])
def test_error_if_the_correction_is_not_a_boolean(value):
    with pytest.raises(TypeError, match='must be True or False'):
        parametric.AmbisonicSynthesizer(2, spectral_decay_correction=value)


# -- SpectralDecayCorrection --------------------------------------------------

@pytest.mark.parametrize('normalization', ['N3D', 'SN3D'])
def test_correction_equals_that_of_the_synthesizer(normalization):
    result = parametric.PseudoIntensityVectorAnalyzer(window_length=5)(
        make_dense_response())
    corrected = parametric.AmbisonicSynthesizer(
        3, normalization=normalization)(result)
    upmixed = parametric.AmbisonicSynthesizer(
        3, normalization=normalization,
        spectral_decay_correction=False)(result)

    npt.assert_array_equal(
        parametric.SpectralDecayCorrection()(upmixed, result).time,
        corrected.time)


def test_correction_error_if_the_signal_is_not_ambisonic():
    result = parametric.PseudoIntensityVectorAnalyzer(None)(
        make_dense_response())
    with pytest.raises(TypeError, match='SpectralDecayCorrection requires '
                                        'a spharpy.SphericalHarmonicSignal'):
        parametric.SpectralDecayCorrection()(
            pf.Signal(np.zeros((2, 3000)), 48000), result)


def test_correction_error_if_the_result_is_not_an_analysis():
    upmixed = parametric.AmbisonicSynthesizer(
        1, spectral_decay_correction=False)(
            parametric.PseudoIntensityVectorAnalyzer(None)(
                make_dense_response()))
    with pytest.raises(TypeError, match='requires the pyspar.parametric'):
        parametric.SpectralDecayCorrection()(upmixed, None)


def test_correction_keeps_the_conventions_of_the_upmix():
    result = parametric.PseudoIntensityVectorAnalyzer(window_length=5)(
        make_dense_response())
    upmixed = parametric.AmbisonicSynthesizer(
        3, spectral_decay_correction=False)(result)
    expected = parametric.SpectralDecayCorrection()(upmixed, result)
    for signal in (upmixed, expected):
        signal.channel_convention = 'FuMa'
        signal.normalization = 'SN3D'

    corrected = parametric.SpectralDecayCorrection()(upmixed, result)

    assert corrected.channel_convention == 'FuMa'
    assert corrected.normalization == 'SN3D'
    npt.assert_allclose(corrected.time, expected.time, rtol=0,
                        atol=1e-12 * np.max(np.abs(expected.time)))


@pytest.mark.parametrize(('signal', 'message'), [
    (SphericalHarmonicSignal(
        np.zeros((4, 3000), dtype=complex), 48000, 'complex', 'N3D', 'ACN',
        False, is_complex=True),
     'SpectralDecayCorrection requires a real signal without the '
     'Condon-Shortley phase'),
    (SphericalHarmonicSignal(
        np.zeros((2, 4, 3000)), 48000, 'real', 'N3D', 'ACN', False),
     'SpectralDecayCorrection expects a single spherical harmonic signal')])
def test_correction_error_for_a_signal_it_cannot_read(signal, message):
    result = parametric.PseudoIntensityVectorAnalyzer(None)(
        make_dense_response())
    with pytest.raises(ValueError, match=message):
        parametric.SpectralDecayCorrection()(signal, result)


def test_correction_error_if_the_sampling_rates_disagree():
    result = parametric.PseudoIntensityVectorAnalyzer(None)(
        make_dense_response())
    signal = SphericalHarmonicSignal(np.zeros((4, 3000)), 44100, 'real',
                                     'N3D', 'ACN', False)
    with pytest.raises(ValueError, match='does not match the result'):
        parametric.SpectralDecayCorrection()(signal, result)


def test_correction_error_if_the_lengths_disagree():
    result = parametric.PseudoIntensityVectorAnalyzer(None)(
        make_dense_response())
    signal = SphericalHarmonicSignal(np.zeros((4, 100)), 48000, 'real',
                                     'N3D', 'ACN', False)
    with pytest.raises(ValueError, match='samples but the analysis'):
        parametric.SpectralDecayCorrection()(signal, result)
