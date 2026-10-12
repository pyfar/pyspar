"""Tests of the spectral equalization."""
import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
from conftest import (
    LAYOUT_LFE_ONE, LFE_ONE, layout, sh_signal, third_octave_bands)
from scipy.signal import butter, lfilter, sosfiltfilt

from pyspar import parametric
import pyspar

MODES = ('minphase', 'linphase')
BLOCK = 2048


def make_result(pressure, sampling_rate=48000):
    """Analysis result of a pressure arriving from the front."""
    pressure = np.asarray(pressure, dtype=float)
    front = pf.Coordinates(np.ones(len(pressure)), np.zeros(len(pressure)),
                           np.zeros(len(pressure)))
    return parametric.TimeResult(
        pressure=pf.Signal(pressure, sampling_rate),
        signal=pf.Signal(pressure, sampling_rate),
        doa=front)


def equalize(feeds, pressure, **kwargs):
    """Equalize ``(n_channels, n_samples)`` feeds towards `pressure`."""
    return parametric.SpectralEqualization(**kwargs)(
        pf.Signal(feeds, 48000), make_result(pressure)).time


@pytest.fixture
def noise():
    """One block of seeded noise."""
    return np.random.default_rng(3).standard_normal(BLOCK)


def band_levels(x, n_fft=8192):
    """Level in dB of three two-octave bands from 125 Hz to 8 kHz."""
    frequencies = np.fft.rfftfreq(n_fft, 1 / 48000)
    energy = np.abs(np.fft.rfft(x, n_fft)) ** 2
    return 10 * np.log10([
        np.sum(energy[(frequencies >= low) & (frequencies < 4 * low)])
        for low in (125, 500, 2000)])


# -- properties ---------------------------------------------------------------

def test_stage_is_loudspeaker():
    assert parametric.SpectralEqualization.stage == 'loudspeaker'


def test_defaults_are_minimum_phase_and_20_db():
    equalizer = parametric.SpectralEqualization()

    assert equalizer.mode == 'minphase'
    assert equalizer.max_boost_db == 20


@pytest.mark.parametrize('n_samples', [1000, BLOCK, 5000])
def test_adds_the_tail_of_the_filters(n_samples):
    # Filters of 1024 taps lengthen the last block by 1023 samples.
    rng = np.random.default_rng(0)
    signal = pf.Signal(rng.standard_normal((3, n_samples)), 48000)
    result = make_result(rng.standard_normal(n_samples))

    equalized = parametric.SpectralEqualization()(signal, result)

    assert equalized.cshape == (3,)
    assert equalized.n_samples == n_samples + BLOCK // 2 - 1
    assert equalized.sampling_rate == 48000
    assert np.any(equalized.time[:, -BLOCK // 4:] != 0)


def band_energies(x, sampling_rate=48000):
    """Energy per third-octave band above 500 Hz."""
    bands, centers = third_octave_bands(x, sampling_rate)
    return np.sum(bands[centers > 500] ** 2, axis=-1)


@pytest.mark.parametrize('mode', MODES)
def test_power_sum_of_the_loudspeakers_follows_the_pressure(
        mode, layout_horizontal):
    # A pressure without high frequencies, arriving from a new direction at
    # every sample.  Distributing its samples over the loudspeakers whitens
    # each loudspeaker signal, so their energies add up to far more
    # high-frequency energy than the pressure has; the equalization brings
    # the sum back to the pressure.
    rng = np.random.default_rng(8)
    n_samples = 12000
    pressure = sosfiltfilt(butter(8, 1000, fs=48000, output='sos'),
                           rng.standard_normal(n_samples)) * np.exp(
                               -np.arange(n_samples) / 4000)
    result = parametric.TimeResult(
        pressure=pf.Signal(pressure, 48000),
        signal=pf.Signal(pressure, 48000),
        doa=pf.Coordinates.from_spherical_elevation(
            rng.uniform(-np.pi, np.pi, n_samples),
            np.arcsin(rng.uniform(-1, 1, n_samples)), 1))
    feeds = parametric.LoudspeakerSynthesizer(layout_horizontal)(result)

    equalized = parametric.SpectralEqualization(mode)(feeds, result)

    def deviation(signals):
        power_sum = np.sum([band_energies(channel[:n_samples])
                            for channel in signals], axis=0)
        return np.sqrt(np.mean(
            (10 * np.log10(power_sum / band_energies(pressure))) ** 2))

    assert deviation(feeds.time) > 8
    assert deviation(equalized.time) < deviation(feeds.time) / 2


@pytest.mark.parametrize('mode', MODES)
def test_keeps_the_energy_of_a_block(noise, mode):
    feeds = np.random.default_rng(4).standard_normal((2, BLOCK)) * [[1], [5]]

    equalized = equalize(feeds, noise, mode=mode)

    npt.assert_allclose(np.sum(equalized ** 2, axis=-1),
                        np.sum(feeds ** 2, axis=-1), rtol=1e-12)


@pytest.mark.parametrize('mode', MODES)
def test_does_not_depend_on_the_level(noise, mode):
    feeds = np.random.default_rng(4).standard_normal((2, BLOCK))

    quiet = equalize(feeds * 1e-4, noise * 1e-4, mode=mode)

    npt.assert_allclose(quiet / 1e-4, equalize(feeds, noise, mode=mode),
                        rtol=0, atol=1e-10)


@pytest.mark.parametrize(('mode', 'delay'), [('minphase', 0),
                                             ('linphase', 512)])
def test_signal_with_the_spectrum_of_the_pressure_passes(noise, mode, delay):
    # Without regularization the filter is a unit impulse, delayed by a
    # quarter block for linear phase.
    equalized = equalize(noise, noise, mode=mode, max_boost_db=300)[0]

    expected = np.zeros(BLOCK + BLOCK // 2 - 1)
    expected[delay:delay + BLOCK] = noise
    npt.assert_allclose(equalized, expected, atol=1e-12)


@pytest.mark.parametrize('mode', MODES)
def test_restores_the_spectrum_of_a_colored_loudspeaker_signal(noise, mode):
    colored = lfilter([1, -0.9], 1, noise)
    target = band_levels(noise)

    def deviation(x):
        difference = band_levels(x) - target
        return np.max(np.abs(difference - np.mean(difference)))

    equalized = equalize(colored, noise, mode=mode)[0]

    assert deviation(colored) > 6
    assert deviation(equalized) < 3


def test_stronger_regularization_inverts_a_notch_less(noise):
    # Adding a copy delayed by 24 samples cuts a notch at 1 kHz.
    notched = noise + np.concatenate([np.zeros(24), noise[:-24]])
    frequencies = np.fft.rfftfreq(8192, 1 / 48000)
    notch = (frequencies > 950) & (frequencies < 1050)

    energy = [
        np.sum(np.abs(np.fft.rfft(
            equalize(notched, noise, max_boost_db=max_boost_db)[0],
            8192))[notch] ** 2)
        for max_boost_db in (0, 20, 300)]

    assert energy[0] < energy[1] < energy[2]


@pytest.mark.parametrize('mode', MODES)
def test_silent_loudspeaker_stays_silent(noise, mode):
    feeds = np.stack([noise, np.zeros(BLOCK)])

    equalized = equalize(feeds, noise, mode=mode)

    assert np.all(np.isfinite(equalized))
    npt.assert_array_equal(equalized[1], 0.0)


@pytest.mark.parametrize('mode', MODES)
def test_block_passes_unfiltered_where_the_pressure_is_silent(mode):
    rng = np.random.default_rng(5)
    feeds = rng.standard_normal(2 * BLOCK)
    pressure = rng.standard_normal(2 * BLOCK)
    pressure[:BLOCK] = 0

    equalized = equalize(feeds, pressure, mode=mode)[0]

    assert np.all(np.isfinite(equalized))
    npt.assert_allclose(equalized[:BLOCK], feeds[:BLOCK], atol=1e-10)


def test_smoothing_keeps_a_flat_spectrum_and_the_end_values():
    frequencies = np.arange(1025) * 48000 / 2048
    spectra = np.full((1025, 2), 3.0)
    spectra[[0, -1], 1] = [7.0, 0.5]

    smoothed = pyspar.utils.dsp._smooth_log_frequency(
        spectra, frequencies, 3)

    npt.assert_allclose(smoothed[:, 0], 3.0, atol=1e-12)
    npt.assert_allclose(smoothed[[0, -1], 1], [7.0, 0.5])


def test_smoothing_spreads_a_peak_over_a_third_octave():
    frequencies = np.arange(1025) * 48000 / 2048
    spectra = np.ones((1025, 1))
    spectra[128] = 100.0                               # at 3 kHz

    smoothed = pyspar.utils.dsp._smooth_log_frequency(
        spectra, frequencies, 3)[:, 0]

    assert 1 < smoothed[128] < 100
    inside = (frequencies > 3000 / 2 ** (1 / 6)) & (
        frequencies < 3000 * 2 ** (1 / 6))
    assert np.all(smoothed[inside] > 1.01)
    npt.assert_allclose(smoothed[frequencies > 6000], 1.0, atol=1e-3)
    npt.assert_allclose(smoothed[(frequencies > 0) & (frequencies < 1500)],
                        1.0, atol=1e-3)


# -- errors -------------------------------------------------------------------

def test_error_if_mode_is_unknown():
    with pytest.raises(ValueError, match="mode is 'minPhase'"):
        parametric.SpectralEqualization('minPhase')


@pytest.mark.parametrize('max_boost_db', [np.inf, np.nan])
def test_error_if_max_boost_is_not_finite(max_boost_db):
    with pytest.raises(ValueError, match='max_boost_db is'):
        parametric.SpectralEqualization(max_boost_db=max_boost_db)


@pytest.mark.parametrize('max_boost_db', [True, '20', [20]])
def test_error_if_max_boost_is_not_a_number(max_boost_db):
    with pytest.raises(TypeError, match='max_boost_db must be a number'):
        parametric.SpectralEqualization(max_boost_db=max_boost_db)


@pytest.mark.parametrize('signal', [
    np.zeros((2, 100)), sh_signal(np.zeros((4, 100)))],
    ids=['array', 'spherical-harmonic-signal'])
def test_error_if_signal_is_not_a_pyfar_signal(signal):
    with pytest.raises(TypeError, match='requires a pyfar.Signal'):
        parametric.SpectralEqualization()(signal, make_result(np.ones(100)))


def test_error_if_the_analysis_result_is_missing(noise):
    with pytest.raises(TypeError,
                       match='requires the pyspar.parametric.TimeResult'):
        parametric.SpectralEqualization()(pf.Signal(noise, 48000))


def test_error_if_signal_and_pressure_differ_in_length(noise):
    with pytest.raises(ValueError, match='2048 samples but the pressure'):
        parametric.SpectralEqualization()(pf.Signal(noise, 48000),
                                          make_result(np.ones(100)))


# -- in the synthesizers ------------------------------------------------------

def test_synthesizers_apply_none_by_default(layout_surround, hrir_set):
    assert parametric.LoudspeakerSynthesizer(
        layout_surround).correction is None
    assert parametric.BinauralSynthesizer(*hrir_set).correction is None


@pytest.mark.parametrize('mode', MODES)
def test_mode_string_gives_the_default_equalizer(layout_surround, mode):
    correction = parametric.LoudspeakerSynthesizer(
        layout_surround, correction=mode).correction

    assert isinstance(correction, parametric.SpectralEqualization)
    assert (correction.mode, correction.max_boost_db) == (mode, 20)


def test_equalizer_instance_is_kept(layout_surround):
    equalizer = parametric.SpectralEqualization('linphase', max_boost_db=40)

    synthesizer = parametric.LoudspeakerSynthesizer(layout_surround,
                                                    correction=equalizer)

    assert synthesizer.correction is equalizer


def test_error_if_correction_string_is_unknown(layout_surround):
    with pytest.raises(ValueError, match="correction is 'flat' but must"):
        parametric.LoudspeakerSynthesizer(layout_surround, correction='flat')


def test_error_if_correction_has_the_wrong_type(layout_surround):
    with pytest.raises(TypeError, match='correction must be'):
        parametric.LoudspeakerSynthesizer(layout_surround, correction=20)


def test_binaural_render_equalizes_before_the_hrirs(analysis, hrir_set):
    equalizer = parametric.SpectralEqualization()
    plain = parametric.BinauralSynthesizer(*hrir_set)
    expected = plain.convolve(
        equalizer(plain.render_loudspeakers(analysis), result=analysis))

    synthesizer = parametric.BinauralSynthesizer(*hrir_set,
                                                 correction=equalizer)

    assert synthesizer.correction is equalizer
    npt.assert_array_equal(synthesizer(analysis).time, expected.time)


@pytest.mark.parametrize('mode', ['minphase', 'linphase'])
def test_lfe_crossover_runs_after_the_equalization(noise_between_silences,
                                                   mode):
    # Equalized before the crossover, the LFE channel keeps to the low
    # frequencies and the other loudspeakers to the high ones.  Equalized
    # after it, each would be given the spectrum of the pressure again.
    rendered = parametric.LoudspeakerSynthesizer(
        layout(*LAYOUT_LFE_ONE.T), lfe_channels=LFE_ONE,
        correction=mode)(noise_between_silences)

    spectra = np.abs(np.fft.rfft(rendered.time, axis=-1))
    frequencies = np.fft.rfftfreq(rendered.n_samples, 1 / 48000)
    lfe, others = spectra[LFE_ONE[0]], np.delete(spectra, LFE_ONE, axis=0)
    assert np.max(lfe[frequencies > 2000]) < 1e-4 * np.max(lfe)
    assert np.max(others[:, frequencies < 10]) < 1e-2 * np.max(others)
