"""Tests of the post equalization of the loudspeaker spectra."""
import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
from conftest import (
    LAYOUT_LFE_ONE, LFE_ONE, layout, sh_signal, third_octave_bands)
from scipy.signal import butter, sosfiltfilt

from pyspar import parametric
from pyspar.parametric.synthesizers import _lfe_crossover


def make_result(pressure, doa=None, sampling_rate=48000):
    """Analysis result of a pressure, arriving from the front by default."""
    pressure = np.asarray(pressure, dtype=float)
    if doa is None:
        doa = pf.Coordinates(np.ones(len(pressure)), np.zeros(len(pressure)),
                             np.zeros(len(pressure)))
    return parametric.TimeResult(
        pressure=pf.Signal(pressure, sampling_rate),
        signal=pf.Signal(pressure, sampling_rate),
        doa=doa)


def equalize(feeds, pressure, **kwargs):
    """Equalize ``(n_channels, n_samples)`` feeds towards `pressure`."""
    return parametric.LoudspeakerSpectrumEqualization(**kwargs)(
        pf.Signal(feeds, 48000), make_result(pressure)).time


def band_energies(x, sampling_rate=48000):
    """Energy per third-octave band above 500 Hz."""
    bands, centers = third_octave_bands(x, sampling_rate)
    return np.sum(bands[centers > 500] ** 2, axis=-1)


def make_scattered_result(seed=8, n_samples=12000):
    """Decaying noise without high frequencies, scattered sample by sample.

    Every sample arrives from a new random direction.
    """
    rng = np.random.default_rng(seed)
    pressure = sosfiltfilt(butter(8, 1000, fs=48000, output='sos'),
                           rng.standard_normal(n_samples)) * np.exp(
                               -np.arange(n_samples) / 4000)
    doa = pf.Coordinates.from_spherical_elevation(
        rng.uniform(-np.pi, np.pi, n_samples),
        np.arcsin(rng.uniform(-1, 1, n_samples)), 1)
    return make_result(pressure, doa)


@pytest.fixture
def noise():
    """Decaying seeded noise, 12000 samples."""
    return (np.random.default_rng(3).standard_normal(12000)
            * np.exp(-np.arange(12000) / 4000))


# -- properties ---------------------------------------------------------------

def test_stage_is_loudspeaker():
    assert parametric.LoudspeakerSpectrumEqualization.stage == 'loudspeaker'


def test_default_parameters():
    equalizer = parametric.LoudspeakerSpectrumEqualization()

    assert equalizer.periods == 7
    assert equalizer.max_boost_db == 6.0


def test_keeps_shape_and_length(noise):
    feeds = np.stack([noise, noise[::-1], np.zeros_like(noise)])

    equalized = parametric.LoudspeakerSpectrumEqualization()(
        pf.Signal(feeds, 48000), make_result(noise))

    assert equalized.cshape == (3,)
    assert equalized.n_samples == 12000
    assert equalized.sampling_rate == 48000


def test_power_sum_of_the_loudspeakers_follows_the_pressure(
        layout_horizontal):
    # Distributing a pressure without high frequencies over the loudspeakers
    # whitens each loudspeaker signal, so their energies add up to far more
    # high-frequency energy than the pressure has; the equalization brings
    # the sum back to the pressure.
    result = make_scattered_result()
    pressure = result.pressure.time[0]
    feeds = parametric.LoudspeakerSynthesizer(layout_horizontal)(result)

    equalized = parametric.LoudspeakerSpectrumEqualization()(feeds, result)

    def deviation(signals):
        power_sum = np.sum([band_energies(channel) for channel in signals],
                           axis=0)
        return np.sqrt(np.mean(
            (10 * np.log10(power_sum / band_energies(pressure))) ** 2))

    assert deviation(feeds.time) > 8
    assert deviation(equalized.time) < deviation(feeds.time) / 3


def test_one_loudspeaker_carrying_the_pressure_passes(noise):
    # The sum of the loudspeaker spectra is the pressure spectrum already,
    # so the gains are one and only the band filters act.
    equalized = equalize(np.stack([noise, np.zeros_like(noise)]), noise)

    npt.assert_allclose(equalized[0], noise, atol=1e-3 * np.max(np.abs(noise)))
    npt.assert_array_equal(equalized[1], 0.0)


def test_silent_loudspeaker_stays_silent(noise):
    equalized = equalize(np.stack([noise, np.zeros_like(noise), noise[::-1]]),
                         noise)

    npt.assert_array_equal(equalized[1], 0.0)


def test_does_not_depend_on_the_level(noise):
    feeds = np.stack([noise, noise[::-1]])

    quiet = equalize(feeds * 1e-4, noise * 1e-4)

    npt.assert_allclose(quiet / 1e-4, equalize(feeds, noise), rtol=0,
                        atol=1e-10)


def test_boost_is_limited_softly(noise):
    # A loudspeaker signal 20 dB below the pressure asks for a gain of 10,
    # which the soft limit compresses to the limit, 2 for 6 dB.
    quiet = 0.1 * noise

    limited = equalize(quiet[None], noise)[0]
    unlimited = equalize(quiet[None], noise, max_boost_db=np.inf)[0]
    none = equalize(quiet[None], noise, max_boost_db=0)[0]

    peak = np.max(np.abs(noise))
    npt.assert_allclose(limited, 2 * quiet, atol=1e-3 * peak)
    npt.assert_allclose(unlimited, noise, atol=1e-3 * peak)
    npt.assert_allclose(none, quiet, atol=1e-3 * peak)


def test_notch_of_the_sum_is_inverted_without_a_limit(noise):
    # The gain follows the ratio of the frame spectra bin by bin, so a
    # 20 dB notch of the loudspeaker signal is filled from the pressure.
    sos = butter(4, [2000, 4000], 'bandstop', fs=48000, output='sos')
    notched = noise - 0.99 * (noise - sosfiltfilt(sos, noise))
    _, centers = third_octave_bands(notched)
    inside = (centers > 2400) & (centers < 3300)

    def notch_energy(x):
        return np.sum(third_octave_bands(x)[0][inside] ** 2)

    unlimited = equalize(notched[None], noise, max_boost_db=np.inf)[0]

    depth = 10 * np.log10(notch_energy(notched) / notch_energy(noise))
    filled = 10 * np.log10(notch_energy(unlimited) / notch_energy(noise))
    assert depth < -15
    assert abs(filled) < 5


def test_frames_follow_the_periods(noise):
    # Longer frames change the result but keep its length.
    feeds = np.stack([noise, noise[::-1]])

    short = equalize(feeds, noise, periods=3)
    long = equalize(feeds, noise, periods=14)

    assert short.shape == long.shape == feeds.shape
    assert np.max(np.abs(short - long)) > 1e-3 * np.max(np.abs(noise))


# -- errors -------------------------------------------------------------------

@pytest.mark.parametrize('periods', [0, -3])
def test_error_if_periods_is_not_positive(periods):
    with pytest.raises(ValueError, match='periods is .* but must be'):
        parametric.LoudspeakerSpectrumEqualization(periods=periods)


@pytest.mark.parametrize('periods', [7.5, True, '7'])
def test_error_if_periods_is_not_an_integer(periods):
    with pytest.raises(TypeError, match='periods must be an integer'):
        parametric.LoudspeakerSpectrumEqualization(periods=periods)


@pytest.mark.parametrize('max_boost_db', [-1.0, np.nan])
def test_error_if_max_boost_is_negative_or_nan(max_boost_db):
    with pytest.raises(ValueError, match='max_boost_db is .* but must be'):
        parametric.LoudspeakerSpectrumEqualization(max_boost_db=max_boost_db)


@pytest.mark.parametrize('max_boost_db', [True, '6', [6.0]])
def test_error_if_max_boost_is_not_a_number(max_boost_db):
    with pytest.raises(TypeError, match='max_boost_db must be a number'):
        parametric.LoudspeakerSpectrumEqualization(max_boost_db=max_boost_db)


@pytest.mark.parametrize('signal', [
    np.zeros((2, 100)), sh_signal(np.zeros((4, 100)))],
    ids=['array', 'spherical-harmonic-signal'])
def test_error_if_signal_is_not_a_pyfar_signal(signal):
    with pytest.raises(TypeError, match='requires a pyfar.Signal'):
        parametric.LoudspeakerSpectrumEqualization()(
            signal, make_result(np.ones(100)))


def test_error_if_the_analysis_result_is_missing(noise):
    with pytest.raises(TypeError,
                       match='requires the pyspar.parametric.TimeResult'):
        parametric.LoudspeakerSpectrumEqualization()(
            pf.Signal(noise[None], 48000))


def test_error_if_signal_and_pressure_differ_in_length(noise):
    with pytest.raises(ValueError, match='12000 samples but the pressure'):
        parametric.LoudspeakerSpectrumEqualization()(
            pf.Signal(noise[None], 48000), make_result(np.ones(100)))


def test_error_if_sampling_rates_differ(noise):
    with pytest.raises(ValueError, match='does not match the result'):
        parametric.LoudspeakerSpectrumEqualization()(
            pf.Signal(noise[None], 44100), make_result(noise))


# -- in the synthesizers ------------------------------------------------------

def test_equalizer_instance_is_kept(layout_surround, hrir_set):
    equalizer = parametric.LoudspeakerSpectrumEqualization()

    assert parametric.LoudspeakerSynthesizer(
        layout_surround, correction=equalizer).correction is equalizer
    assert parametric.BinauralSynthesizer(
        *hrir_set, correction=equalizer).correction is equalizer


def test_binaural_render_equalizes_before_the_hrirs(dense_analysis, hrir_set):
    equalizer = parametric.LoudspeakerSpectrumEqualization()
    plain = parametric.BinauralSynthesizer(*hrir_set)
    expected = plain.convolve(equalizer(
        plain.render_loudspeakers(dense_analysis), result=dense_analysis))

    rendered = parametric.BinauralSynthesizer(
        *hrir_set, correction=equalizer)(dense_analysis)

    npt.assert_array_equal(rendered.time, expected.time)


def test_lfe_crossover_runs_after_the_equalization(noise_between_silences):
    loudspeakers = layout(*LAYOUT_LFE_ONE.T)
    equalizer = parametric.LoudspeakerSpectrumEqualization()
    others = np.delete(LAYOUT_LFE_ONE, LFE_ONE, axis=0)
    panned = np.insert(
        parametric.LoudspeakerSynthesizer(layout(*others.T))(
            noise_between_silences).time, LFE_ONE[0], 0.0, axis=0)
    expected = _lfe_crossover(
        equalizer(pf.Signal(panned, 48000), noise_between_silences),
        loudspeakers, LFE_ONE, 80.0)

    rendered = parametric.LoudspeakerSynthesizer(
        loudspeakers, lfe_channels=LFE_ONE, correction=equalizer)(
            noise_between_silences)

    npt.assert_array_equal(rendered.time, expected.time)
