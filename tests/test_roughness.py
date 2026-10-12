"""Tests of the roughness compensation."""
import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
from conftest import (
    LAYOUT_LFE_ONE, LFE_ONE, ROUGHNESS_SEED, impulse_response_start, layout,
    make_mixing_response, sh_signal)

from pyspar import parametric
from pyspar.parametric.synthesizers import _lfe_crossover
from pyspar.utils import room_acoustics
import pyspar

MODES = ('schroeder', 'velvet')


def impulses(n_samples, positions, n_channels=1, sampling_rate=48000):
    """Signal with unit impulses at ``[(channel, sample), ...]``."""
    data = np.zeros((n_channels, n_samples))
    for channel, sample in positions:
        data[channel, sample] = 1.0
    return pf.Signal(data, sampling_rate)


def make_result(pressure, sampling_rate=48000):
    """Analysis result of a pressure arriving from the front."""
    pressure = np.asarray(pressure, dtype=float)
    front = pf.Coordinates(np.ones(len(pressure)), np.zeros(len(pressure)),
                           np.zeros(len(pressure)))
    return parametric.TimeResult(
        pressure=pf.Signal(pressure, sampling_rate),
        signal=pf.Signal(pressure, sampling_rate),
        doa=front)


def whole(n_samples, sampling_rate=48000):
    """Result whose onset is the first sample: ``start='onset'`` covers all."""
    pressure = np.zeros(n_samples)
    pressure[0] = 1.0
    return make_result(pressure, sampling_rate)


@pytest.fixture
def mixing():
    """Pressure with a mixing time, its result and its late-part start."""
    pressure = make_mixing_response()
    window = room_acoustics.mixing_time_crossfade(pressure, 48000)
    return pressure, make_result(pressure), int(np.argmax(window >= 1))


def test_stage_is_loudspeaker():
    assert parametric.RoughnessCompensation.stage == 'loudspeaker'


def test_default_parameters():
    compensator = parametric.RoughnessCompensation()

    assert compensator.mode == 'schroeder'
    assert compensator.decay_t60 == 0.05
    assert compensator.seed is None
    assert compensator.start == 'mixing_time'


@pytest.mark.parametrize('mode', MODES)
def test_keeps_cshape_and_length(mode, mixing):
    pressure, result, _ = mixing
    signal = pf.Signal(np.stack([pressure] * 3), 48000)

    compensated = parametric.RoughnessCompensation(mode, seed=0)(signal,
                                                                 result)

    assert compensated.cshape == (3,)
    assert compensated.n_samples == 6000
    assert compensated.sampling_rate == 48000


@pytest.mark.parametrize('mode', MODES)
def test_samples_after_the_pressure_belong_to_the_late_part(mode, mixing):
    # Ear signals outlast the pressure by the head-related impulse responses.
    _, result, _ = mixing
    longer = np.random.default_rng(2).standard_normal((2, 6100))
    compensator = parametric.RoughnessCompensation(mode, seed=0)

    compensated = compensator(pf.Signal(longer, 48000), result)
    cropped = compensator(pf.Signal(longer[:, :6000], 48000), result)

    npt.assert_allclose(compensated.time[:, :6000], cropped.time, atol=1e-12)
    assert np.all(compensated.time[:, 6000:] != longer[:, 6000:])


def test_late_part_after_the_pressure_leaves_a_longer_signal_unchanged(
        mixing):
    _, result, _ = mixing
    longer = pf.Signal(np.random.default_rng(2).standard_normal((2, 6100)),
                       48000)

    with pytest.warns(UserWarning,
                      match='after the end of the impulse response'):
        compensated = parametric.RoughnessCompensation(start=1.0)(
            longer, result)

    npt.assert_array_equal(compensated.time, longer.time)


@pytest.mark.parametrize('mode', MODES)
def test_silence_stays_silent(mode, mixing):
    _, result, _ = mixing
    signal = pf.Signal(np.zeros((2, 6000)), 48000)

    compensated = parametric.RoughnessCompensation(mode, seed=0)(signal,
                                                                 result)

    npt.assert_array_equal(compensated.time, 0.0)


# -- the filters, applied to the whole signal with start='onset' --------------

def test_schroeder_is_an_allpass():
    signal = impulses(2**15, [(0, 0)])

    compensated = parametric.RoughnessCompensation(
        'schroeder', decay_t60=0.01, start='onset')(signal, whole(2**15))

    npt.assert_allclose(np.abs(compensated.freq), 1.0, atol=1e-9)


def test_schroeder_matches_the_difference_equation():
    # y[n] = g x[n] + x[n - M] - g y[n - M] for one allpass.
    delay, decay_t60, fs = 37, 0.05, 48000
    gain = 10 ** (-60 * delay / (fs * decay_t60) / 20)
    x = np.random.default_rng(1).standard_normal(400)
    y = np.zeros_like(x)
    for n in range(len(x)):
        y[n] = gain * x[n]
        if n >= delay:
            y[n] += x[n - delay] - gain * y[n - delay]

    npt.assert_allclose(
        pyspar.utils.dsp._schroeder_allpass(delay, decay_t60, fs).process(
            pf.Signal(x, fs)).time[0], y,
        atol=1e-14)


def test_schroeder_scales_the_delays_with_the_sampling_rate():
    signal = impulses(400, [(0, 0)], sampling_rate=24000)

    compensated = parametric.RoughnessCompensation('schroeder', start='onset')(
        signal, whole(400, 24000))

    # The shortest delay, 37 samples at 48 kHz, is 18 at 24 kHz.
    response = compensated.time[0]
    assert np.flatnonzero(response)[:2].tolist() == [0, 18]


def test_velvet_filter_spans_twice_the_decay_time():
    decay_t60 = 0.005
    n_taps = round(2 * decay_t60 * 48000)

    response = parametric.RoughnessCompensation(
        'velvet', decay_t60=decay_t60, seed=0, start='onset')(
            impulses(1000, [(0, 0)]), whole(1000))

    # The last tap is 60 dB below the first; the FFT convolution leaves
    # round-off beyond.
    assert np.all(np.abs(response.time[0, :n_taps]) > 1e-7)
    npt.assert_allclose(response.time[0, n_taps:], 0.0, atol=1e-15)


def test_velvet_filter_has_unit_energy():
    # Taps of +-1 under exp(-t / tau), divided by the root of the summed
    # squared envelope.
    decay_t60, fs = 0.005, 48000
    n_taps = round(2 * decay_t60 * fs)
    tau = -decay_t60 / np.log(1e-3) * fs
    envelope = np.exp(-np.arange(n_taps) / tau)

    response = parametric.RoughnessCompensation(
        'velvet', decay_t60=decay_t60, seed=0, start='onset')(
            impulses(1000, [(0, 0)]), whole(1000))

    npt.assert_allclose(np.sum(response.time ** 2), 1.0)
    npt.assert_allclose(np.abs(response.time[0, :n_taps]),
                        envelope / np.linalg.norm(envelope))


def test_velvet_draws_a_sequence_per_channel():
    signal = impulses(800, [(0, 5), (1, 5)], n_channels=2)

    compensated = parametric.RoughnessCompensation(
        'velvet', decay_t60=0.005, seed=0, start='onset')(signal, whole(800))

    assert np.any(np.sign(compensated.time[0]) != np.sign(compensated.time[1]))


def test_velvet_sequences_are_drawn_channel_by_channel():
    # All taps of the first channel come first, then those of the second.
    n_taps = round(2 * 0.005 * 48000)
    signs = np.random.default_rng(0).integers(0, 2, (2, n_taps)) * 2 - 1
    signal = impulses(800, [(0, 0), (1, 0)], n_channels=2)

    compensated = parametric.RoughnessCompensation(
        'velvet', decay_t60=0.005, seed=0, start='onset')(signal, whole(800))

    npt.assert_array_equal(np.sign(compensated.time[:, :n_taps]), signs)


def test_seed_makes_the_velvet_mode_reproducible():
    signal = impulses(600, [(0, 0), (0, 100)])

    first = parametric.RoughnessCompensation(
        'velvet', decay_t60=0.005, seed=9, start='onset')(signal, whole(600))
    second = parametric.RoughnessCompensation(
        'velvet', decay_t60=0.005, seed=9, start='onset')(signal, whole(600))

    npt.assert_array_equal(first.time, second.time)


def test_no_seed_draws_new_sequences():
    signal = impulses(600, [(0, 0)])
    compensator = parametric.RoughnessCompensation('velvet', decay_t60=0.005,
                                                   start='onset')

    assert compensator.seed is None
    assert np.any(compensator(signal, whole(600)).time
                  != compensator(signal, whole(600)).time)


# -- the late part ------------------------------------------------------------

@pytest.mark.parametrize('mode', MODES)
def test_only_the_late_part_is_compensated(mode, mixing):
    pressure, result, start = mixing
    signal = pf.Signal(np.stack([pressure, pressure]), 48000)

    compensated = parametric.RoughnessCompensation(mode, seed=0)(
        signal, result).time

    # The late part fades in over the 128 samples before its start.
    npt.assert_array_equal(compensated[:, :start - 128],
                           signal.time[:, :start - 128])
    assert np.max(np.abs(compensated[:, start:] - pressure[start:])) > 0.01
    npt.assert_allclose(np.sum(compensated[:, start:] ** 2, axis=1),
                        np.sum(pressure[start:] ** 2), rtol=0.3)


@pytest.mark.parametrize('mode', MODES)
def test_late_part_equals_the_filtered_fade(mode, mixing):
    pressure, result, start = mixing
    signal = pf.Signal(pressure[None], 48000)
    window = room_acoustics.mixing_time_crossfade(pressure, 48000)
    compensator = parametric.RoughnessCompensation(mode, seed=0)
    filtered = parametric.RoughnessCompensation(mode, seed=0, start='onset')(
        pf.Signal((pressure * window)[None], 48000), whole(6000)).time[0]

    compensated = compensator(signal, result).time[0]

    npt.assert_allclose(compensated, pressure * (1 - window) + filtered,
                        atol=1e-12)


def test_start_is_given_in_seconds_after_the_onset(mixing):
    pressure, result, _ = mixing
    start = impulse_response_start(pressure) + 480
    signal = pf.Signal(pressure[None], 48000)

    compensated = parametric.RoughnessCompensation(
        'schroeder', start=0.01)(signal, result).time[0]

    npt.assert_array_equal(compensated[:start - 128],
                           pressure[:start - 128])
    assert np.max(np.abs(compensated[start:] - pressure[start:])) > 0.01


@pytest.mark.parametrize('mode', MODES)
def test_late_part_beyond_the_end_leaves_the_signal_unchanged(mode, mixing):
    pressure, result, _ = mixing
    signal = pf.Signal(np.stack([pressure, pressure]), 48000)

    with pytest.warns(UserWarning,
                      match='after the end of the impulse response'):
        compensated = parametric.RoughnessCompensation(
            mode, seed=0, start=5.0)(signal, result)

    npt.assert_array_equal(compensated.time, signal.time)


@pytest.mark.parametrize('mode', MODES)
def test_warns_and_applies_nothing_without_a_mixing_time(mode, analysis):
    signal = pf.Signal(np.stack([analysis.pressure.time[0]] * 2), 48000)

    with pytest.warns(UserWarning, match='has no mixing time'):
        compensated = parametric.RoughnessCompensation(mode, seed=0)(
            signal, analysis)

    npt.assert_array_equal(compensated.time, signal.time)


# -- errors -------------------------------------------------------------------

@pytest.mark.parametrize('mode', ['none', 'velvet-time-varying'])
def test_error_if_mode_is_unknown(mode):
    with pytest.raises(ValueError, match=f"mode is '{mode}'"):
        parametric.RoughnessCompensation(mode)


@pytest.mark.parametrize('decay_t60', [0, -0.05])
def test_error_if_decay_time_is_not_positive(decay_t60):
    with pytest.raises(ValueError, match='decay_t60 is'):
        parametric.RoughnessCompensation(decay_t60=decay_t60)


@pytest.mark.parametrize('decay_t60', [True, 'long', [0.05]])
def test_error_if_decay_time_is_not_a_number(decay_t60):
    with pytest.raises(TypeError, match='decay_t60 must be a number'):
        parametric.RoughnessCompensation(decay_t60=decay_t60)


@pytest.mark.parametrize('start', [True, [0.08], None])
def test_error_if_start_is_not_an_option_or_a_number(start):
    with pytest.raises(TypeError, match='start must be'):
        parametric.RoughnessCompensation(start=start)


@pytest.mark.parametrize('start', ['full', 'Mixing_time'])
def test_error_if_start_is_an_unknown_option(start):
    with pytest.raises(ValueError, match=f"start is '{start}' but must be"):
        parametric.RoughnessCompensation(start=start)


@pytest.mark.parametrize('start', [-0.01, np.nan, np.inf])
def test_error_if_start_is_not_a_non_negative_time(start):
    with pytest.raises(ValueError,
                       match='start is .* but must be a non-negative'):
        parametric.RoughnessCompensation(start=start)


@pytest.mark.parametrize('start', [np.float32(0.01), np.int64(0)])
def test_numpy_numbers_are_accepted_as_start(start):
    given = parametric.RoughnessCompensation(start=start).start

    assert type(given) is float and given == float(start)


def test_onset_equals_a_start_of_zero(mixing):
    pressure, result, _ = mixing
    signal = pf.Signal(np.stack([pressure, pressure]), 48000)

    npt.assert_array_equal(
        parametric.RoughnessCompensation(start='onset')(signal, result).time,
        parametric.RoughnessCompensation(start=0)(signal, result).time)


@pytest.mark.parametrize('signal', [
    np.zeros((2, 100)), sh_signal(np.zeros((4, 100)))],
    ids=['array', 'spherical-harmonic-signal'])
def test_error_if_signal_is_not_a_pyfar_signal(signal):
    with pytest.raises(TypeError, match='requires a pyfar.Signal'):
        parametric.RoughnessCompensation()(signal, whole(100))


def test_error_if_the_analysis_result_is_missing():
    with pytest.raises(TypeError,
                       match='requires the pyspar.parametric.TimeResult'):
        parametric.RoughnessCompensation()(impulses(100, [(0, 0)]))


def test_error_if_signal_is_shorter_than_the_pressure():
    with pytest.raises(ValueError, match='100 samples but the pressure'):
        parametric.RoughnessCompensation()(impulses(100, [(0, 0)]),
                                           whole(200))


def test_error_if_sampling_rates_differ():
    with pytest.raises(ValueError, match='does not match the result'):
        parametric.RoughnessCompensation()(
            impulses(100, [(0, 0)], sampling_rate=44100), whole(100))


# -- in the synthesizers ------------------------------------------------------

def test_synthesizers_apply_none_by_default(layout_surround, hrir_set):
    assert parametric.LoudspeakerSynthesizer(layout_surround).roughness is None
    assert parametric.BinauralSynthesizer(*hrir_set).roughness is None


@pytest.mark.parametrize('mode', MODES)
def test_mode_string_gives_the_default_compensator(layout_surround, mode):
    roughness = parametric.LoudspeakerSynthesizer(
        layout_surround, roughness=mode).roughness

    assert isinstance(roughness, parametric.RoughnessCompensation)
    assert (roughness.mode, roughness.decay_t60, roughness.seed,
            roughness.start) == (mode, 0.05, None, 'mixing_time')


def test_compensator_instance_is_kept(layout_surround):
    compensator = parametric.RoughnessCompensation('schroeder', decay_t60=0.02)

    synthesizer = parametric.LoudspeakerSynthesizer(layout_surround,
                                                    roughness=compensator)

    assert synthesizer.roughness is compensator


def test_error_if_roughness_string_is_unknown(layout_surround):
    with pytest.raises(ValueError, match="mode is 'rough'"):
        parametric.LoudspeakerSynthesizer(layout_surround, roughness='rough')


def test_error_if_roughness_has_the_wrong_type(layout_surround):
    with pytest.raises(TypeError, match='roughness must be'):
        parametric.LoudspeakerSynthesizer(layout_surround, roughness=0.05)


def test_loudspeaker_render_compensates_after_panning(dense_analysis,
                                                      layout_surround):
    compensator = parametric.RoughnessCompensation('schroeder', start=0.01)
    plain = parametric.LoudspeakerSynthesizer(layout_surround)(dense_analysis)

    rendered = parametric.LoudspeakerSynthesizer(
        layout_surround, roughness=compensator)(dense_analysis)

    npt.assert_array_equal(rendered.time,
                           compensator(plain, dense_analysis).time)


def test_binaural_render_compensates_before_the_hrirs(dense_analysis,
                                                      hrir_set):
    compensator = parametric.RoughnessCompensation('schroeder', start=0.01)
    plain = parametric.BinauralSynthesizer(*hrir_set)
    expected = plain.convolve(compensator(
        plain.render_loudspeakers(dense_analysis), dense_analysis))

    synthesizer = parametric.BinauralSynthesizer(*hrir_set,
                                                 roughness=compensator)

    assert synthesizer.roughness is compensator
    npt.assert_array_equal(synthesizer(dense_analysis).time, expected.time)


def test_loudspeaker_spectrum_equalization_runs_after_the_compensation(
        dense_analysis, layout_surround):
    roughness = parametric.RoughnessCompensation('schroeder', start=0.01)
    equalizer = parametric.LoudspeakerSpectrumEqualization()
    plain = parametric.LoudspeakerSynthesizer(layout_surround)(dense_analysis)
    expected = equalizer(roughness(plain, dense_analysis),
                         result=dense_analysis)

    rendered = parametric.LoudspeakerSynthesizer(
        layout_surround, roughness=roughness, correction=equalizer)(
            dense_analysis)

    npt.assert_array_equal(rendered.time, expected.time)


def test_spectral_equalization_runs_after_the_compensation(
        dense_analysis, layout_surround):
    roughness = parametric.RoughnessCompensation('schroeder', start=0.01)
    equalizer = parametric.SpectralEqualization()
    plain = parametric.LoudspeakerSynthesizer(layout_surround)(dense_analysis)
    expected = equalizer(roughness(plain, dense_analysis),
                         result=dense_analysis)

    rendered = parametric.LoudspeakerSynthesizer(
        layout_surround, roughness=roughness, correction=equalizer)(
            dense_analysis)

    npt.assert_array_equal(rendered.time, expected.time)


def test_lfe_crossover_runs_after_the_compensation(noise_between_silences):
    loudspeakers = layout(*LAYOUT_LFE_ONE.T)
    compensator = parametric.RoughnessCompensation('velvet', start='onset',
                                                   seed=ROUGHNESS_SEED)
    # Panned to the other loudspeakers, the LFE channel silent.
    others = np.delete(LAYOUT_LFE_ONE, LFE_ONE, axis=0)
    panned = np.insert(
        parametric.LoudspeakerSynthesizer(layout(*others.T))(
            noise_between_silences).time, LFE_ONE[0], 0.0, axis=0)
    expected = _lfe_crossover(
        compensator(pf.Signal(panned, 48000), noise_between_silences),
        loudspeakers, LFE_ONE, 80.0)

    rendered = parametric.LoudspeakerSynthesizer(
        loudspeakers, lfe_channels=LFE_ONE, roughness=compensator)(
            noise_between_silences)

    npt.assert_array_equal(rendered.time, expected.time)
