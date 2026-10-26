"""Tests of the LFE channels of the loudspeaker synthesizer."""
import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
from conftest import (
    LAYOUT_LFE_ONE, LAYOUT_LFE_TWO, LAYOUT_SURROUND, LFE_ONE, LFE_TWO, layout)

from pyspar import parametric
from pyspar.parametric.synthesizers import _lfe_shares

LAYOUTS = {'one': (LAYOUT_LFE_ONE, LFE_ONE), 'two': (LAYOUT_LFE_TWO, LFE_TWO)}


def render(result, loudspeakers, **kwargs):
    """Loudspeaker signals as ``(n_samples, n_loudspeakers)``."""
    signal = parametric.LoudspeakerSynthesizer(loudspeakers, **kwargs)(result)
    return np.asarray(signal.time).T


def without(degrees, lfe_channels):
    """Layout in degrees without its LFE channels."""
    return layout(*np.delete(degrees, lfe_channels, axis=0).T)


def make_result(pressure, azimuth, elevation, sampling_rate=48000):
    """Analysis result of `pressure` arriving from directions in degrees."""
    pressure = np.asarray(pressure, dtype=float)
    doa = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(azimuth) * np.ones(pressure.size),
        np.deg2rad(elevation) * np.ones(pressure.size), 1)
    return parametric.TimeResult(
        pressure=pf.Signal(pressure, sampling_rate), doa=doa,
        signal=pf.Signal(pressure, sampling_rate))


def impulse_result(azimuth, elevation=0, n_samples=6000):
    """A unit impulse from one direction in the middle of silence."""
    pressure = np.zeros(n_samples)
    pressure[n_samples // 2] = 1.0
    return make_result(pressure, azimuth, elevation)


# -- crossover ----------------------------------------------------------------

@pytest.mark.parametrize('name', ['one', 'two'])
def test_channels_add_up_to_the_pressure(noise_between_silences, name):
    degrees, lfe_channels = LAYOUTS[name]
    result = noise_between_silences

    data = render(result, layout(*degrees.T), lfe_channels=lfe_channels)

    npt.assert_allclose(np.sum(data, axis=1), result.pressure.time[0], rtol=0,
                        atol=1e-9 * np.max(np.abs(result.pressure.time[0])))


@pytest.mark.parametrize('kwargs', [
    {'panning': 'nls'}, {'dimensionality': 2}])
def test_channels_add_up_as_without_lfe_channels(noise_between_silences,
                                                 kwargs):
    # A layout enclosing the listener with an LFE channel appended.
    degrees = np.vstack([LAYOUT_SURROUND, [[20, -30]]])
    result = noise_between_silences

    data = render(result, layout(*degrees.T), lfe_channels=8, **kwargs)
    plain = render(result, layout(*LAYOUT_SURROUND.T), **kwargs)

    expected = np.sum(plain, axis=1)
    npt.assert_allclose(np.sum(data, axis=1), expected, rtol=0,
                        atol=1e-9 * np.max(np.abs(expected)))


def test_lfe_channels_are_left_out_of_the_panning(noise_between_silences):
    # Far above the crossover the other loudspeakers play what they play in
    # the layout without the LFE channel, and the LFE channel is silent.
    degrees = np.vstack([LAYOUT_SURROUND, [[20, -30]]])
    result = noise_between_silences
    high = np.fft.rfftfreq(result.pressure.n_samples, 1 / 48000) > 2000

    data = render(result, layout(*degrees.T), lfe_channels=8)
    plain = render(result, layout(*LAYOUT_SURROUND.T))

    spectrum = np.fft.rfft(data, axis=0)[high]
    expected = np.fft.rfft(plain, axis=0)[high]
    peak = np.max(np.abs(expected))
    npt.assert_allclose(spectrum[:, :8], expected, rtol=0, atol=1e-6 * peak)
    npt.assert_allclose(spectrum[:, 8], 0, atol=1e-6 * peak)


def test_sound_from_the_direction_of_an_lfe_channel_is_panned_elsewhere():
    # The LFE channel of the 5.1 layout is listed 30 degrees below the front.
    result = impulse_result(0, -30)

    data = render(result, layout(*LAYOUT_LFE_ONE.T), lfe_channels=LFE_ONE)

    center = 2
    energy = np.sum(data ** 2, axis=0)
    assert np.argmax(energy) == center
    npt.assert_array_equal(np.delete(energy, [center, LFE_ONE[0]]), 0.0)


@pytest.mark.parametrize('lfe_cutoff', [80, 200])
def test_cutoff_is_where_both_parts_have_half_the_amplitude(lfe_cutoff):
    # Forward and backward, the third-order filters have -6 dB at the cutoff.
    result = impulse_result(30)
    n_samples = result.pressure.n_samples
    at_cutoff = round(lfe_cutoff * n_samples / 48000)

    data = render(result, layout(*LAYOUT_LFE_ONE.T), lfe_channels=LFE_ONE,
                  lfe_cutoff=lfe_cutoff)

    spectrum = np.abs(np.fft.rfft(data, axis=0))[at_cutoff]
    npt.assert_allclose(spectrum[LFE_ONE[0]], 0.5, rtol=1e-3)
    npt.assert_allclose(np.sum(np.delete(spectrum, LFE_ONE)), 0.5, rtol=1e-3)


def test_lfe_channel_takes_the_lows_and_the_others_keep_the_highs():
    result = impulse_result(30)
    frequencies = np.fft.rfftfreq(result.pressure.n_samples, 1 / 48000)

    data = render(result, layout(*LAYOUT_LFE_ONE.T), lfe_channels=LFE_ONE)

    spectrum = np.abs(np.fft.rfft(data, axis=0))
    lfe, others = spectrum[:, LFE_ONE[0]], np.sum(
        np.delete(spectrum, LFE_ONE, axis=1), axis=1)
    npt.assert_allclose(lfe[frequencies < 20], 1.0, atol=1e-3)
    npt.assert_allclose(others[frequencies < 20], 0.0, atol=1e-3)
    npt.assert_allclose(lfe[frequencies > 500], 0.0, atol=1e-3)
    npt.assert_allclose(others[frequencies > 500], 1.0, atol=1e-3)


def test_crossover_has_zero_phase():
    # The low-passed impulse is symmetric around the sample of the impulse.
    result = impulse_result(30)
    center = result.pressure.n_samples // 2

    data = render(result, layout(*LAYOUT_LFE_ONE.T), lfe_channels=LFE_ONE)

    lfe = data[:, LFE_ONE[0]]
    assert np.argmax(lfe) == center
    npt.assert_allclose(lfe[center + 1:center + 2000],
                        lfe[center - 1:center - 2000:-1], rtol=0,
                        atol=1e-8 * np.max(lfe))


def test_short_signals_are_filtered_too():
    result = make_result(np.ones(6), 30, 0)

    data = render(result, layout(*LAYOUT_LFE_ONE.T), lfe_channels=LFE_ONE)

    assert data.shape == (6, 6)
    assert np.all(np.isfinite(data))


# -- sides of two and more LFE channels ---------------------------------------

@pytest.mark.parametrize(('azimuth', 'left', 'right'), [
    (30, 1, 0), (-110, 0, 1), (0, 0.5, 0.5), (180, 0.5, 0.5)])
def test_two_lfe_channels_share_the_loudspeakers_by_side(azimuth, left,
                                                         right):
    # 5.2 with a loudspeaker behind; the left LFE channel is at +45 degrees.
    degrees = np.vstack([LAYOUT_LFE_TWO, [[180, 0]]])
    result = impulse_result(azimuth)

    data = render(result, layout(*degrees.T), lfe_channels=LFE_TWO)
    single = render(result, layout(*LAYOUT_LFE_ONE.T), lfe_channels=LFE_ONE)

    low_passed = single[:, LFE_ONE[0]]
    npt.assert_allclose(data[:, LFE_TWO[0]], left * low_passed, atol=1e-15)
    npt.assert_allclose(data[:, LFE_TWO[1]], right * low_passed, atol=1e-15)


def test_side_of_an_lfe_channel_follows_its_position():
    # The same layout with the two LFE channels listed in the other order.
    swapped = LAYOUT_LFE_TWO[[0, 1, 2, 3, 4, 6, 5]]
    result = impulse_result(30)

    data = render(result, layout(*LAYOUT_LFE_TWO.T), lfe_channels=LFE_TWO)
    other = render(result, layout(*swapped.T), lfe_channels=LFE_TWO)

    npt.assert_array_equal(other[:, [0, 1, 2, 3, 4, 6, 5]], data)


@pytest.mark.parametrize(('lfe_azimuths', 'expected'), [
    # One LFE channel gets everything, wherever it is.
    ([20], [[1], [1], [1]]),
    # Two on opposite sides: by side, the center half to each.
    ([45, -45], [[1, 0], [0, 1], [0.5, 0.5]]),
    # Two on the same side share everything equally.
    ([45, 135], [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
    # Two per side, as front and rear subwoofers.
    ([26, 154, -26, -154], [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5],
                            [0.25, 0.25, 0.25, 0.25]]),
    # One in the median plane counts for both sides.
    ([45, 0, -45], [[0.5, 0.5, 0], [0, 0.5, 0.5], [0.25, 0.5, 0.25]]),
])
def test_shares_of_the_lfe_channels(lfe_azimuths, expected):
    # Loudspeakers on the left, on the right and in the center, then the LFE
    # channels.
    azimuth = np.deg2rad([60, -60, 0, *lfe_azimuths])
    positions = np.column_stack(
        [np.cos(azimuth), np.sin(azimuth), np.zeros(azimuth.size)])
    lfe_channels = tuple(range(3, 3 + len(lfe_azimuths)))

    shares = _lfe_shares(positions, lfe_channels)

    npt.assert_allclose(shares, expected, atol=1e-15)
    npt.assert_allclose(np.sum(shares, axis=1), 1.0)


# -- arguments ----------------------------------------------------------------

def test_default_is_no_lfe_channel(layout_surround):
    synthesizer = parametric.LoudspeakerSynthesizer(layout_surround)

    assert synthesizer.lfe_channels == ()
    assert synthesizer.lfe_cutoff == 80.0


@pytest.mark.parametrize('lfe_channels', [None, [], ()])
def test_no_lfe_channel_leaves_the_render_unfiltered(analysis,
                                                     layout_surround,
                                                     lfe_channels):
    data = render(analysis, layout_surround, lfe_channels=lfe_channels)

    npt.assert_array_equal(np.sum(data, axis=1), analysis.pressure.time[0])


@pytest.mark.parametrize('lfe_channels', [
    3, [3], (3,), np.array([3]), np.int64(3)])
def test_lfe_channels_accept_an_index_or_a_sequence(analysis, lfe_channels):
    loudspeakers = layout(*LAYOUT_LFE_ONE.T)
    synthesizer = parametric.LoudspeakerSynthesizer(
        loudspeakers, lfe_channels=lfe_channels)

    assert synthesizer.lfe_channels == (3,)
    npt.assert_array_equal(
        np.asarray(synthesizer(analysis).time).T,
        render(analysis, loudspeakers, lfe_channels=LFE_ONE))


def test_lfe_channels_are_stored_sorted(analysis):
    loudspeakers = layout(*LAYOUT_LFE_TWO.T)
    synthesizer = parametric.LoudspeakerSynthesizer(
        loudspeakers, lfe_channels=[6, 5])

    assert synthesizer.lfe_channels == (5, 6)
    npt.assert_array_equal(
        np.asarray(synthesizer(analysis).time).T,
        render(analysis, loudspeakers, lfe_channels=LFE_TWO))


@pytest.mark.parametrize('lfe_channels', [
    [6], [-1], [3, 3], [0.5], [True], 'front', [[0, 1]]])
def test_error_if_the_lfe_channels_are_not_distinct_indices(lfe_channels):
    with pytest.raises(ValueError, match='must hold distinct indices of '
                                         'the 6 loudspeakers'):
        parametric.LoudspeakerSynthesizer(
            layout(*LAYOUT_LFE_ONE.T), lfe_channels=lfe_channels)


def test_error_if_every_loudspeaker_is_an_lfe_channel():
    with pytest.raises(ValueError, match='leaves none to pan to'):
        parametric.LoudspeakerSynthesizer(
            layout(*LAYOUT_LFE_ONE.T), lfe_channels=range(6))


@pytest.mark.parametrize('lfe_cutoff', [0, -80, np.nan, np.inf])
def test_error_if_the_cutoff_is_not_positive_and_finite(lfe_cutoff):
    with pytest.raises(ValueError, match='lfe_cutoff is .* but must be '
                                         'positive and finite'):
        parametric.LoudspeakerSynthesizer(
            layout(*LAYOUT_LFE_ONE.T), lfe_channels=LFE_ONE,
            lfe_cutoff=lfe_cutoff)


@pytest.mark.parametrize('lfe_cutoff', ['low', True, None])
def test_error_if_the_cutoff_is_not_a_number(lfe_cutoff):
    with pytest.raises(TypeError, match='lfe_cutoff must be a number'):
        parametric.LoudspeakerSynthesizer(
            layout(*LAYOUT_LFE_ONE.T), lfe_channels=LFE_ONE,
            lfe_cutoff=lfe_cutoff)


def test_error_if_the_cutoff_is_not_below_half_the_sampling_rate(analysis):
    synthesizer = parametric.LoudspeakerSynthesizer(
        layout(*LAYOUT_LFE_ONE.T), lfe_channels=LFE_ONE, lfe_cutoff=24000)

    with pytest.raises(ValueError, match='below half the sampling rate'):
        synthesizer(analysis)


def test_cutoff_is_not_checked_without_lfe_channels(analysis,
                                                    layout_surround):
    data = render(analysis, layout_surround, lfe_cutoff=30000)

    npt.assert_array_equal(np.sum(data, axis=1), analysis.pressure.time[0])


# -- pipeline -----------------------------------------------------------------

def test_sdm_loudspeakers_passes_the_lfe_channels(impulse_response, analysis):
    loudspeakers = layout(*LAYOUT_LFE_ONE.T)
    expected = parametric.LoudspeakerSynthesizer(
        loudspeakers, lfe_channels=LFE_ONE, lfe_cutoff=120)(analysis)

    rendered = parametric.sdm_loudspeakers(
        impulse_response, loudspeakers, window_length=5,
        lfe_channels=LFE_ONE, lfe_cutoff=120)

    npt.assert_array_equal(rendered.time, expected.time)


def test_sdm_loudspeakers_has_no_lfe_channel_by_default(impulse_response,
                                                        analysis,
                                                        layout_surround):
    rendered = parametric.sdm_loudspeakers(impulse_response, layout_surround,
                                           window_length=5)

    npt.assert_array_equal(np.sum(rendered.time, axis=0),
                           analysis.pressure.time[0])
