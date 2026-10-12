"""Tests of the mixing time and the late part of impulse responses."""
import warnings

import numpy as np
import numpy.testing as npt
import pytest
from conftest import (
    impulse_response_start, make_dense_response, make_mixing_response)

from pyspar.utils import room_acoustics


# -- mixing time --------------------------------------------------------------

def test_late_part_counts_from_the_start_of_the_impulse_response():
    # The start is the last sample 30 dB below the peak, one sample before
    # the direct sound of this response.
    pressure = make_mixing_response()

    window = room_acoustics.mixing_time_crossfade(pressure, 48000,
                                                  start=0.01)

    assert impulse_response_start(pressure) == 239
    assert int(np.argmax(window >= 1)) == 239 + 480


def test_first_sample_within_30_db_of_a_later_peak_starts_the_response():
    # pyfar returns the sample before the first one, -1; the response then
    # starts at its first sample.
    rng = np.random.default_rng(4)
    pressure = np.concatenate([[0.5, 1.0], 0.3 * rng.standard_normal(9600)
                               * np.exp(-np.arange(9600) / 4000)])

    window = room_acoustics.mixing_time_crossfade(pressure, 48000, start=0.0)

    npt.assert_array_equal(window, 1.0)
    assert room_acoustics.mixing_time(pressure, 48000) > 0


def test_no_warning_about_the_noise_of_a_short_response():
    # The dense response ends before its noise floor, which the search for
    # the start would take for a low signal-to-noise ratio.
    pressure = np.asarray(make_dense_response().time)[0, 0]

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        room_acoustics.mixing_time(pressure, 48000)
        room_acoustics.mixing_time_crossfade(pressure, 48000, start=0.01)


def test_silent_response_has_no_late_part():
    with pytest.warns(UserWarning, match='has no mixing time'):
        window = room_acoustics.mixing_time_crossfade(np.zeros(4800), 48000)

    assert window is None


def test_mixing_time_counts_from_the_onset():
    pressure = make_mixing_response()
    delayed = np.concatenate([np.zeros(500), pressure])

    assert room_acoustics.mixing_time(delayed, 48000) == (
        room_acoustics.mixing_time(pressure, 48000))


def test_mixing_time_counts_from_a_start_near_the_first_sample():
    # With the direct sound within the first 100 samples, fewer samples
    # precede the start than the margin kept before it; the time is still
    # counted from the start, the sample before the direct sound.
    pressure = make_mixing_response()
    early = pressure[190:]

    assert impulse_response_start(early) == 49
    assert room_acoustics.mixing_time(early, 48000) == (
        room_acoustics.mixing_time(pressure, 48000))


def test_response_beginning_with_its_peak_starts_at_the_peak():
    # No sample lies before the peak, so the start is the peak itself, one
    # sample later than for the same response with silence before it.
    pressure = make_mixing_response()
    trimmed = pressure[240:]

    assert impulse_response_start(trimmed) == 0
    one_sample = 1 / 48000
    npt.assert_allclose(
        room_acoustics.mixing_time(trimmed, 48000),
        room_acoustics.mixing_time(pressure, 48000) - one_sample, rtol=1e-12)


def test_mixing_time_follows_the_start_of_the_noise_tail():
    # The tail sets in 20 ms after the direct sound.
    mixing_time = room_acoustics.mixing_time(make_mixing_response(), 48000)

    assert 0.020 < mixing_time < 0.040


def test_mixing_time_window_scales_with_the_sampling_rate():
    # The same response with every sample repeated is the response at twice
    # the sampling rate; the window then covers the same time, up to the
    # repeated samples at its ends.
    pressure = make_mixing_response()

    npt.assert_allclose(
        room_acoustics.mixing_time(np.repeat(pressure, 2), 96000),
        room_acoustics.mixing_time(pressure, 48000), atol=2e-4)


def test_response_dense_from_its_start_mixes_at_its_start():
    pressure = np.asarray(make_dense_response().time)[0, 0]

    assert room_acoustics.mixing_time(pressure, 48000) < 0.001


def test_error_if_response_never_gets_as_dense_as_noise(impulse_response):
    pressure = np.asarray(impulse_response.time)[0, 0]

    with pytest.raises(ValueError, match='has no mixing time'):
        room_acoustics.mixing_time(pressure, 48000)


def test_error_if_response_is_too_short_for_a_mixing_time():
    pressure = np.random.default_rng(0).standard_normal(800)

    with pytest.raises(ValueError, match='needs at least 1024'):
        room_acoustics.mixing_time(pressure, 48000)


# -- late part ----------------------------------------------------------------

def test_late_part_starts_at_the_onset_plus_the_mixing_time():
    pressure = make_mixing_response()
    start = impulse_response_start(pressure) + round(
        room_acoustics.mixing_time(pressure, 48000) * 48000)

    window = room_acoustics.mixing_time_crossfade(pressure, 48000)

    npt.assert_array_equal(window[:start - 128], 0.0)
    npt.assert_array_equal(window[start:], 1.0)
    npt.assert_allclose(window[start - 128:start], np.hanning(256)[:128],
                        atol=1e-12)


def test_late_part_start_can_be_given_in_seconds_after_the_onset():
    pressure = make_mixing_response()
    start = impulse_response_start(pressure) + 480

    window = room_acoustics.mixing_time_crossfade(pressure, 48000, start=0.01)

    assert np.argmax(window >= 1) == start


def test_late_part_fade_shrinks_at_the_ends():
    pressure = make_mixing_response()

    with pytest.warns(UserWarning,
                      match='after the end of the impulse response'):
        assert room_acoustics.mixing_time_crossfade(
            pressure, 48000, start=1.0) is None
    window = room_acoustics.mixing_time_crossfade(pressure, 48000, start=0.0)
    onset = impulse_response_start(pressure)
    assert window[0] == 0.0 and window[onset - 1] < 1.0
    npt.assert_array_equal(window[onset:], 1.0)


def test_mixing_time_crossfade_warns_and_is_none_without_a_mixing_time(
        impulse_response):
    pressure = np.asarray(impulse_response.time)[0, 0]

    with pytest.warns(UserWarning, match='has no mixing time'):
        window = room_acoustics.mixing_time_crossfade(pressure, 48000)

    assert window is None


def test_mixing_time_crossfade_warns_and_is_none_for_a_short_response():
    pressure = np.random.default_rng(0).standard_normal(800)

    with pytest.warns(UserWarning, match='needs at least 1024'):
        assert room_acoustics.mixing_time_crossfade(pressure, 48000) is None
