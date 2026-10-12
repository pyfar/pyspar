"""Room acoustic parameters of impulse responses."""
import warnings
from math import erfc

import numpy as np
import pyfar as pf


def mixing_time(pressure, sampling_rate):
    """Mixing time of an impulse response from its echo density.

    The echo density profile [#]_ is the fraction of the samples in a
    sliding rectangular window of 1024 samples at 48 kHz, scaled with the
    sampling rate and truncated at the ends of the response, whose magnitude
    exceeds the sample standard deviation of the window, normalized by the
    fraction ``erfc(1 / sqrt(2))`` expected for Gaussian noise. The mixing
    time is where the profile first exceeds one.

    Parameters
    ----------
    pressure : numpy.ndarray, shape (n_samples,)
        Impulse response.
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    mixing_time : float
        Mixing time in seconds after the start of the impulse response, the
        last sample 30 dB below its peak as
        :func:`pyfar.dsp.find_impulse_response_start` finds it, or its first
        sample if that is within 30 dB already.

    References
    ----------
    .. [#] J. S. Abel and P. Huang, "A simple, robust measure of
           reverberation echo density," in Proc. 121st Conv. Audio Eng.
           Soc., San Francisco, CA, USA, 2006.
    """
    pressure = np.asarray(pressure, dtype=float)
    # The last sample 30 dB below the peak before it, zero where the first
    # sample is within 30 dB already.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        start = max(int(np.ravel(pf.dsp.find_impulse_response_start(
            pf.Signal(pressure, sampling_rate), threshold=30))[0]), 0)
    # 100 samples at 48 kHz before the start are kept if there are that
    # many.
    margin = round(100 * sampling_rate / 48000)
    kept = 0 if start + 1 <= margin else margin
    response = pressure[start - kept:]
    n_window = 2 * round(512 * sampling_rate / 48000)
    n_samples = len(response)
    if n_samples < n_window:
        raise ValueError(
            f"The impulse response has {n_samples} samples from its start "
            f"but the mixing time needs at least {n_window}.")

    gaussian = 1 / erfc(1 / np.sqrt(2))
    half = n_window // 2
    for k in range(1, n_samples + 1):
        window = response[max(k - half - 1, 0):min(k + half - 1, n_samples)]
        outside = np.count_nonzero(
            np.abs(window) > np.std(window, ddof=1))
        if outside / len(window) * gaussian > 1:
            # Counted from the start, after the samples kept before it.
            return (k - kept) / sampling_rate
    raise ValueError(
        "The echo density of the impulse response never reaches that of "
        "Gaussian noise, so it has no mixing time.")


def mixing_time_crossfade(pressure, sampling_rate, start=None, fade=256):
    """Crossfade from the early to the late part of an impulse response.

    The late part begins at the start of the impulse response, as
    :func:`mixing_time` takes it, plus the mixing time or plus `start`. The
    window fades the late part in: zero before the fade, the rising half of a
    Hann window of `fade` samples ending at that sample, and one from there on;
    one minus the window fades the early part out. The fade is shortened where
    fewer samples lie before or after it.

    Parameters
    ----------
    pressure : numpy.ndarray, shape (n_samples,)
        Impulse response.
    sampling_rate : float
        Sampling rate in Hz.
    start : float, optional
        Start of the late part in seconds after the start of the impulse
        response, non-negative and finite. ``None``, the default, takes the
        mixing time.
    fade : int, optional
        Length of the Hann window of the fade in samples. The default is
        ``256``.

    Returns
    -------
    window : numpy.ndarray, shape (n_samples,) or None
        The window, or ``None`` with a warning if `start` is ``None`` and
        the mixing time cannot be estimated, or if the late part begins
        after the last sample of `pressure`.
    """
    pressure = np.asarray(pressure, dtype=float)
    n_samples = len(pressure)
    # The last sample 30 dB below the peak before it, zero where the first
    # sample is within 30 dB already.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        first = max(int(np.ravel(pf.dsp.find_impulse_response_start(
            pf.Signal(pressure, sampling_rate), threshold=30))[0]), 0)
    if start is None:
        try:
            start = mixing_time(pressure, sampling_rate)
        except ValueError as error:
            warnings.warn(
                f"{error} The late part cannot be told from the early part, "
                "so nothing is applied to it.", stacklevel=3)
            return None
    begin = first + round(start * sampling_rate)
    if begin >= n_samples:
        warnings.warn(
            f"The late part begins {start:g} s after the start of the impulse "
            "response, after the end of the impulse response, so nothing is "
            "applied to it.", stacklevel=3)
        return None
    half = min(fade // 2, begin, n_samples - begin)
    if half == 0:
        return (np.arange(n_samples) >= begin).astype(float)
    return pf.dsp.time_window(
        pf.Signal(pressure, sampling_rate), (begin - half, begin),
        shape='left', return_window=True)[1].time[0]
