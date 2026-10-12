"""Signal processing shared by the postprocessors."""
import numpy as np
import pyfar as pf
from scipy.interpolate import PchipInterpolator
from scipy.signal import filtfilt


def _schroeder_allpass(delay, decay_t60, sampling_rate):
    """Schroeder allpass as a pyfar.FilterIIR.

    ``y[n] = g x[n] + x[n - delay] - g y[n - delay]`` with
    ``g = 10^(-3 delay / (sampling_rate decay_t60))``, for which the impulse
    response decays by 60 dB in `decay_t60`.
    """
    gain = 10 ** (-60 * delay / (sampling_rate * decay_t60) / 20)
    b = np.zeros(delay + 1)
    a = np.zeros(delay + 1)
    b[0], b[delay] = gain, 1.0
    a[0], a[delay] = 1.0, gain
    return pf.FilterIIR([b, a], sampling_rate)


def _smooth_log_frequency(spectra, frequencies, num_fractions):
    """Smooth magnitude spectra on a logarithmic frequency axis.

    The magnitudes are interpolated onto a logarithmic axis, filtered forward
    and backward with a normalized square-root Hann window that spans
    ``2 / num_fractions`` octaves, and interpolated back. The values at 0 Hz
    and at the last frequency are kept.

    Parameters
    ----------
    spectra : numpy.ndarray, shape (n_bins, n_spectra)
        Spectra from 0 Hz to the Nyquist frequency, one per column.
    frequencies : numpy.ndarray, shape (n_bins,)
        Frequencies of the bins in Hz.
    num_fractions : int
        Smoothing width: the window spans ``2 / num_fractions`` octaves.

    Returns
    -------
    smoothed : numpy.ndarray, shape (n_bins, n_spectra)
        Smoothed magnitudes.
    """
    magnitude = np.abs(spectra)
    # Round half away from zero.
    octaves = np.floor(np.log2(frequencies[-1] / frequencies[1]) + 0.5)
    steps = np.linspace(1, octaves, len(frequencies))
    log_frequencies = frequencies[1] * 2.0 ** steps
    n_window = int(np.floor(2 / (steps[1] - steps[0]) / num_fractions + 0.5))

    window = np.sqrt(pf.dsp.time_window(pf.Signal(
        np.ones(n_window), 1), (0, n_window - 1)).time[0])
    window /= np.sum(window)
    smoothed = PchipInterpolator(frequencies, magnitude)(log_frequencies)
    # Extended at both ends by three times the filter order.
    smoothed = filtfilt(window, 1.0, smoothed, axis=0,
                        padlen=3 * (n_window - 1))
    smoothed = PchipInterpolator(log_frequencies, smoothed)(frequencies)
    smoothed[[0, -1]] = magnitude[[0, -1]]
    return smoothed
