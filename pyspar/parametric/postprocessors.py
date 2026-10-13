"""Corrections applied to rendered or loudspeaker signals."""
from __future__ import annotations

import warnings

import numpy as np
import pyfar as pf
from spharpy import SphericalHarmonicSignal

from .classes import TimeResult, Postprocessor
from pyspar.utils.spherical_harmonics import _check_sh_input


class SpectralDecayCorrection(Postprocessor):
    r"""Correct the spectral decay of an upmixed Ambisonic signal.

    The spectral decay correction of Zaunschirm et al. [#]_ equalizes, in every
    third-octave band :math:`b`, the energetic envelope of every order
    :math:`n` of an upmixed signal to that of a reference,

    .. math::
        \hat{h}_{nm}(t, b) = \tilde{h}_{nm}(t, b) \sqrt{\frac{
        (2n + 1)\, E_\mathrm{ref}(t, b)}{
        \sum_m E\{\tilde{h}_{nm}^2(t, b)\}}},

    with the energies :math:`E\{\cdot\}` smoothed by a Hann window of 1024
    samples at 48 kHz and the reference energy, the mixture of the
    omnidirectional and the first-order channels of [#]_ normalized so that
    it equals :math:`E\{w^2\}` for a single plane wave,

    .. math::
        E_\mathrm{ref} = \frac{1}{2} \left( E\{w^2\} +
        \frac{E\{x^2\} + E\{y^2\} + E\{z^2\}}{3} \right)

    of the N3D channels. The correction includes the order zero.

    :class:`~pyspar.parametric.AmbisonicSynthesizer` applies the correction by
    default. This class applies it to an upmix made with
    ``spectral_decay_correction=False``.

    See Also
    --------
    pyspar.parametric.AmbisonicSynthesizer

    Notes
    -----
    The bands are zero phase and add up to the signal. The window has
    ``round(fs / 48000) * 1024`` samples, and 32 below 24 kHz. The gains are
    regularized by adding ``1e-6`` to the root of the smoothed energy of the
    upmixed signal, so the correction depends slightly on the level of the
    input. The band signals and the smoothing are computed over a transform of
    the next power of two of the signal length; the smoothing filters the
    squares with the window and advances the result circularly by half its
    length.

    An analysis by time differences of arrival keeps no first-order signal. The
    reference energy is then that of the pressure alone,
    :math:`E_\mathrm{ref} = E\{p^2\}` as in the correction of Zaunschirm et
    al., and the omnidirectional channel stays the pressure up to the
    regularization of the gains.

    References
    ----------
    .. [#] M. Zaunschirm, M. Frank and F. Zotter, "Binaural rendering with
           measured room responses: First-order Ambisonic microphone vs.
           dummy head," Appl. Sci., vol. 10, no. 5, p. 1631, 2020.
    .. [#] E. Hoffbauer and M. Frank, "Four-directional Ambisonic spatial
           decomposition method with reduced temporal artifacts," J. Audio
           Eng. Soc., vol. 70, no. 12, pp. 1002-1014, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> import pyspar
    >>> from spharpy import SphericalHarmonicSignal
    >>> signal = SphericalHarmonicSignal(
    ...     np.random.default_rng(0).standard_normal((4, 256)), 48000,
    ...     'real', 'N3D', 'ACN', False)
    >>> result = pyspar.parametric.PseudoIntensityVectorAnalyzer(
    ...     window_length=5)(signal)
    >>> upmixed = pyspar.parametric.AmbisonicSynthesizer(
    ...     order=3, spectral_decay_correction=False)(result)
    >>> corrected = pyspar.parametric.SpectralDecayCorrection()(
    ...     upmixed, result)
    >>> corrected.n_max, corrected.normalization
    (3, 'N3D')
    """

    stage = 'output'

    def __call__(self, signal, result: TimeResult | None = None):
        """Correct the spectral decay of `signal`.

        Parameters
        ----------
        signal : spharpy.SphericalHarmonicSignal
            Upmixed Ambisonic signal, real, in any normalization and channel
            convention, without the Condon-Shortley phase, with as many
            samples as the analysis.
        result : pyspar.parametric.TimeResult
            Analysis the signal was upmixed from. Its first-order signal or,
            after an analysis by time differences of arrival, its pressure
            is the reference.

        Returns
        -------
        signal : spharpy.SphericalHarmonicSignal
            Corrected signal, with the order and conventions of `signal`.
        """
        if not isinstance(result, TimeResult):
            raise TypeError(
                "SpectralDecayCorrection requires the "
                "pyspar.parametric.TimeResult the signal was upmixed "
                f"from; got {type(result).__name__}.")
        _check_sh_input(signal, 'SpectralDecayCorrection', 0)
        if abs(signal.sampling_rate - result.sampling_rate) > 0.5:
            raise ValueError(
                f"The signal sampling_rate ({signal.sampling_rate} Hz) does "
                f"not match the result ({result.sampling_rate} Hz).")
        if signal.n_samples != result.pressure.n_samples:
            raise ValueError(
                f"signal has {signal.n_samples} samples but the analysis "
                f"{result.pressure.n_samples}.")
        # Computed on N3D channels, against the first-order signal or,
        # without one, the pressure.
        upmix = signal.copy()
        upmix.channel_convention = 'ACN'
        upmix.normalization = 'N3D'
        data = np.asarray(upmix.time, dtype=float).reshape(
            -1, signal.n_samples).T
        if isinstance(result.signal, SphericalHarmonicSignal):
            n3d = result.signal.copy()
            n3d.channel_convention = 'ACN'
            n3d.normalization = 'N3D'
            reference = np.asarray(n3d.time, dtype=float).reshape(
                -1, n3d.n_samples).T[:, :4]
        else:
            reference = np.tile(result.pressure.time[0][:, None], (1, 4))
        corrected = SphericalHarmonicSignal(
            _spectral_decay_correction(
                data, reference, result.sampling_rate).T,
            signal.sampling_rate, 'real', 'N3D', 'ACN', False)
        corrected.normalization = signal.normalization
        corrected.channel_convention = signal.channel_convention
        return corrected


def _spectral_decay_correction(encoded, reference, sampling_rate):
    """Match the decay of an upmix to that of the first-order signal.

    The spectral decay correction described in
    :class:`SpectralDecayCorrection`.

    Parameters
    ----------
    encoded : numpy.ndarray, shape (n_samples, (order + 1) ** 2)
        Encoded signal in ACN order and N3D normalization.
    reference : numpy.ndarray, shape (n_samples, 4)
        First-order signal in N3D normalization, omnidirectional channel
        first.
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    corrected : numpy.ndarray, shape (n_samples, (order + 1) ** 2)
        Corrected signal.
    """
    n_samples, n_coefficients = encoded.shape
    order = int(np.sqrt(n_coefficients)) - 1

    # round(fs / 48000) * 1024 with halves rounded up, at least 32.
    n_window = max(int(np.floor(sampling_rate / 48000 + 0.5)) * 1024, 32)
    window = pf.dsp.time_window(pf.Signal(
        np.ones(n_window), sampling_rate), (0, n_window - 1)).time[0][:, None]

    n_fft = 2 ** int(np.ceil(np.log2(n_samples)))
    # pyfar's reconstructing third-octave bank, 25 Hz to 20 kHz, as zero-phase
    # magnitudes interpolated onto the bins of the transform.
    n_taps = 2 ** int(np.ceil(np.log2(sampling_rate)))
    with warnings.catch_warnings():
        # pyfar warns about its deprecated centre frequencies and skipped
        # bands.
        warnings.simplefilter('ignore')
        filters = pf.dsp.filter.reconstructing_fractional_octave_bands(
            None, 3, (25, 20000), n_samples=n_taps,
            sampling_rate=sampling_rate)
    coefficients = np.asarray((filters[0] if isinstance(filters, tuple)
                               else filters).coefficients).reshape(-1, n_taps)
    designed = np.real(pf.dsp.zero_phase(pf.dsp.pad_zeros(
        pf.Signal(coefficients, sampling_rate), n_taps)).freq)
    bank = np.column_stack([
        np.interp(pf.dsp.fft.rfftfreq(n_fft, sampling_rate),
                  pf.dsp.fft.rfftfreq(2 * n_taps, sampling_rate), band)
        for band in designed])

    def smooth(squares):
        # The window as a causal filter, shifted back circularly by half its
        # length.
        return pf.dsp.time_shift(pf.dsp.time_crop(pf.dsp.convolve(
            pf.Signal(squares.T, sampling_rate),
            pf.Signal(window[:, 0], sampling_rate),
            method='fft'), (0, n_fft - 1)), -(window.size // 2)).time.T

    reference = pf.dsp.fft.rfft(reference.T, n_fft, sampling_rate, 'none').T
    encoded = pf.dsp.fft.rfft(encoded.T, n_fft, sampling_rate, 'none').T
    corrected = np.zeros((n_fft, n_coefficients))
    for magnitudes in bank.T[:, :, None]:
        energy = smooth(pf.dsp.fft.irfft(
            (reference * magnitudes).T, n_fft, sampling_rate, 'none').T ** 2)
        target = np.sqrt(np.abs(
            0.5 * energy[:, 0] + 0.5 * np.sum(energy[:, 1:4], axis=1) / 3))
        for n in range(order + 1):
            degree = slice(n ** 2, (n + 1) ** 2)
            band = pf.dsp.fft.irfft(
                (encoded[:, degree] * magnitudes).T, n_fft, sampling_rate,
                'none').T
            rms = np.sqrt(np.abs(np.sum(smooth(band ** 2), axis=1)))
            gain = target / (rms + 1e-6) * np.sqrt(2 * n + 1)
            corrected[:, degree] += band * gain[:, None]
    return corrected[:n_samples]
