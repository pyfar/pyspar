"""Corrections applied to rendered or loudspeaker signals."""
from __future__ import annotations

import warnings

import numpy as np
import pyfar as pf
from spharpy import SphericalHarmonicSignal

from .classes import TimeResult, Postprocessor
from pyspar.utils.dsp import _smooth_log_frequency


class SpectralEqualization(Postprocessor):
    """Match the spectrum of loudspeaker signals to that of the pressure.

    The pressure and every loudspeaker signal are cut into blocks of 2048
    samples without overlap. Per block, the magnitude of the regularized ratio
    of their spectra, ``P conj(H) / (|H|^2 + beta^2)``, is smoothed in third
    octaves and turned into a filter of 1024 taps that keeps the energy of the
    block. The filter is applied to the block of the loudspeaker signal, and
    the filtered blocks are added with their tails overlapping.

    Parameters
    ----------
    mode : ``'minphase'``, ``'linphase'``, optional
        Phase of the filters. ``'minphase'`` (default) gives minimum-phase
        filters, which do not ring before the sound; ``'linphase'`` gives
        linear-phase filters, which delay every block by 512 samples.
    max_boost_db : float, optional
        Strength of the regularization in dB,
        ``beta = rms / (2 * 10^(max_boost_db / 20))`` with the rms magnitude
        of the spectrum of the block: notches of the loudspeaker spectrum
        more than ``max_boost_db + 6`` dB below that rms are not inverted
        fully. It does not limit the gain of the filters, which follows from
        keeping the energy. The default is ``20``; very large values turn the
        regularization off.

    Attributes
    ----------
    mode : str
        Phase of the filters.
    max_boost_db : float
        Strength of the regularization in dB.

    See Also
    --------
    LoudspeakerSpectrumEqualization

    Notes
    -----
    `beta` is relative to the level of each block. A loudspeaker that is silent
    in a block stays silent, and a block of a loudspeaker passes unfiltered if
    the pressure is silent in it. The minimum phase is computed with
    :func:`pyfar.dsp.minimum_phase`.

    Examples
    --------
    Equalize the loudspeaker signals inside the synthesizer.

    >>> import numpy as np
    >>> import pyfar as pf
    >>> import pyspar
    >>> from spharpy import SphericalHarmonicSignal
    >>> signal = SphericalHarmonicSignal(
    ...     np.random.default_rng(0).standard_normal((4, 4096)), 48000,
    ...     'real', 'N3D', 'ACN', False)
    >>> loudspeakers = pf.Coordinates.from_spherical_elevation(
    ...     np.deg2rad([0, 30, -30, 110, -110]), 0, 1)
    >>> result = pyspar.parametric.PseudoIntensityVectorAnalyzer(
    ...     window_length=5)(signal)
    >>> rendered = pyspar.parametric.LoudspeakerSynthesizer(
    ...     loudspeakers, correction='minphase')(result)
    >>> rendered.cshape, rendered.n_samples
    ((5,), 5119)
    """

    stage = 'loudspeaker'

    def __init__(self, mode='minphase', max_boost_db=20):
        if mode not in ('minphase', 'linphase'):
            raise ValueError(
                f"mode is {mode!r} but must be 'minphase' or 'linphase'.")
        if (isinstance(max_boost_db, bool)
                or not isinstance(max_boost_db, (int, float, np.number))):
            raise TypeError(
                "max_boost_db must be a number; got "
                f"{type(max_boost_db).__name__}.")
        if not np.isfinite(max_boost_db):
            raise ValueError(
                f"max_boost_db is {max_boost_db} but must be finite.")
        self.mode = mode
        self.max_boost_db = float(max_boost_db)

    def __call__(self, signal, result: TimeResult | None = None):
        """Equalize every channel of `signal` towards the pressure.

        Parameters
        ----------
        signal : pyfar.Signal
            Loudspeaker signals with as many samples as the pressure.
        result : pyspar.parametric.TimeResult
            Analysis the signals were rendered from; its pressure is the
            spectral target.

        Returns
        -------
        signal : pyfar.Signal
            Equalized signals with the cshape of `signal` and 1023 samples
            more for the tail of the filters.
        """
        if (not isinstance(signal, pf.Signal)
                or isinstance(signal, SphericalHarmonicSignal)):
            raise TypeError(
                "SpectralEqualization requires a pyfar.Signal; got "
                f"{type(signal).__name__}.")
        if not isinstance(result, TimeResult):
            raise TypeError(
                "SpectralEqualization requires the "
                "pyspar.parametric.TimeResult the signal was rendered "
                f"from; got {type(result).__name__}.")
        pressure = result.pressure.time[0]
        if signal.n_samples != len(pressure):
            raise ValueError(
                f"The signal has {signal.n_samples} samples but the pressure "
                f"of the result has {len(pressure)}.")

        n = 2048                       # block length in samples
        fs = signal.sampling_rate
        feeds = np.asarray(signal.time, dtype=float).reshape(
            -1, signal.n_samples).T                           # (N, channels)
        n_samples, n_channels = feeds.shape
        n_blocks = max(1, -(-n_samples // n))
        pad = n_blocks * n - n_samples

        # Consecutive blocks, the last one zero-padded: (n, n_blocks[, ch]).
        target = pf.Signal(
            np.pad(pressure, (0, pad)).reshape(n_blocks, n), fs).freq.T
        blocks = np.pad(feeds, ((0, pad), (0, 0))).reshape(
            n_blocks, n, n_channels).transpose(1, 0, 2)
        spectra = np.moveaxis(pf.Signal(
            np.moveaxis(blocks, 0, -1), fs).freq, -1, 0)
        frequencies = pf.dsp.fft.rfftfreq(n, fs)
        regularization = 1 / (2 * 10 ** (self.max_boost_db / 20))

        out = np.zeros((n_blocks * n + n // 2 - 1, n_channels))
        for ch in range(n_channels):
            power = np.abs(spectra[..., ch]) ** 2
            floor = power + (regularization ** 2) * np.mean(power, axis=0)
            # Zero where the block is silent; those blocks are skipped below.
            inverse = np.divide(target * np.conj(spectra[..., ch]), floor,
                                out=np.zeros_like(target), where=floor > 0)
            # Smoothed in third octaves.
            smoothed = _smooth_log_frequency(inverse, frequencies, 3)
            if self.mode == 'minphase':
                # The cepstrum is sampled on the block length.
                filters = pf.dsp.minimum_phase(
                    pf.Signal(smoothed.T, fs, n, 'freq'), n_fft=n).time.T
            else:
                filters = pf.dsp.time_shift(pf.Signal(
                    smoothed.T, fs, n, 'freq'), n // 4).time.T[:n // 2]

            for k in range(n_blocks):
                block = blocks[:, k, ch]
                block = pf.Signal(block, fs)
                energy = pf.dsp.energy(block)[0]
                if energy == 0:
                    continue
                filtered = pf.dsp.convolve(
                    pf.Signal(filters[:, k], fs), block, method='fft')
                if pf.dsp.energy(filtered)[0] > 0:
                    filtered = pf.dsp.normalize(
                        filtered, reference_method='energy',
                        target=energy).time[0]
                else:
                    filtered = block.time[0]
                out[k * n:k * n + len(filtered), ch] += filtered

        # The padding of the last block adds nothing beyond the tail.
        out = out[:n_samples + n // 2 - 1]
        return pf.Signal(out.T.reshape(signal.cshape + (-1,)), fs)


class LoudspeakerSpectrumEqualization(Postprocessor):
    r"""Post equalization of the loudspeaker spectra.

    The post equalization of the loudspeaker spectra of Tervo et al. [#]_ in
    octave bands. For each band, the loudspeaker signals and the pressure are
    cut into short-time Fourier-transform frames of `periods` periods of the
    band's lower edge with 50 % overlap. Every loudspeaker spectrum is scaled
    per frequency bin, keeping its phase, so that the sum of the loudspeaker
    powers matches the power of the pressure; the lowest bands deviate, see
    Notes. The scaled frames are added with their overlap and filtered by the
    band filter, and the bands are summed.

    Parameters
    ----------
    periods : int, optional
        Length of the frames of a band in periods of its lower edge; the
        low-pass band, which reaches down to 0 Hz, counts periods of the
        geometric mean of 1 Hz and its upper edge of 44 Hz. The default is
        ``7``: about 1 s for the low-pass, 157 ms for the octave at 63 Hz
        and 0.6 ms for the octave at 16 kHz.
    max_boost_db : float, optional
        Soft limit of the gain in dB. Gains above 0 dB are compressed
        towards the limit by :func:`pyfar.dsp.soft_limit_spectrum` with a
        knee of twice the limit, so notches of the sum of the loudspeaker
        spectra are not inverted fully. The default is ``6``;
        ``numpy.inf`` applies no limit.

    Attributes
    ----------
    periods : int
        Frame length in periods of the lower band edge.
    max_boost_db : float
        Soft limit of the gain in dB.

    See Also
    --------
    SpectralEqualization

    Notes
    -----
    The bands are the amplitude-preserving filters of
    :func:`pyfar.dsp.filter.reconstructing_fractional_octave_bands`: a low-pass
    below 44 Hz, the octaves at 63 Hz to 8 kHz, and the octave at 16 kHz
    extended to the Nyquist frequency. Each band's frames are weighted by a
    Hann window, transformed with :math:`2 L - 1` points for :math:`L` samples,
    scaled by the zero-phase gain

    .. math::
        g(k) = \frac{|P(k)|}{\sqrt{\sum_l |X_l(k)|^2}}

    with the frame spectrum :math:`P` of the pressure and :math:`X_l` of the
    loudspeakers, and added with their overlap.

    In the low-pass band and the octaves at 63 and 125 Hz, the root of the
    summed powers in the denominator is mixed with the sum of the magnitudes,
    :math:`\sum_l |X_l(k)|`, in shares of 1, 1/2 and 1/4 of the latter. A
    loudspeaker that is silent stays silent, and the signals keep their length.

    References
    ----------
    .. [#] S. Tervo, J. Patynen, N. Kaplanis, M. Lydolf, S. Bech and T.
           Lokki, "Spatial analysis and synthesis of car audio system and
           car cabin acoustics with a compact microphone array," J. Audio
           Eng. Soc., vol. 63, no. 11, pp. 914-925, 2015.

    Examples
    --------
    Equalize the loudspeaker signals inside the synthesizer.

    >>> import numpy as np
    >>> import pyfar as pf
    >>> import pyspar
    >>> from spharpy import SphericalHarmonicSignal
    >>> signal = SphericalHarmonicSignal(
    ...     np.random.default_rng(0).standard_normal((4, 4096)), 48000,
    ...     'real', 'N3D', 'ACN', False)
    >>> loudspeakers = pf.Coordinates.from_spherical_elevation(
    ...     np.deg2rad([0, 30, -30, 110, -110]), 0, 1)
    >>> result = pyspar.parametric.PseudoIntensityVectorAnalyzer(
    ...     window_length=5)(signal)
    >>> rendered = pyspar.parametric.LoudspeakerSynthesizer(
    ...     loudspeakers,
    ...     correction=pyspar.parametric.LoudspeakerSpectrumEqualization())(
    ...         result)
    >>> rendered.cshape, rendered.n_samples
    ((5,), 4096)
    """

    stage = 'loudspeaker'

    def __init__(self, periods=7, max_boost_db=6.0):
        if isinstance(periods, bool) or not isinstance(periods,
                                                       (int, np.integer)):
            raise TypeError(
                f"periods must be an integer; got {type(periods).__name__}.")
        if periods < 1:
            raise ValueError(f"periods is {periods} but must be at least 1.")
        if (isinstance(max_boost_db, bool)
                or not isinstance(max_boost_db, (int, float, np.number))):
            raise TypeError(
                "max_boost_db must be a number; got "
                f"{type(max_boost_db).__name__}.")
        if np.isnan(max_boost_db) or max_boost_db < 0:
            raise ValueError(
                f"max_boost_db is {max_boost_db} but must be non-negative.")
        self.periods = int(periods)
        self.max_boost_db = float(max_boost_db)

    def __call__(self, signal, result: TimeResult | None = None):
        """Equalize the loudspeaker signals together towards the pressure.

        Parameters
        ----------
        signal : pyfar.Signal
            Loudspeaker signals with as many samples as the pressure.
        result : pyspar.parametric.TimeResult
            Analysis the signals were rendered from; its pressure is the
            spectral target.

        Returns
        -------
        signal : pyfar.Signal
            Equalized signals with the cshape and length of `signal`.
        """
        if (not isinstance(signal, pf.Signal)
                or isinstance(signal, SphericalHarmonicSignal)):
            raise TypeError(
                "LoudspeakerSpectrumEqualization requires a pyfar.Signal; "
                f"got {type(signal).__name__}.")
        if not isinstance(result, TimeResult):
            raise TypeError(
                "LoudspeakerSpectrumEqualization requires the "
                "pyspar.parametric.TimeResult the signal was rendered "
                f"from; got {type(result).__name__}.")
        fs = signal.sampling_rate
        if abs(fs - result.sampling_rate) > 0.5:
            raise ValueError(
                f"The signal sampling_rate ({fs} Hz) does not match the "
                f"result ({result.sampling_rate} Hz).")
        pressure = result.pressure.time[0]
        n_samples = signal.n_samples
        if n_samples != len(pressure):
            raise ValueError(
                f"The signal has {n_samples} samples but the pressure of "
                f"the result has {len(pressure)}.")

        feeds = np.asarray(signal.time, dtype=float).reshape(-1, n_samples)
        equalized = _equalize_loudspeaker_spectra(
            feeds, pressure, fs, self.periods, self.max_boost_db)
        return pf.Signal(equalized.reshape(signal.cshape + (-1,)), fs)


def _equalize_loudspeaker_spectra(feeds, pressure, sampling_rate, periods,
                                  max_boost_db):
    """Post equalization of the loudspeaker spectra, see the class.

    Parameters
    ----------
    feeds : numpy.ndarray, shape (n_loudspeakers, n_samples)
        Loudspeaker signals.
    pressure : numpy.ndarray, shape (n_samples,)
        Pressure, the spectral target.
    sampling_rate : float
        Sampling rate in Hz.
    periods : int
        Frame length in periods of the lower band edge.
    max_boost_db : float
        Soft limit of the gain in dB, ``inf`` for none.

    Returns
    -------
    equalized : numpy.ndarray, shape (n_loudspeakers, n_samples)
        Equalized loudspeaker signals.
    """
    n_loudspeakers, n_samples = feeds.shape
    # pyfar's reconstructing octave bank with a low-pass below 44 Hz.
    n_taps = 2 ** int(np.ceil(np.log2(sampling_rate)))
    with warnings.catch_warnings():
        # The returned centre frequencies are deprecated.
        warnings.simplefilter('ignore')
        bank = pf.dsp.filter.reconstructing_fractional_octave_bands(
            None, 1, (31.5, 16000), n_samples=n_taps,
            sampling_rate=sampling_rate)
    filters = np.asarray((bank[0] if isinstance(bank, tuple)
                          else bank).coefficients).reshape(-1, n_taps)
    # Frames of `periods` periods of the lower band edge, of even length.
    _, lower, upper = pf.constants.fractional_octave_frequencies_exact(
        1, (31.5, 16000))
    lower = lower[:len(filters)]
    lower[0] = np.sqrt(upper[0])
    frame_lengths = 2 * np.ceil(
        periods * sampling_rate / lower / 2).astype(int)
    pad = int(frame_lengths.max())
    # Shares of the coherent sum of the magnitudes in the denominator of
    # the gain, for the lowest bands.
    coherent = np.zeros(len(filters))
    coherent[:3] = [1.0, 0.5, 0.25]

    padded = np.pad(feeds, ((0, 0), (pad, pad)))
    pressure_padded = np.pad(pressure, (pad, pad))
    buffers = np.zeros((n_loudspeakers, n_samples + 4 * pad))
    for response, length, share in zip(filters, frame_lengths, coherent,
                                       strict=True):
        hop = length // 2
        window = pf.dsp.time_window(pf.Signal(
            np.ones(length + 1), sampling_rate), (0, length)).time[0, :-1]
        n_fft = 2 * length - 1
        delay = n_fft // 2
        starts = np.arange(0, padded.shape[1] - length + 1, hop)
        frames = np.lib.stride_tricks.sliding_window_view(
            padded, length, axis=1)[:, starts]                # (L, F, length)
        spectra = pf.dsp.fft.rfft(
            frames * window, n_fft, sampling_rate, 'none')
        target = np.abs(pf.dsp.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(
                pressure_padded, length)[starts] * window, n_fft,
            sampling_rate, 'none'))
        magnitudes = np.abs(spectra)
        incoherent = np.sqrt(np.sum(magnitudes ** 2, axis=0))
        summed = np.sum(magnitudes, axis=0)
        denominator = (share * np.maximum(summed, 1e-9)
                       + (1 - share) * np.maximum(incoherent, 1e-9))
        gain = target / denominator
        if np.isfinite(max_boost_db):
            with np.errstate(divide='ignore'):
                gain = np.abs(pf.dsp.soft_limit_spectrum(
                    pf.FrequencyData(gain, pf.dsp.fft.rfftfreq(
                        n_fft, sampling_rate)), max_boost_db,
                    2 * max_boost_db).freq)
        equalized = pf.dsp.time_shift(pf.Signal(
            spectra * gain, sampling_rate, n_fft, 'freq'), delay).time
        band_buffer = np.zeros_like(buffers)
        for frame, start in enumerate(starts):
            first = pad + start - delay
            band_buffer[:, first:first + n_fft] += equalized[:, frame]
        # The band filter delays by half its length.
        filtered = pf.dsp.convolve(
            pf.Signal(band_buffer, sampling_rate),
            pf.Signal(response, sampling_rate), method='fft').time
        buffers += filtered[:, n_taps // 2:n_taps // 2 + buffers.shape[1]]
    return buffers[:, 2 * pad:2 * pad + n_samples]


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
            Upmixed Ambisonic signal, real, in ACN with N3D or SN3D
            normalization and without the Condon-Shortley phase, with as
            many samples as the analysis.
        result : pyspar.parametric.TimeResult
            Analysis the signal was upmixed from. Its first-order signal or,
            after an analysis by time differences of arrival, its pressure
            is the reference.

        Returns
        -------
        signal : spharpy.SphericalHarmonicSignal
            Corrected signal, with the order and normalization of `signal`.
        """
        if not isinstance(signal, SphericalHarmonicSignal):
            raise TypeError(
                "SpectralDecayCorrection corrects Ambisonic signals: signal "
                "must be a spharpy.SphericalHarmonicSignal; got "
                f"{type(signal).__name__}.")
        if not isinstance(result, TimeResult):
            raise TypeError(
                "SpectralDecayCorrection requires the "
                "pyspar.parametric.TimeResult the signal was upmixed "
                f"from; got {type(result).__name__}.")
        if (signal.basis_type != 'real' or signal.channel_convention != 'ACN'
                or signal.normalization not in ('N3D', 'SN3D')
                or signal.condon_shortley):
            raise ValueError(
                "signal must be real, in ACN with N3D or SN3D normalization "
                "and without the Condon-Shortley phase; got "
                f"{signal.basis_type}, {signal.channel_convention}, "
                f"{signal.normalization}, condon_shortley="
                f"{signal.condon_shortley}.")
        if np.prod(signal.cshape[:-1]) != 1:
            raise ValueError(
                "signal must hold one Ambisonic signal; got cshape "
                f"{signal.cshape}.")
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
        upmix.normalization = 'N3D'
        data = np.asarray(upmix.time, dtype=float).reshape(
            -1, signal.n_samples).T
        if isinstance(result.signal, SphericalHarmonicSignal):
            n3d = result.signal.copy()
            n3d.normalization = 'N3D'
            # Contiguous, so that the result does not depend on the order
            # of the stored signal.
            reference = np.ascontiguousarray(np.asarray(
                n3d.time, dtype=float).reshape(-1, n3d.n_samples).T)[:, :4]
        else:
            reference = np.tile(result.pressure.time[0][:, None], (1, 4))
        corrected = SphericalHarmonicSignal(
            _spectral_decay_correction(
                data, reference, result.sampling_rate).T,
            signal.sampling_rate, 'real', 'N3D', 'ACN', False)
        corrected.normalization = signal.normalization
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
