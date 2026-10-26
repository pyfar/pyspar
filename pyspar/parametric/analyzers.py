"""Spatial analyzers estimating a direction of arrival per sample."""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pyfar as pf
from scipy.fft import fft, fftshift, ifft
from scipy.ndimage import convolve1d
from spharpy import SphericalHarmonicSignal

from .classes import AnalyzerBase, TimeResult
from pyspar.utils.spherical_harmonics import sh_to_n3d


class PseudoIntensityVectorAnalyzer(AnalyzerBase):
    """Estimate directions of arrival by the pseudo-intensity vector.

    The Spatial Decomposition Method [#]_ assigns every sample of a measured
    impulse response the direction its energy arrives from. This analyzer
    takes a first-order spherical harmonic signal and forms the product of
    the omnidirectional channel with each of the three dipole channels, the
    pseudo-intensity vector, which is averaged over time by a Hann window
    before it is normalized, see Notes. For the capsule signals of an open
    microphone array, use :class:`TimeDifferenceOfArrivalAnalyzer`.

    Parameters
    ----------
    window_length : int, optional
        Length in samples of the window smoothing the vector, centred on
        each sample; it must be odd. The default ``1`` estimates from single
        samples; a few samples smooth the directions of a measured response.

    Attributes
    ----------
    window_length : int
        Window length in samples, as given.

    Notes
    -----
    Each Cartesian component of the vector is convolved with a symmetric Hann
    window of `window_length` samples without its zero end points, so the
    smoothed vector is the energy-weighted mean of the directions within the
    window. The window is centred, so the direction of an arrival does not lag
    its pressure. Where the smoothed vector vanishes, the sample carries no
    direction.

    References
    ----------
    .. [#] S. Tervo, J. Patynen, A. Kuusinen and T. Lokki, "Spatial
           decomposition method for room impulse responses," J. Audio Eng.
           Soc., vol. 61, no. 1/2, pp. 17-28, 2013.

    Examples
    --------
    Estimate the direction of a plane wave arriving from the left.

    >>> import numpy as np
    >>> import pyfar as pf
    >>> import pyspar
    >>> from spharpy import SphericalHarmonicSignal
    >>> from spharpy.spherical import spherical_harmonic_basis_real
    >>> direction = pf.Coordinates.from_spherical_elevation(np.pi / 2, 0, 1)
    >>> data = np.zeros((4, 64))
    >>> data[:, 8] = spherical_harmonic_basis_real(1, direction)[0]
    >>> signal = SphericalHarmonicSignal(
    ...     data, 48000, 'real', 'N3D', 'ACN', False)
    >>> result = pyspar.parametric.PseudoIntensityVectorAnalyzer()(signal)
    >>> np.round(result.doa.cartesian[8], 12)
    array([0., 1., 0.])
    """

    def __init__(self, window_length=1):
        if (isinstance(window_length, bool)
                or not isinstance(window_length, (int, np.integer))):
            raise TypeError(
                "window_length must be an integer; got "
                f"{type(window_length).__name__}.")
        if window_length < 1:
            raise ValueError(
                f"window_length is {window_length} but must be at least 1.")
        if window_length % 2 == 0:
            raise ValueError(
                f"window_length is {window_length} but must be odd, so that "
                "the window is centred.")

        self.window_length = int(window_length)

    def __call__(self, signal) -> TimeResult:
        """Analyze `signal` and return an analysis result.

        Parameters
        ----------
        signal : spharpy.SphericalHarmonicSignal
            Spherical harmonic signal of order one or higher in any channel
            convention and normalization; orders above one are truncated to
            first order with a warning.

        Returns
        -------
        result : pyspar.parametric.TimeResult
            Pressure and direction of arrival per sample, and the first-order
            signal in ACN/N3D. Samples without a direction are NaN.
        """
        if (isinstance(signal, pf.Signal)
                and not isinstance(signal, SphericalHarmonicSignal)):
            raise TypeError(
                f"{type(self).__name__} requires a "
                "spharpy.SphericalHarmonicSignal; got "
                f"{type(signal).__name__}. Use "
                "TimeDifferenceOfArrivalAnalyzer for the capsule signals "
                "of an open microphone array.")
        data = sh_to_n3d(signal, type(self).__name__)
        if signal.n_max > 1:
            # Converted in full above, then cut to the first order.
            warnings.warn(
                f"signal is of order {signal.n_max}. The pseudo-intensity "
                "vector uses first order only, the remaining channels are "
                "ignored.", stacklevel=2)
            data = data[:, :4]

        first_order = SphericalHarmonicSignal(
            data.T, signal.sampling_rate, 'real', 'N3D', 'ACN', False)
        doa = self._pseudo_intensity_vector(data)

        return TimeResult(
            pressure=pf.Signal(data[:, 0].copy(), signal.sampling_rate),
            doa=pf.Coordinates(doa[:, 0], doa[:, 1], doa[:, 2]),
            signal=first_order)

    def _pseudo_intensity_vector(self, data):
        """Return unit Cartesian directions of the pseudo-intensity vector.

        Parameters
        ----------
        data : numpy.ndarray, shape (n_samples, 4)
            First-order time data in ACN/N3D.

        Returns
        -------
        doa : numpy.ndarray, shape (n_samples, 3)
            Unit Cartesian directions. Samples whose vector vanishes are
            NaN and carry no direction. The vector vanishes where the
            signal is digitally silent within the window and where the
            omnidirectional channel is exactly zero, which recordings of
            finite resolution reach near some of its zero crossings.
        """
        # ACN order is [W, Y, Z, X], the vector is [W X, W Y, W Z].
        vector = data[:, [3, 1, 2]] * data[:, :1]
        # Direct, not FFT, convolution: a vanishing vector stays exactly zero.
        window = np.hanning(self.window_length + 2)[1:-1]
        return _unit(convolve1d(vector, window, axis=0, mode='constant'))


class TimeDifferenceOfArrivalAnalyzer(AnalyzerBase):
    """Estimate directions of arrival by time differences of arrival.

    The Spatial Decomposition Method [#]_ assigns every sample of a measured
    impulse response the direction its energy arrives from. This analyzer
    takes the capsule signals of an open microphone array: the time
    differences of arrival between all capsule pairs come from their
    generalized cross-correlation in a sliding window, and the direction is
    the least-squares plane-wave solution. For a spherical harmonic signal, use
    :class:`PseudoIntensityVectorAnalyzer`.

    Parameters
    ----------
    microphones : pyfar.Coordinates
        Capsule positions in meters, one per channel of the analyzed signal.
        They must span all three dimensions.
    reference_channel : int
        The capsule whose signal is taken as the pressure, best the one
        nearest the centre of the array, for example the top capsule of an
        array without a central one. There is no default, since the choice
        depends on the array.
    window_length : int, optional
        Length of the analysis frame in samples, an even number of at least
        the minimum the array needs, ``2 ceil(d_max / c fs) + 8`` for the
        largest capsule spacing ``d_max``, the speed of sound ``c``,
        :py:data:`pyfar.constants.reference_speed_of_sound`, and the
        sampling rate ``fs`` of the signal. ``None``, the default, uses that
        minimum: 22 samples for a spacing of 5 cm at 48 kHz.

    Attributes
    ----------
    microphones : pyfar.Coordinates
        Capsule positions.
    reference_channel : int
        Capsule used as the pressure.
    window_length : int or None
        Window length in samples, as given.

    References
    ----------
    .. [#] S. Tervo, J. Patynen, A. Kuusinen and T. Lokki, "Spatial
           decomposition method for room impulse responses," J. Audio Eng.
           Soc., vol. 61, no. 1/2, pp. 17-28, 2013.

    Examples
    --------
    Estimate the direction of a plane wave arriving from the left at six
    capsules on the axes, placed so that the wave reaches them at whole
    samples.

    >>> import numpy as np
    >>> import pyfar as pf
    >>> import pyspar
    >>> radius = 4 * pf.constants.reference_speed_of_sound / 48000
    >>> axes = np.vstack([np.eye(3), -np.eye(3)])
    >>> microphones = pf.Coordinates(*(radius * axes).T)
    >>> data = np.zeros((6, 256))
    >>> data[np.arange(6), 100 - 4 * axes[:, 1].astype(int)] = 1
    >>> analyzer = pyspar.parametric.TimeDifferenceOfArrivalAnalyzer(
    ...     microphones, reference_channel=2)
    >>> result = analyzer(pf.Signal(data, 48000))
    >>> np.round(result.doa.cartesian[100] / result.doa.radius[100], 12)
    array([0., 1., 0.])
    """

    def __init__(self, microphones, reference_channel, window_length=None):
        if window_length is not None and (
                isinstance(window_length, bool)
                or not isinstance(window_length, (int, np.integer))):
            raise TypeError(
                "window_length must be an integer or None; got "
                f"{type(window_length).__name__}.")
        self.microphones = microphones
        self.reference_channel = reference_channel
        self.window_length = (None if window_length is None
                              else int(window_length))

        if not isinstance(microphones, pf.Coordinates):
            raise TypeError(
                "microphones must be a pyfar.Coordinates; got "
                f"{type(microphones).__name__}.")
        positions = np.atleast_2d(microphones.cartesian).reshape(-1, 3)
        n_microphones = len(positions)
        pairs = np.array(list(combinations(range(n_microphones), 2)),
                         dtype=int).reshape(-1, 2)
        differences = positions[pairs[:, 0]] - positions[pairs[:, 1]]
        if len(differences) < 3 or np.linalg.matrix_rank(differences) < 3:
            raise ValueError(
                "microphones must span all three dimensions, got "
                f"{n_microphones} capsules that do not.")
        if (isinstance(reference_channel, bool)
                or not isinstance(reference_channel, (int, np.integer))):
            raise TypeError(
                "reference_channel must be an integer; got "
                f"{type(reference_channel).__name__}.")
        if not 0 <= reference_channel < n_microphones:
            raise ValueError(
                f"reference_channel is {reference_channel} but must be the "
                f"index of one of the {n_microphones} capsules.")

        self._pairs = pairs
        self._inverse = np.linalg.pinv(differences)
        self._spacing = np.max(np.linalg.norm(differences, axis=1))

    def __call__(self, signal) -> TimeResult:
        """Analyze `signal` and return an analysis result.

        Parameters
        ----------
        signal : pyfar.Signal
            Capsule signals with ``cshape (n_microphones,)``.

        Returns
        -------
        result : pyspar.parametric.TimeResult
            Pressure, image-source position per sample, and the capsule
            signals. Samples without a direction are NaN.
        """
        if (not isinstance(signal, pf.Signal)
                or isinstance(signal, SphericalHarmonicSignal)):
            raise TypeError(
                f"{type(self).__name__} requires the capsule signals as a "
                f"pyfar.Signal; got {type(signal).__name__}. Use "
                "PseudoIntensityVectorAnalyzer for a spherical harmonic "
                "signal.")
        if signal.cshape != (self.microphones.csize,):
            raise ValueError(
                f"signal has cshape {signal.cshape} but the array has "
                f"{self.microphones.csize} capsules.")
        fs = float(signal.sampling_rate)
        minimum = 2 * int(np.ceil(
            self._spacing / pf.constants.reference_speed_of_sound * fs)) + 8
        frame = minimum if self.window_length is None else self.window_length
        if frame < minimum:
            raise ValueError(
                f"window_length is {self.window_length} but the array needs "
                f"at least {minimum} samples at {fs:g} Hz.")
        if frame % 2:
            # The direction is assigned to the centre of the frame, which
            # falls on a sample only for an even length.
            raise ValueError(
                f"window_length is {self.window_length} but the frame must "
                "be even.")

        data = np.asarray(signal.time, dtype=float).T
        if data.shape[0] <= frame:
            raise ValueError(
                f"signal has {data.shape[0]} samples, fewer than the "
                f"{frame}-sample analysis frame.")

        return TimeResult(
            pressure=pf.Signal(data[:, self.reference_channel].copy(), fs),
            doa=pf.Coordinates(*self._image_sources(data, fs, frame).T),
            signal=signal.copy())

    def _image_sources(self, data, fs, frame):
        """Image-source position per sample, ``(n_samples, 3)``.

        The direction is scaled by the distance travelled since the start of
        the response. Samples at either end not covered by a full frame, and
        frames whose correlation has no peak, such as silent ones, are zero
        vectors.
        """
        c = pf.constants.reference_speed_of_sound
        pairs = self._pairs

        # A symmetric Hann window without its zero end points.
        window = np.hanning(frame + 2)[1:-1]
        max_delay = int(np.ceil(self._spacing / c * fs)) + 1
        centre = frame // 2
        region = np.arange(centre - max_delay, centre + max_delay + 2)

        n_samples = data.shape[0]
        n_frames = n_samples - frame
        positions = np.full((n_samples, 3), np.nan)

        # Frames per batch; it bounds memory only, not the result.
        batch = 8192
        for start in range(0, n_frames, batch):
            frames = np.arange(start, min(start + batch, n_frames))
            segments = data[frames[:, None] + np.arange(frame)]  # (F, N, M)
            spectra = fft(window[None, :, None] * segments, axis=1)
            cross = spectra[:, :, pairs[:, 0]] * np.conj(
                spectra[:, :, pairs[:, 1]])
            correlation = fftshift(np.real(ifft(cross, axis=1)), axes=1)
            # Only positive correlation is accepted.
            correlation = np.maximum(correlation, 2 * np.finfo(float).eps)

            peak = (np.argmax(correlation[:, region, :], axis=1)
                    + region[0])                                  # (F, P)
            at_peak, before, after = (
                np.take_along_axis(correlation, peak[:, None, :] + shift,
                                   axis=1)[:, 0]
                for shift in (0, -1, 1))

            # Exponential interpolation of the peak over -1, 0, +1 samples.
            with np.errstate(divide='ignore', invalid='ignore'):
                fraction = (np.log(after) - np.log(before)) / (
                    4 * np.log(at_peak) - 2 * np.log(before)
                    - 2 * np.log(after))
                delay = (peak - centre + fraction) / fs
                direction = -(self._inverse @ delay.T)             # (3, F)
                direction /= np.linalg.norm(direction, axis=0)

            distance = (frames + centre + 1) / fs * c
            estimate = (direction * distance).T
            estimate[~np.all(np.isfinite(estimate), axis=1)] = np.nan
            positions[frames + centre] = estimate

        return positions


def _unit(vector):
    """Normalize the rows of `vector`; vanishing rows become NaN."""
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    return np.divide(vector, norm, out=np.full_like(vector, np.nan),
                     where=norm > 0)
