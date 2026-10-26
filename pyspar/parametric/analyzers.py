"""Spatial analyzers estimating a direction of arrival per sample."""
from __future__ import annotations

import warnings

import numpy as np
import pyfar as pf
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
    before it is normalized, see Notes.

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


def _unit(vector):
    """Normalize the rows of `vector`; vanishing rows become NaN."""
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    return np.divide(vector, norm, out=np.full_like(vector, np.nan),
                     where=norm > 0)
