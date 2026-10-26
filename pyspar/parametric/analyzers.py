"""Analyzers estimating spatial parameters from measured signals."""
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

    Assigns a direction of arrival to every sample of the omnidirectional
    channel of a first-order spherical harmonic signal, as the spatial
    decomposition method does [#]_. The direction is that of the
    pseudo-intensity vector, the product of the omnidirectional channel with
    each of the dipole channels, smoothed by a centred Hann window and
    normalized [#]_.

    Parameters
    ----------
    window_length : int or None
        Length in samples of the window smoothing the vector, centred on
        each sample, an odd number of at least 3. ``None`` does not smooth:
        every sample takes the direction of its own vector.

    Attributes
    ----------
    window_length : int or None
        Window length in samples, as given.

    Notes
    -----
    The Hann window has `window_length` samples without its zero end points and
    is applied to each Cartesian component. A sample whose smoothed vector
    vanishes has no direction.

    References
    ----------
    .. [#] S. Tervo, J. Patynen, A. Kuusinen and T. Lokki, "Spatial
           decomposition method for room impulse responses," J. Audio Eng.
           Soc., vol. 61, no. 1/2, pp. 17-28, 2013.
    .. [#] S. V. Amengual Gari, J. M. Arend, P. T. Calamia and P. W. Robinson,
           "Optimizations of the spatial decomposition method for binaural
           reproduction," J. Audio Eng. Soc., vol. 68, no. 12, pp. 959-976,
           2020.

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
    >>> result = pyspar.parametric.PseudoIntensityVectorAnalyzer(
    ...     window_length=5)(signal)
    >>> np.round(result.doa.cartesian[8], 12)
    array([0., 1., 0.])
    """

    def __init__(self, window_length):
        if window_length is not None:
            if (isinstance(window_length, bool)
                    or not isinstance(window_length, (int, np.integer))):
                raise TypeError(
                    "window_length must be an integer or None; got "
                    f"{type(window_length).__name__}.")
            if window_length < 3:
                raise ValueError(
                    f"window_length is {window_length} but must be at "
                    "least 3.")
            if window_length % 2 == 0:
                raise ValueError(
                    f"window_length is {window_length} but must be odd, so "
                    "that the window is centred.")

        self.window_length = (None if window_length is None
                              else int(window_length))

    def __call__(self, signal) -> TimeResult:
        """Analyze `signal` and return an analysis result.

        Parameters
        ----------
        signal : spharpy.SphericalHarmonicSignal
            Real spherical harmonic signal without the Condon-Shortley phase,
            of order one or higher, in any channel convention and
            normalization; orders above one are truncated to first order
            with a warning.

        Returns
        -------
        result : pyspar.parametric.TimeResult
            Pressure and direction of arrival per sample, and the first-order
            signal in ACN/N3D. Samples without a direction are NaN, or the
            zenith with a `passband`.
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
            signal is digitally silent within the window and, without
            smoothing, where the omnidirectional channel is exactly zero,
            which recordings of finite resolution reach near some of its
            zero crossings.
        """
        # ACN order is [W, Y, Z, X], the vector is [W X, W Y, W Z].
        vector = data[:, [3, 1, 2]] * data[:, :1]
        if self.window_length is not None:
            # Direct, not FFT, convolution: a vanishing vector stays
            # exactly zero.
            window = np.hanning(self.window_length + 2)[1:-1]
            vector = convolve1d(vector, window, axis=0, mode='constant')
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        return np.divide(vector, norm, out=np.full_like(vector, np.nan),
                         where=norm > 0)
