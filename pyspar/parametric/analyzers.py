"""Analyzers estimating spatial parameters from measured signals."""
from __future__ import annotations

import warnings

import numpy as np
import pyfar as pf
from scipy.ndimage import convolve1d
from spharpy import SphericalHarmonicSignal

from .classes import AnalyzerBase, TimeResult
from pyspar.utils.spherical_harmonics import _check_sh_input


class PseudoIntensityVectorAnalyzer(AnalyzerBase):
    """Estimate directions of arrival by the pseudo-intensity vector.

    Estimate a direction of arrival for every sample of a first-order spherical
    harmonic signal, as the spatial decomposition method does [#]_. The
    direction is that of the pseudo-intensity vector, the product of the
    omnidirectional channel with each of the dipole channels, smoothed by a
    centred Hann window and normalized.

    Parameters
    ----------
    window_length : int or None
        Length in samples of the window smoothing the vector, centred on
        each sample, an odd number of at least 3. ``None`` does not smooth:
        every sample takes the direction of its own vector.

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
    >>> analyzer = pyspar.parametric.PseudoIntensityVectorAnalyzer(
    ...     window_length=5)
    >>> result = analyzer(signal)
    >>> np.round(result.doa.cartesian[8], 12)
    array([0., 1., 0.])
    """

    def __init__(self, window_length):
        if isinstance(window_length, (int, np.integer)):
            if window_length < 3:
                raise ValueError(
                    f"window_length is {window_length} but must be at "
                    "least 3.")
            if window_length % 2 == 0:
                raise ValueError(
                    f"window_length is {window_length} but must be odd, so "
                    "that the window is centred.")
        elif window_length is not None:
            raise TypeError(
                "window_length must be an integer or None; got "
                f"{type(window_length).__name__}.")

        self._window_length = window_length

    @property
    def window_length(self):
        """Get the window length in samples, as given."""
        return self._window_length

    def __call__(self, signal) -> TimeResult:
        """Analyze `signal` and return an analysis result.

        Parameters
        ----------
        signal : spharpy.SphericalHarmonicSignal
            Real spherical harmonic signal of order one or higher, in any
            normalization and channel convention, without the Condon-Shortley
            phase; orders above one are truncated to first order with a
            warning.

        Returns
        -------
        result : pyspar.parametric.TimeResult
            Pressure and direction of arrival per sample, and the first-order
            signal in ACN/N3D. Samples without a direction are NaN.
        """
        _check_sh_input(signal, type(self).__name__, 1)
        if signal.n_max > 1:
            warnings.warn(
                f"signal is of order {signal.n_max}. The pseudo-intensity "
                "vector uses first order only, the remaining channels are "
                "ignored.", stacklevel=2)
        n3d = signal.copy()
        n3d.channel_convention = 'ACN'
        n3d.normalization = 'N3D'
        data = np.asarray(n3d.time, dtype=float).reshape(
            -1, n3d.n_samples)[:4].T

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
            n = self.window_length + 2
            window = pf.dsp.time_window(pf.Signal(
                np.ones(n), 1), (0, n - 1)).time[0, 1:-1]
            # Direct, not FFT, convolution: a vanishing vector stays
            # exactly zero.
            vector = convolve1d(vector, window, axis=0, mode='constant')
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        return np.divide(vector, norm, out=np.full_like(vector, np.nan),
                         where=norm > 0)
