"""Synthesizers turning an analysis result into loudspeaker signals."""
from __future__ import annotations

import numpy as np
import pyfar as pf

from .classes import SynthesizerBase, TimeResult
from pyspar.utils.directions import _replace_missing_doa, nearest_directions


class LoudspeakerSynthesizer(SynthesizerBase):
    """Render an analysis result to a loudspeaker array.

    Each sample of the pressure is reproduced from the direction estimated for
    it, as in the Spatial Decomposition Method [#]_.

    Parameters
    ----------
    loudspeakers : pyfar.Coordinates
        Loudspeaker directions, one per channel of the output. The radius is
        ignored but must not be zero.
    panning : ``'nls'``, optional
        ``'nls'`` (default) assigns each sample to the nearest loudspeaker.
    dimensionality : ``3``, ``2``, optional
        With ``2``, intended for horizontal layouts, the pressure of each
        sample is weighted by the cosine of its elevation, so energy arriving
        from above or below is attenuated instead of folded onto the horizon.
        The default is ``3``.

    Attributes
    ----------
    loudspeakers : pyfar.Coordinates
        Loudspeaker directions.
    panning : str
        Panning law in use.
    dimensionality : int
        Dimensionality of the layout.

    References
    ----------
    .. [#] S. Tervo, J. Patynen, A. Kuusinen and T. Lokki, "Spatial
           decomposition method for room impulse responses," J. Audio Eng.
           Soc., vol. 61, no. 1/2, pp. 17-28, 2013.

    Examples
    --------
    Render a measured impulse response to a 5.0 layout.

    >>> import numpy as np
    >>> import pyfar as pf
    >>> import pyspar
    >>> from spharpy import SphericalHarmonicSignal
    >>> signal = SphericalHarmonicSignal(
    ...     np.random.default_rng(0).standard_normal((4, 256)), 48000,
    ...     'real', 'N3D', 'ACN', False)
    >>> loudspeakers = pf.Coordinates.from_spherical_elevation(
    ...     np.deg2rad([0, 30, -30, 110, -110]), 0, 1)
    >>> result = pyspar.parametric.PseudoIntensityVectorAnalyzer()(signal)
    >>> pyspar.parametric.LoudspeakerSynthesizer(loudspeakers)(result).cshape
    (5,)
    """

    def __init__(self, loudspeakers, panning='nls', dimensionality=3):
        if not isinstance(loudspeakers, pf.Coordinates):
            raise TypeError(
                "loudspeakers must be a pyfar.Coordinates; got "
                f"{type(loudspeakers).__name__}.")
        if loudspeakers.csize < 1:
            raise ValueError("loudspeakers must hold at least one direction.")
        if np.any(~(np.atleast_1d(loudspeakers.radius) > 0)):
            raise ValueError("loudspeakers must have a nonzero radius.")
        if panning != 'nls':
            raise ValueError(
                f"panning is '{panning}' but must be 'nls'.")
        if dimensionality not in (2, 3):
            raise ValueError(
                f"dimensionality is {dimensionality} but must be 2 or 3.")

        self.loudspeakers = loudspeakers
        self.panning = panning
        self.dimensionality = dimensionality

    def __call__(self, result: TimeResult) -> pf.Signal:
        """Render `result` to loudspeaker signals.

        Parameters
        ----------
        result : pyspar.parametric.TimeResult
            Pressure and direction of arrival per sample.

        Returns
        -------
        signal : pyfar.Signal
            Loudspeaker signals with ``cshape (n_loudspeakers,)``, in the order
            of `loudspeakers`.
        """
        if not isinstance(result, TimeResult):
            raise TypeError(
                "LoudspeakerSynthesizer requires a "
                "pyspar.parametric.TimeResult; got "
                f"{type(result).__name__}.")

        # TimeResult guarantees one direction per pressure sample.
        pressure = result.pressure.time[0]
        directions = _replace_missing_doa(result.doa, pressure)

        doa = pf.Coordinates(directions[:, 0], directions[:, 1],
                             directions[:, 2])
        n_samples = len(pressure)

        layout = np.atleast_2d(self.loudspeakers.cartesian).reshape(-1, 3)
        if self.dimensionality == 2:
            elevation = np.arcsin(np.clip(directions[:, 2], -1, 1))
            pressure = pressure * np.cos(elevation)
        index = nearest_directions(self.loudspeakers, doa)
        data = np.zeros((n_samples, len(layout)))
        data[np.arange(n_samples), index] = pressure

        return pf.Signal(data.T, result.sampling_rate)
