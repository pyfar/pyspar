"""Synthesizers turning an analysis result into loudspeaker or ear signals."""
from __future__ import annotations

import numpy as np
import pyfar as pf
from scipy.signal import sosfiltfilt
from spharpy import SphericalHarmonicSignal

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
    lfe_channels : int, sequence of int, optional
        Indices of the low-frequency effects (LFE) channels in `loudspeakers`.
        They are left out of the panning. The signals of the other
        loudspeakers are high-passed at `lfe_cutoff`, and the LFE channels
        receive what lies below it, see Notes. The default is ``None``, no
        LFE channels.
    lfe_cutoff : float, optional
        Crossover frequency of the LFE channels in Hz. The default is ``80``.

    Attributes
    ----------
    loudspeakers : pyfar.Coordinates
        Loudspeaker directions.
    panning : str
        Panning law in use.
    dimensionality : int
        Dimensionality of the layout.
    lfe_channels : tuple of int
        Indices of the LFE channels.
    lfe_cutoff : float
        Crossover frequency of the LFE channels in Hz.

    Notes
    -----
    With `lfe_channels`, the loudspeaker signals pass a crossover after the
    panning: third-order Butterworth low-pass and high-pass filters at
    `lfe_cutoff`, applied forward and backward. The signals of the other
    loudspeakers are high-passed, and their low-passed signals are added to the
    LFE channels: a loudspeaker on the left to the LFE channels on the left, in
    equal shares, a loudspeaker on the right to those on the right, and a
    loudspeaker in the median plane half to each side. An LFE channel in the
    median plane counts for both sides, and a side without an LFE channel is
    served by all of them. Left and right follow from the positions in
    `loudspeakers`. The filters have zero phase and add up to one, so all
    channels together still add up to the signals before the crossover. In
    return, the low frequencies of an arrival begin before it, and a response
    that does not begin and end in silence is filtered inexactly at that end.

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

    def __init__(self, loudspeakers, panning='nls', dimensionality=3,
                 lfe_channels=None, lfe_cutoff=80):
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
        lfe_channels = _check_lfe_channels(lfe_channels, loudspeakers.csize)
        if (isinstance(lfe_cutoff, bool)
                or not isinstance(lfe_cutoff, (int, float, np.number))):
            raise TypeError(
                "lfe_cutoff must be a number; got "
                f"{type(lfe_cutoff).__name__}.")
        if not np.isfinite(lfe_cutoff) or lfe_cutoff <= 0:
            raise ValueError(
                f"lfe_cutoff is {lfe_cutoff} but must be positive and "
                "finite.")

        self.loudspeakers = loudspeakers
        self.panning = panning
        self.dimensionality = dimensionality
        self.lfe_channels = lfe_channels
        self.lfe_cutoff = float(lfe_cutoff)

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

        # The LFE channels are left out of the panning.
        layout = np.atleast_2d(self.loudspeakers.cartesian).reshape(-1, 3)
        panned = np.setdiff1d(np.arange(len(layout)), self.lfe_channels)
        data = np.zeros((n_samples, len(layout)))

        if self.dimensionality == 2:
            elevation = np.arcsin(np.clip(directions[:, 2], -1, 1))
            pressure = pressure * np.cos(elevation)
        index = nearest_directions(pf.Coordinates(*layout[panned].T), doa)
        data[np.arange(n_samples), panned[index]] = pressure

        signal = pf.Signal(data.T, result.sampling_rate)
        if self.lfe_channels:
            signal = _lfe_crossover(
                signal, self.loudspeakers, self.lfe_channels,
                self.lfe_cutoff)
        return signal


def _check_lfe_channels(lfe_channels, n_loudspeakers):
    """Return the indices of the LFE channels as a sorted tuple."""
    if lfe_channels is None or np.size(lfe_channels) == 0:
        return ()
    channels = np.atleast_1d(lfe_channels)
    if (channels.ndim != 1 or channels.dtype.kind not in 'iu'
            or np.any(channels < 0) or np.any(channels >= n_loudspeakers)
            or len(np.unique(channels)) != len(channels)):
        raise ValueError(
            f"lfe_channels is {lfe_channels!r} but must hold distinct "
            f"indices of the {n_loudspeakers} loudspeakers.")
    if len(channels) == n_loudspeakers:
        raise ValueError(
            "lfe_channels names every loudspeaker, which leaves none to pan "
            "to.")
    return tuple(int(channel) for channel in np.sort(channels))


def _lfe_shares(layout, lfe_channels):
    """Share of the low frequencies of each loudspeaker per LFE channel.

    A loudspeaker on the left is shared equally among the LFE channels on the
    left, one on the right among those on the right, and one in the median
    plane goes half to each side. An LFE channel in the median plane counts
    for both sides, and a side without an LFE channel is served by all.

    Parameters
    ----------
    layout : numpy.ndarray, shape (n_loudspeakers, 3)
        Cartesian positions of all loudspeakers, the LFE channels included.
    lfe_channels : tuple of int
        Indices of the LFE channels.

    Returns
    -------
    shares : numpy.ndarray, shape (n_loudspeakers - n_lfe, n_lfe)
        One row per loudspeaker that is not an LFE channel; every row adds up
        to one.
    """
    y = layout[:, 1] / np.linalg.norm(layout, axis=1)
    # 1 on the left, -1 on the right and 0 in the median plane.
    side = np.sign(np.where(np.abs(y) <= 1e-9, 0.0, y))
    lfe_side = side[list(lfe_channels)]

    def served_by(sign):
        serving = (lfe_side == sign) | (lfe_side == 0)
        if not np.any(serving):
            serving = np.ones(len(lfe_channels), dtype=bool)
        return serving / np.sum(serving)

    left, right = served_by(1), served_by(-1)
    shares = {1: left, -1: right, 0: (left + right) / 2}
    others = np.setdiff1d(np.arange(len(layout)), lfe_channels)
    return np.array([shares[int(s)] for s in side[others]])


def _lfe_crossover(signal, loudspeakers, lfe_channels, cutoff):
    """Move the low frequencies of loudspeaker signals to the LFE channels.

    The crossover described in the notes of :class:`LoudspeakerSynthesizer`.

    Parameters
    ----------
    signal : pyfar.Signal
        Loudspeaker signals, ``cshape (n_loudspeakers,)``, silent in the LFE
        channels.
    loudspeakers : pyfar.Coordinates
        Positions of all loudspeakers, the LFE channels included.
    lfe_channels : tuple of int
        Indices of the LFE channels.
    cutoff : float
        Crossover frequency in Hz.

    Returns
    -------
    signal : pyfar.Signal
        High-passed loudspeaker signals and the signals of the LFE channels.
    """
    sampling_rate = signal.sampling_rate
    if cutoff >= sampling_rate / 2:
        raise ValueError(
            f"lfe_cutoff is {cutoff} Hz but must be below half the sampling "
            f"rate, {sampling_rate / 2} Hz.")
    low, high = (
        pf.dsp.filter.butterworth(
            None, 3, cutoff, btype, sampling_rate).coefficients[0]
        for btype in ('lowpass', 'highpass'))

    layout = np.atleast_2d(loudspeakers.cartesian).reshape(-1, 3)
    lfe = list(lfe_channels)
    others = np.setdiff1d(np.arange(len(layout)), lfe)
    data = np.asarray(signal.time, dtype=float).T
    # The signals are extended by 9 samples at both ends, three times the
    # filter order.
    padding = min(9, len(data) - 1)
    split = np.zeros_like(data)
    split[:, others] = sosfiltfilt(
        high, data[:, others], axis=0, padlen=padding)
    split[:, lfe] = sosfiltfilt(
        low, data[:, others], axis=0, padlen=padding) @ _lfe_shares(
            layout, lfe_channels)
    return pf.Signal(split.T, sampling_rate)


class BinauralSynthesizer(SynthesizerBase):
    """Render an analysis result to binaural room impulse responses.

    The analysis is rendered, as by :class:`LoudspeakerSynthesizer`, to the
    measurement grid of the head-related impulse responses, and each
    signal is convolved with the head-related impulse response of its
    direction; the ear signals are the sums. `virtual_loudspeakers`
    restrict the rendering to a subset of the grid: the analysis is
    rendered to the virtual loudspeakers, and each takes the head-related
    impulse response measured nearest to it.

    Parameters
    ----------
    hrirs : pyfar.Signal
        Head-related impulse responses with ``cshape (n_directions, 2)``, the
        left ear first, at the sampling rate of the analyses to be rendered.
        A measured set is read from a SOFA file with
        :py:func:`pyfar.io.read_sofa`, which also returns `hrir_directions`
        as its second output (the source positions).
    hrir_directions : pyfar.Coordinates
        Measurement directions of `hrirs`, ``cshape (n_directions,)``, the
        source positions of the SOFA file.
    virtual_loudspeakers : pyfar.Coordinates, optional
        Loudspeaker layout that selects a subset of the measurement grid:
        the analysis is rendered to these directions, and each takes the
        head-related impulse response measured nearest to it, matched by
        angle alone. ``None``, the default, uses no virtual loudspeakers
        and renders on the whole grid. A grid of fewer, evenly spread
        directions, for example a Lebedev grid, quantizes the directions of
        the analysis to its points, see Examples. A layout of a listening
        room is passed without its LFE channels, which would otherwise be
        rendered as loudspeakers at their positions.
    panning : ``'nls'``, optional
        Panning law, see :class:`LoudspeakerSynthesizer`. The default is
        ``'nls'``.
    dimensionality : ``3``, ``2``, optional
        Dimensionality of the rendering, see
        :class:`LoudspeakerSynthesizer`. The default is ``3``.

    Attributes
    ----------
    virtual_loudspeakers : pyfar.Coordinates or None
        The layout given, or ``None``.
    panning : str
        Panning law in use.
    dimensionality : int
        Dimensionality of the rendering.

    Notes
    -----
    The ear signals have ``n_samples + n_hrir - 1`` samples, the full linear
    convolution, so the tail the head-related impulse responses add to the
    last samples is kept.

    Examples
    --------
    Measured head-related impulse responses are read with
    :py:func:`pyfar.io.read_sofa`.

    >>> import numpy as np
    >>> import pyfar as pf
    >>> import pyspar
    >>> from spharpy import SphericalHarmonicSignal
    >>> signal = SphericalHarmonicSignal(
    ...     np.random.default_rng(0).standard_normal((4, 256)), 48000,
    ...     'real', 'N3D', 'ACN', False)
    >>> directions = pf.Coordinates.from_spherical_elevation(
    ...     np.deg2rad([0, 90, 180, 270, 0, 0]),
    ...     np.deg2rad([0, 0, 0, 0, 90, -90]), 1)
    >>> hrirs = pf.Signal(np.zeros((6, 2, 32)), 48000)
    >>> result = pyspar.parametric.PseudoIntensityVectorAnalyzer()(signal)
    >>> brir = pyspar.parametric.BinauralSynthesizer(hrirs, directions)(result)
    >>> brir.cshape, brir.n_samples
    ((2,), 287)

    The directions of the analysis can be quantized to a smaller grid, for
    example the 50 points of a Lebedev grid. spharpy asks for the spherical
    harmonics order ``n_max`` of the grid, not for a number of points: the
    orders 1 to 8 give 6, 14, 26, 38, 50, 74, 86 and 110 points.

    >>> from spharpy.samplings import lebedev
    >>> grid = lebedev(n_max=5)
    >>> grid.csize
    50
    >>> synthesizer = pyspar.parametric.BinauralSynthesizer(
    ...     hrirs, directions, virtual_loudspeakers=grid)
    >>> synthesizer(result).cshape
    (2,)
    """

    def __init__(self, hrirs, hrir_directions, virtual_loudspeakers=None,
                 panning='nls', dimensionality=3):
        if (not isinstance(hrirs, pf.Signal)
                or isinstance(hrirs, SphericalHarmonicSignal)):
            raise TypeError(
                "hrirs must be a pyfar.Signal; got "
                f"{type(hrirs).__name__}.")
        if len(hrirs.cshape) != 2 or hrirs.cshape[1] != 2:
            raise ValueError(
                "hrirs must have cshape (n_directions, 2), got "
                f"{hrirs.cshape}.")
        if not isinstance(hrir_directions, pf.Coordinates):
            raise TypeError(
                "hrir_directions must be a pyfar.Coordinates; got "
                f"{type(hrir_directions).__name__}.")
        if hrir_directions.cshape != (hrirs.cshape[0],):
            raise ValueError(
                f"hrirs holds {hrirs.cshape[0]} directions but "
                f"hrir_directions has cshape {hrir_directions.cshape}.")
        if virtual_loudspeakers is not None and not isinstance(
                virtual_loudspeakers, pf.Coordinates):
            raise TypeError(
                "virtual_loudspeakers must be a pyfar.Coordinates or None; "
                f"got {type(virtual_loudspeakers).__name__}.")
        for name, directions in (("hrir_directions", hrir_directions),
                                 ("virtual_loudspeakers",
                                  virtual_loudspeakers)):
            if directions is not None and np.any(
                    ~(np.atleast_1d(directions.radius) > 0)):
                raise ValueError(f"{name} must have a nonzero radius.")

        self._virtual_loudspeakers = virtual_loudspeakers
        measured = np.asarray(hrirs.time, dtype=float)        # (Q, 2, M)
        if virtual_loudspeakers is None:
            virtual_loudspeakers = hrir_directions
        else:
            measured = measured[nearest_directions(hrir_directions,
                                                   virtual_loudspeakers)]

        self._loudspeaker_synthesizer = LoudspeakerSynthesizer(
            virtual_loudspeakers, panning=panning,
            dimensionality=dimensionality)
        self._hrirs = pf.Signal(measured, hrirs.sampling_rate)  # (L, 2)
        self._sampling_rate = float(hrirs.sampling_rate)

    @property
    def virtual_loudspeakers(self):
        """The layout given, or ``None``."""
        return self._virtual_loudspeakers

    @property
    def panning(self):
        """Panning law in use."""
        return self._loudspeaker_synthesizer.panning

    @property
    def dimensionality(self):
        """Dimensionality of the rendering."""
        return self._loudspeaker_synthesizer.dimensionality

    def render_loudspeakers(self, result: TimeResult) -> pf.Signal:
        """Render `result` to the loudspeaker signals only.

        Together with :meth:`convolve`, this lets a correction act on the
        loudspeaker signals between the two stages.

        Parameters
        ----------
        result : pyspar.parametric.TimeResult
            Pressure and direction of arrival per sample.

        Returns
        -------
        loudspeaker_signals : pyfar.Signal
            Loudspeaker signals, one per virtual loudspeaker or, without
            them, per measured direction, ``cshape (n_loudspeakers,)``.
        """
        return self._loudspeaker_synthesizer(result)

    def convolve(self, loudspeaker_signals) -> pf.Signal:
        """Convolve loudspeaker signals to ear signals.

        Parameters
        ----------
        loudspeaker_signals : pyfar.Signal
            Loudspeaker signals as returned by :meth:`render_loudspeakers`,
            ``cshape (n_loudspeakers,)``, at the sampling rate of the
            head-related impulse responses.

        Returns
        -------
        ear_signals : pyfar.Signal
            Left and right ear signals, ``cshape (2,)``.
        """
        n_loudspeakers = self._hrirs.cshape[0]
        if (not isinstance(loudspeaker_signals, pf.Signal)
                or isinstance(loudspeaker_signals, SphericalHarmonicSignal)):
            raise TypeError(
                "loudspeaker_signals must be a pyfar.Signal; got "
                f"{type(loudspeaker_signals).__name__}.")
        if loudspeaker_signals.cshape != (n_loudspeakers,):
            raise ValueError(
                f"loudspeaker_signals must have cshape ({n_loudspeakers},), "
                "as returned by render_loudspeakers; got "
                f"{loudspeaker_signals.cshape}.")
        if abs(loudspeaker_signals.sampling_rate - self._sampling_rate) > 0.5:
            raise ValueError(
                f"The signal sampling_rate "
                f"({loudspeaker_signals.sampling_rate} Hz) does not match "
                f"the hrirs ({self._sampling_rate} Hz); resample the hrirs "
                "with pyfar.dsp.resample first.")

        feeds = pf.Signal(loudspeaker_signals.time[:, np.newaxis],
                          self._sampling_rate)
        ears = pf.dsp.convolve(feeds, self._hrirs, mode='full')
        return pf.Signal(ears.time.sum(axis=0), self._sampling_rate)

    def __call__(self, result: TimeResult) -> pf.Signal:
        """Render `result` to left and right ear signals.

        Parameters
        ----------
        result : pyspar.parametric.TimeResult
            Pressure and direction of arrival per sample, at the sampling
            rate of the head-related impulse responses.

        Returns
        -------
        ear_signals : pyfar.Signal
            Left and right ear signals, ``cshape (2,)``.
        """
        return self.convolve(self.render_loudspeakers(result))
