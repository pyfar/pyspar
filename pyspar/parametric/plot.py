"""Plots of spatial impulse response analyses.

The plots show a :class:`~pyspar.parametric.TimeResult`, the pressure and
the direction of arrival per sample of an impulse response: a map of the
directions of arrival, and the time-frequency and spatiotemporal
responses of Patynen et al. [#]_, integrated forward or backward, the
backward spatiotemporal response as in Tervo et al. [#]_.

References
----------
.. [#] J. Patynen, S. Tervo and T. Lokki, "Analysis of concert hall acoustics
       via visualizations of time-frequency and spatiotemporal
       responses," J. Acoust. Soc. Am., vol. 133, no. 2, pp. 842-857, 2013.
.. [#] S. Tervo, J. Patynen, N. Kaplanis, M. Lydolf, S. Bech and T. Lokki,
       "Spatial analysis and synthesis of car audio system and car cabin
       acoustics with a compact microphone array," J. Audio Eng. Soc.,
       vol. 63, no. 11, pp. 914-925, 2015.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pyfar as pf
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.transforms import Bbox

from pyspar.utils.plot import _map_axes, _project

from .classes import TimeResult

# Direction labels at 0, 90, 180 and 270 degrees of each plane.
_PLANES = {
    'lateral': ('Front', 'Left', 'Back', 'Right'),
    'median': ('Front', 'Top', 'Back', 'Bottom'),
    'transverse': ('Right', 'Top', 'Left', 'Bottom'),
}


def doa_map(result, time_window, dynamic_range=45, direct_sound=True,
            cmap=None, ax=None, style='light'):
    """Plot the directions of arrival in a time window on a map of the sphere.

    Every sample in `time_window` is drawn as a dot at its direction of
    arrival on a Hammer-Aitoff projection of the sphere, seen from the
    listener: the front is in the center and the left on the left. The dot
    grows with the level of the sample and has the color of the level curve
    below or, with `cmap`, the color of its level; the loudest dots are drawn
    on top. With `direct_sound`, the direction of arrival at the peak of the
    pressure is marked with a black cross. The level of the impulse response
    is shown below the map with the time window marked, up to the last sample
    within `dynamic_range` or to the end of `time_window` if later.

    Parameters
    ----------
    result : pyspar.parametric.TimeResult
        Pressure and direction of arrival per sample.
    time_window : (float, float)
        Start and end of the window in milliseconds after the onset of the
        direct sound, 1 ms before the first sample whose magnitude exceeds a
        fifth of the peak of the pressure and not before the start of the
        response.
    dynamic_range : float, optional
        Range in dB below the peak of the pressure that is shown: samples at
        the peak get the largest dots, samples at or below ``-dynamic_range``
        none. The default is ``45``.
    direct_sound : bool, optional
        Mark the direction of arrival at the peak of the pressure with a
        black cross, with a warning if that sample has no direction. The
        default is ``True``.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap that colors the dots by their level, from ``-dynamic_range``
        to 0 dB, with a colorbar beside the map, for example ``'turbo'``.
        ``None``, the default, gives all dots the color of the level curve.
    ax : array like of two matplotlib.axes.Axes, optional
        Axes for the map and for the level of the impulse response. The
        default ``None`` creates a new figure.
    style : str, optional
        Plot style, see :func:`pyfar.plot.context`. The default is
        ``'light'``.

    Returns
    -------
    ax : numpy.ndarray of matplotlib.axes.Axes
        The axes of the map and of the level of the impulse response and,
        with `cmap`, of the colorbar.

    Examples
    --------
    .. plot::

        >>> import numpy as np
        >>> import pyspar
        >>> from spharpy import SphericalHarmonicSignal
        >>> signal = SphericalHarmonicSignal(
        ...     np.random.default_rng(0).standard_normal((4, 2048)), 48000,
        ...     'real', 'N3D', 'ACN', False)
        >>> analyzer = pyspar.parametric.PseudoIntensityVectorAnalyzer(
        ...     window_length=5)
        >>> result = analyzer(signal)
        >>> ax = pyspar.parametric.plot.doa_map(result, (5, 20))
    """
    result, = _as_results(result, 'doa_map', single=True)
    time_window = np.asarray(time_window, dtype=float)
    if time_window.shape != (2,) or time_window[0] > time_window[1]:
        raise ValueError(
            "time_window must hold a start and an end in milliseconds, with "
            f"the start first; got {time_window.tolist()}.")
    _check_positive(dynamic_range, 'dynamic_range')
    if not isinstance(direct_sound, bool):
        raise TypeError(
            "direct_sound must be a bool; got "
            f"{type(direct_sound).__name__}.")

    pressure, directions, fs = _data(result)
    pressure = pressure / _peak(pressure)
    with np.errstate(divide='ignore'):
        level = np.maximum(20 * np.log10(np.abs(pressure)) + dynamic_range, 0)
    onset = _onset(pressure, fs)

    # Samples counted from one, from ceil(start fs) to ceil(end fs) after the
    # onset; in samples, so that round-off cannot move the edges.
    first, last = onset + np.ceil(time_window / 1000 * fs).astype(int)
    time_window = onset / fs * 1000 + time_window
    samples = np.arange(max(first, 1) - 1, min(last, len(pressure)))
    samples = samples[(level[samples] > 0)
                      & np.all(np.isfinite(directions[samples]), axis=1)]
    centers = _project(directions[samples])
    radii = (level[samples] / dynamic_range) ** 1.5 * 0.035

    with pf.plot.context(style):
        if ax is None:
            _, ax = plt.subplots(2, 1, height_ratios=(2, 1))
        ax = np.asarray(ax)
        if ax.shape != (2,):
            raise ValueError(
                "ax must hold two axes, for the map and for the level of "
                f"the impulse response; got {ax.size} axes.")

        _map_axes(ax[0])
        ax[0].set_anchor('S')
        # Room below the sphere for the legend.
        ax[0].set_ylim(-0.64, 0.55)
        if cmap is None:
            # The color pf.plot.time gives the level curve below.
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
            colors = [color] * len(radii)
        else:
            mappable = ScalarMappable(Normalize(-dynamic_range, 0), cmap)
            colors = mappable.to_rgba(level[samples] - dynamic_range)
        for center, radius, color, dot_level in zip(
                centers, radii, colors, level[samples], strict=True):
            # The loudest dots on top, all below the cross.
            ax[0].add_patch(Circle(center, radius, facecolor=color,
                                   alpha=0.8, edgecolor='none',
                                   zorder=1 + dot_level / dynamic_range))
        peak = np.argmax(np.abs(pressure))
        if direct_sound and np.all(np.isfinite(directions[peak])):
            cross, = ax[0].plot(*_project(directions[[peak]]).T, 'x',
                                color='k', markersize=10, markeredgewidth=2,
                                zorder=3, label='Direct sound')
            # Slightly below the map; left out of the layout, so the
            # panels keep their place.
            legend = ax[0].legend(handles=[cross], loc='lower right',
                                  bbox_to_anchor=(1, -0.03), borderaxespad=0,
                                  fontsize=8)
            legend.set_in_layout(False)
        elif direct_sound:
            warnings.warn(
                "The direction at the peak of the pressure is missing; the "
                "direct sound is not marked.", stacklevel=2)

        with np.errstate(divide='ignore'):
            pf.plot.time(pf.Signal(pressure, fs), dB=True, unit='ms',
                         ax=ax[1], style=style)
        ax[1].axvspan(*time_window, color='0.8', zorder=0)
        ax[1].set_ylim(-dynamic_range, 0)
        end = max(np.flatnonzero(level > 0)[-1] / fs * 1000, time_window[1])
        ax[1].set_xlim(0, 1.05 * end)
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_ylabel('Amplitude (dB)')
        if cmap is not None:
            # Width from both panels, so that they stay aligned; height from
            # pole to pole of the map, set whenever the figure is drawn.
            bar = ax[0].figure.colorbar(mappable, ax=list(ax),
                                        label='Level (dB)')
            bar.ax.set_axes_locator(
                lambda cax, _renderer, map_ax=ax[0]: _beside_the_sphere(
                    cax, map_ax))
            ax = np.append(ax, bar.ax)

    return ax



def _beside_the_sphere(cax, map_ax):
    """Box of `cax` in its layout slot, from pole to pole of `map_ax`."""
    map_ax.apply_aspect()
    to_figure = map_ax.transData + map_ax.figure.transFigure.inverted()
    (_, bottom), (_, top) = to_figure.transform([(0, -0.5), (0, 0.5)])
    slot = cax.get_position(original=True)
    return Bbox.from_bounds(slot.x0, bottom, slot.width, top - bottom)


def spatio_temporal(result, plane='lateral', integration='forward',
                    times=(5, 10, 15, 20, 50, 80, 200, 2000), resolution=1,
                    dynamic_range=45, level_spacing=6, angle_spacing=30,
                    smoothing=3, fill=False, colors=None, ax=None,
                    style='light', **kwargs):
    """Plot the directional energy in cumulative time windows in a plane.

    For each of `times`, the energy of the pressure in a rectangular time
    window, forward from the direct sound to `times` or backward from
    `times` to the end of the response, is collected in direction bins by
    the angle of its direction of arrival in `plane` [#]_, weighted by the
    absolute cosine of the angle to the plane [#]_, smoothed by a moving
    average over the angle, and drawn as a polar curve in dB relative to the
    largest value of all curves. The direct sound starts 1 ms before the
    first sample whose magnitude exceeds a fifth of the peak of the
    pressure and not before the start of the response, for each result
    separately.

    Parameters
    ----------
    result : pyspar.parametric.TimeResult or sequence
        Pressure and direction of arrival per sample. The energies of
        several results, for example of several source positions, are
        summed.
    plane : ``'lateral'``, ``'median'``, ``'transverse'``, optional
        Plane of the plot: the horizontal plane, the vertical plane through
        the front, or the vertical plane through the ears. The default is
        ``'lateral'``.
    integration : ``'forward'``, ``'backward'``, optional
        ``'forward'`` (default) integrates from the direct sound to each of
        `times`, ``'backward'`` from each of `times` to the end of the
        response.
    times : array like, optional
        Limits of the time windows in milliseconds after the direct sound.
        The default is ``(5, 10, 15, 20, 50, 80, 200, 2000)``.
    resolution : float, optional
        Width of the direction bins in degrees; it must divide 180. The
        default is ``1``.
    dynamic_range : float, optional
        Range of the radial axis in dB. The default is ``45``.
    level_spacing : float, array like, optional
        Spacing of the level grid in dB, or the levels to mark in dB relative
        to the maximum; the centre of the plot, ``-dynamic_range``, is not
        marked. The default is ``6``.
    angle_spacing : float, optional
        Spacing of the angular grid in degrees. The default is ``30``.
    smoothing : float, optional
        Width in degrees of the moving average applied to each curve around
        the circle. The default is ``3``.
    fill : bool, optional
        Fill the curves instead of drawing lines. The default is ``False``.
    colors : array like, optional
        One color per entry of `times`. The default takes them from the
        ``'jet'`` colormap.
    ax : matplotlib.axes.Axes, optional
        Polar axes to plot on. The default ``None`` creates a new figure.
    style : str, optional
        Plot style, see :func:`pyfar.plot.context`. The default is
        ``'light'``.
    **kwargs
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.plot`, or
        to :meth:`matplotlib.axes.Axes.fill` if `fill` is ``True``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The polar axes. Its lines, or polygons if `fill` is ``True``, are in
        drawing order, the longest window first.

    References
    ----------
    .. [#] J. Patynen, S. Tervo and T. Lokki, "Analysis of concert hall
           acoustics via visualizations of time-frequency and
           spatiotemporal responses," J. Acoust. Soc. Am., vol. 133, no. 2,
           pp. 842-857, 2013.
    .. [#] S. Tervo, J. Patynen, N. Kaplanis, M. Lydolf, S. Bech and T. Lokki,
           "Spatial analysis and synthesis of car audio system and car cabin
           acoustics with a compact microphone array," J. Audio Eng. Soc., vol.
           63, no. 11, pp. 914-925, 2015.

    Examples
    --------
    .. plot::

        >>> import numpy as np
        >>> import pyspar
        >>> from spharpy import SphericalHarmonicSignal
        >>> signal = SphericalHarmonicSignal(
        ...     np.random.default_rng(0).standard_normal((4, 2048)), 48000,
        ...     'real', 'N3D', 'ACN', False)
        >>> analyzer = pyspar.parametric.PseudoIntensityVectorAnalyzer(
        ...     window_length=5)
        >>> result = analyzer(signal)
        >>> ax = pyspar.parametric.plot.spatio_temporal(
        ...     result, times=(5, 10, 20))
    """
    results = _as_results(result, 'spatio_temporal')
    if plane not in _PLANES:
        raise ValueError(
            f"plane is '{plane}' but must be one of {tuple(_PLANES)}.")
    times = _check_plot_arguments(integration, times, dynamic_range)
    _check_positive(resolution, 'resolution')
    if abs(round(180 / resolution) - 180 / resolution) > 1e-9:
        raise ValueError(
            f"resolution is {resolution} but must divide 180 degrees.")
    _check_positive(smoothing, 'smoothing')

    angles = np.arange(round(360 / resolution)) * resolution - 180
    energy = np.zeros((len(angles), len(times)))
    for single in results:
        pressure, directions, fs = _data(single)
        for k, time in enumerate(times):
            window = _window(pressure, time, fs, integration)
            energy[:, k] += _circular_smooth(
                _polar_response(directions[window], pressure[window],
                                resolution, plane),
                int(np.ceil(smoothing / resolution)))
    with np.errstate(divide='ignore'):
        level = np.clip(
            10 * np.log10(energy) - 10 * np.log10(np.max(energy)),
            -dynamic_range, 0)

    with pf.plot.context(style):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        if ax.name != 'polar':
            raise ValueError("ax must be a polar axes.")

        theta = np.deg2rad(angles)
        handles = _draw_windows(
            lambda k, color: (
                ax.fill(theta, level[:, k], color=color,
                        **{'alpha': 0.5, **kwargs})[0] if fill else
                ax.plot(np.append(theta, theta[0]),
                        np.append(level[:, k], level[0, k]), color=color,
                        **kwargs)[0]),
            times, integration, colors)

        ax.set_rlim(-dynamic_range, 0)
        # The centre is -dynamic_range; a grid circle there has no radius.
        levels, labels = _level_grid(level_spacing, dynamic_range)
        shown = levels > -dynamic_range
        ax.set_rgrids(levels[shown], np.array(labels)[shown], angle=75)
        # The names of the four directions are part of the tick labels, so
        # that a title is placed above them.
        names = dict(zip((0, 90, 180, 270), _PLANES[plane], strict=True))
        placement = {0: '{degree} {name}', 90: '{name}\n{degree}',
                     180: '{name} {degree}', 270: '{degree}\n{name}'}
        grid = np.arange(0, 360, angle_spacing)
        labels = []
        for angle in grid:
            degree = f"{(angle - 360 if angle > 180 else angle):g}$^\\circ$"
            labels.append(placement[angle].format(
                degree=degree, name=names[angle]) if angle in names
                else degree)
        ax.set_thetagrids(grid, labels)
        ax.legend(handles, _labels(times, integration, results, fs),
                  loc='upper left', bbox_to_anchor=(1.1, 1))

    return ax


def time_frequency(result, integration='forward',
                   times=(5, 10, 15, 20, 50, 80, 200, 2000),
                   dynamic_range=45, level_spacing=6, num_fractions=3,
                   fill=False, colors=None, ax=None, style='light',
                   **kwargs):
    """Plot the spectrum of the pressure in cumulative time windows.

    For each of `times`, the pressure in a rectangular time window, forward
    from the direct sound to `times` or backward from `times` to the end of
    the response, is transformed by a discrete Fourier transform with the
    length of the first result. Its energy spectrum is smoothed in
    fractional octaves and drawn in dB relative to the largest value of all
    curves [#]_. The direct sound starts 1 ms before the first sample whose
    magnitude exceeds a fifth of the peak of the pressure and not before the
    start of the response, for each result separately.

    Parameters
    ----------
    result : pyspar.parametric.TimeResult or sequence
        Pressure per sample; the directions are not used. The energies of
        several results, for example of several source positions, are
        summed.
    integration : ``'forward'``, ``'backward'``, optional
        ``'forward'`` (default) integrates from the direct sound to each of
        `times`, ``'backward'`` from each of `times` to the end of the
        response.
    times : array like, optional
        Limits of the time windows in milliseconds after the direct sound.
        The default is ``(5, 10, 15, 20, 50, 80, 200, 2000)``.
    dynamic_range : float, optional
        Range of the level axis in dB. The default is ``45``.
    level_spacing : float, array like, optional
        Spacing of the level grid in dB, or the levels to mark in dB
        relative to the maximum. The default is ``6``.
    num_fractions : int, optional
        Width of the smoothing as a fraction of an octave, ``3`` for third
        octaves. The default is ``3``.
    fill : bool, optional
        Fill the area below each curve. The default is ``False``.
    colors : array like, optional
        One color per entry of `times`. The default takes them from the
        ``'jet'`` colormap.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. The default ``None`` creates a new figure.
    style : str, optional
        Plot style, see :func:`pyfar.plot.context`. The default is
        ``'light'``.
    **kwargs
        Keyword arguments passed to :func:`pyfar.plot.freq` for the curves,
        or to :meth:`matplotlib.axes.Axes.fill_between` for the areas if
        `fill` is ``True``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes. Its lines are in drawing order, the longest window first.

    References
    ----------
    .. [#] J. Patynen, S. Tervo and T. Lokki, "Analysis of concert hall
           acoustics via visualizations of time-frequency and
           spatiotemporal responses," J. Acoust. Soc. Am., vol. 133, no. 2,
           pp. 842-857, 2013.

    Examples
    --------
    .. plot::

        >>> import numpy as np
        >>> import pyspar
        >>> from spharpy import SphericalHarmonicSignal
        >>> signal = SphericalHarmonicSignal(
        ...     np.random.default_rng(0).standard_normal((4, 2048)), 48000,
        ...     'real', 'N3D', 'ACN', False)
        >>> analyzer = pyspar.parametric.PseudoIntensityVectorAnalyzer(
        ...     window_length=5)
        >>> result = analyzer(signal)
        >>> ax = pyspar.parametric.plot.time_frequency(
        ...     result, times=(5, 10, 20))
    """
    results = _as_results(result, 'time_frequency')
    times = _check_plot_arguments(integration, times, dynamic_range)
    if (isinstance(num_fractions, bool)
            or not isinstance(num_fractions, (int, np.integer))):
        raise TypeError(
            "num_fractions must be an integer; got "
            f"{type(num_fractions).__name__}.")
    if num_fractions < 1:
        raise ValueError(
            f"num_fractions is {num_fractions} but must be at least 1.")

    # The length of the first response sets the frequency resolution.
    n_fft = results[0].pressure.n_samples
    energy = np.zeros((n_fft, len(times)))
    for single in results:
        pressure, _, fs = _data(single)
        for k, time in enumerate(times):
            window = _window(pressure, time, fs, integration)
            spectrum = np.abs(
                np.fft.fft(pressure[window], n_fft) / np.sqrt(n_fft)) ** 2
            energy[:, k] += _octave_smooth(spectrum, num_fractions)
    frequencies = np.arange(n_fft // 2 + 1) * fs / n_fft
    with np.errstate(divide='ignore'):
        level = (10 * np.log10(energy[:n_fft // 2 + 1])
                 - 10 * np.log10(np.max(energy)))

    with pf.plot.context(style):
        if ax is None:
            _, ax = plt.subplots()

        def draw(k, color):
            # pyfar.plot.freq draws the curve and formats the axes.
            data = pf.FrequencyData(10 ** (level[:, k] / 20), frequencies)
            with np.errstate(divide='ignore'):
                pf.plot.freq(data, ax=ax, style=style, color=color,
                             **({} if fill else kwargs))
            if not fill:
                return ax.lines[-1]
            return ax.fill_between(
                frequencies[1:], np.maximum(level[1:, k], -dynamic_range),
                -dynamic_range, color=color, **{'alpha': 0.5, **kwargs})

        handles = _draw_windows(draw, times, integration, colors)
        ax.set_ylim(-dynamic_range, 3)
        ax.set_yticks(_level_grid(level_spacing, dynamic_range)[0])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.legend(handles, _labels(times, integration, results, fs))

    return ax


def _as_results(result, caller, single=False):
    """Return `result` as a list of analysis results."""
    results = [result] if isinstance(result, TimeResult) else result
    if single and not isinstance(result, TimeResult):
        raise TypeError(
            f"{caller} requires a pyspar.parametric.TimeResult; got "
            f"{type(result).__name__}.")
    if (isinstance(results, (str, bytes)) or not hasattr(results, '__len__')
            or len(results) == 0
            or not all(isinstance(r, TimeResult) for r in results)):
        raise TypeError(
            f"{caller} requires a pyspar.parametric.TimeResult or a "
            f"sequence of them; got {type(result).__name__}.")
    rates = {float(r.sampling_rate) for r in results}
    if len(rates) > 1:
        raise ValueError(
            f"The results have different sampling rates: {sorted(rates)}.")
    return list(results)


def _data(result):
    """Pressure, directions with NaN where there is none, sampling rate."""
    directions = np.array(result.doa.cartesian, dtype=float).reshape(-1, 3)
    return (result.pressure.time[0], directions,
            float(result.sampling_rate))


def _peak(pressure):
    """Absolute peak of the pressure, which must not be silent."""
    peak = np.max(np.abs(pressure))
    if not peak > 0:
        raise ValueError("The pressure of the result is silent.")
    return peak


def _check_positive(value, name):
    if isinstance(value, bool) or not isinstance(value,
                                                 (int, float, np.number)):
        raise TypeError(
            f"{name} must be a number; got {type(value).__name__}.")
    if not value > 0:
        raise ValueError(f"{name} is {value} but must be positive.")


def _check_plot_arguments(integration, times, dynamic_range):
    """Validate the arguments the windowed plots share; return the times."""
    if integration not in ('forward', 'backward'):
        raise ValueError(
            f"integration is '{integration}' but must be 'forward' or "
            "'backward'.")
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if times.ndim != 1 or times.size == 0 or np.any(times < 0):
        raise ValueError(
            "times must be one or more non-negative times in milliseconds; "
            f"got {times.tolist()}.")
    _check_positive(dynamic_range, 'dynamic_range')
    return times


def _round_half_away(x):
    """Round half away from zero."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _onset(pressure, fs):
    """First sample above a fifth of the peak, less one millisecond."""
    return max(int(np.argmax(np.abs(pressure) > 0.2 * _peak(pressure)))
               - int(_round_half_away(0.001 * fs)), 0)


def _window(pressure, time, fs, integration):
    """Samples of the window ending or starting `time` ms after the onset.

    Both windows include the sample `time` after the onset.
    """
    onset = _onset(pressure, fs)
    limit = onset + int(_round_half_away(time / 1000 * fs))
    if integration == 'forward':
        return slice(onset, min(limit + 1, len(pressure)))
    return slice(min(limit, len(pressure) - 1), len(pressure))


def _polar_response(directions, pressure, resolution, plane):
    """Energy per direction bin in a plane.

    The plane's normal takes the role of the z axis: the angle within the
    plane selects the bin and the energy is weighted by the cosine of the
    angle to the plane. Returns one value per bin of `resolution` degrees
    from -180 degrees on; the bin at -180 degrees holds the back.
    """
    x, y, z = directions.T
    if plane == 'lateral':
        u, v, w, turn = x, y, z, 0.0
    elif plane == 'median':
        u, v, w, turn = x, z, -y, 0.0
    else:
        u, v, w, turn = z, y, -x, np.pi / 2
    azimuth = np.arctan2(v, u) + turn
    elevation = np.arctan2(w, np.hypot(u, v))
    # Wrapped to (-180, 180] by a round trip through Cartesian coordinates.
    azimuth = np.arctan2(np.cos(elevation) * np.sin(azimuth),
                         np.cos(elevation) * np.cos(azimuth))

    n_bins = round(360 / resolution)
    index = (_round_half_away(np.rad2deg(azimuth) / resolution)
             + n_bins / 2)
    weight = pressure ** 2 * np.abs(np.cos(elevation))
    valid = np.isfinite(index) & np.isfinite(weight)
    return np.bincount(index[valid].astype(int) % n_bins,
                       weights=weight[valid], minlength=n_bins)


def _circular_smooth(response, size):
    """Centered moving average of `size` bins around the circle."""
    if size <= 1:
        return response
    n = len(response)
    wrapped = np.concatenate([response, response, response])
    smoothed = np.convolve(wrapped, np.ones(size) / size, mode='full')
    # Sample n + i of the full convolution averages wrapped[n + i - size + 1
    # : n + i + 1]; the offset centers the window on bin i.
    return smoothed[n + size // 2:2 * n + size // 2]


def _octave_smooth(spectrum, num_fractions):
    """Average a two-sided spectrum in fractional octaves around each bin.

    Each bin of the energy spectrum becomes the mean of the bins from half a
    fractional octave below it to half a fractional octave above it.
    """
    n = len(spectrum)
    bins = np.arange(n)
    factor = 2 ** (1 / (2 * num_fractions))
    lower = np.minimum(_round_half_away(bins / factor), n - 1).astype(int)
    upper = np.minimum(_round_half_away(bins * factor), n - 1).astype(int)
    cumulative = np.concatenate([[0.0], np.cumsum(np.abs(spectrum))])
    return (cumulative[upper + 1] - cumulative[lower]) / (upper - lower + 1)


def _level_grid(level_spacing, dynamic_range):
    """Levels of the grid in dB, ascending, and their labels."""
    if np.ndim(level_spacing) == 0:
        _check_positive(level_spacing, 'level_spacing')
        levels = -np.arange(0, dynamic_range + level_spacing / 2,
                            level_spacing)[::-1]
    else:
        levels = np.sort(-np.abs(np.asarray(level_spacing, dtype=float)))
    # Adding zero turns the negative zero of the top level into "0".
    return levels, [f"{level + 0.0:g} dB" for level in levels]


def _draw_windows(draw, times, integration, colors):
    """Draw one curve per time, the longest window first.

    Returns the artists in the order of `times`.
    """
    if colors is None:
        colors = plt.get_cmap('jet')(np.linspace(0, 1, len(times)))
    elif len(colors) != len(times):
        raise ValueError(
            f"colors holds {len(colors)} colors for {len(times)} times.")
    order = np.argsort(times)
    if integration == 'forward':
        order = order[::-1]
    handles = {k: draw(k, colors[k]) for k in order}
    return [handles[k] for k in range(len(times))]


def _labels(times, integration, results, fs):
    """Legend entries of the time windows."""
    if integration == 'forward':
        return [f"0 - {time:g} ms" for time in times]
    end = _round_half_away(results[-1].pressure.n_samples / fs * 1000)
    return [f"{time:g} - {end:g} ms" for time in times]
