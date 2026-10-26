"""Reading loudspeaker layouts."""
import json

import numpy as np
import pyfar as pf

from pyspar.utils.plot import _draw_loudspeakers, _map_axes, _project


def load_loudspeaker_array(json_path, return_imaginary=False, plot=False):
    """Load a loudspeaker layout from an IEM-convention JSON file.

    The file holds a ``LoudspeakerLayout`` object whose ``Loudspeakers``
    entries give ``Azimuth`` and ``Elevation`` in degrees and ``Radius`` in
    meters, the JSON layout format of the IEM Plug-in Suite.

    The real loudspeakers are returned in ascending order of their ``Channel``
    numbers, or in the order of the file if it gives none. Gaps in the channel
    numbers are not kept. Loudspeakers marked ``IsImaginary`` have no output of
    their own and are not part of `loudspeakers`; their channel numbers are
    ignored. Gains are ignored.

    Parameters
    ----------
    json_path : str or path-like
        Path to the JSON file.
    return_imaginary : bool, optional
        If ``True``, the imaginary loudspeakers are returned as well. The
        default is ``False``.
    plot : bool, optional
        If ``True``, the layout is plotted on a new figure, a Hammer-Aitoff
        map of the sphere seen from the listener with the front in the
        center and the left on the left. The real loudspeakers are dots
        labelled with their channel numbers, or numbered from 1 in the order
        of `loudspeakers` if the file has none, the imaginary loudspeakers
        open markers, both placed by their azimuth and elevation. The
        default is ``False``.

    Returns
    -------
    loudspeakers : pyfar.Coordinates
        Directions and distances of the real loudspeakers.
    imaginary_loudspeakers : pyfar.Coordinates or None
        Directions and distances of the imaginary loudspeakers in the order
        of the file, or ``None`` if the file has none. Only returned if
        `return_imaginary` is ``True``.

    Examples
    --------
    Load and plot a layout of five loudspeakers at ear height and one
    above, listed in the file in another order than their channel numbers
    and closed below by an imaginary loudspeaker.

    .. plot::

        >>> import json
        >>> import os
        >>> import tempfile
        >>> import numpy as np
        >>> import pyspar
        >>> speakers = [{'Azimuth': azimuth, 'Elevation': elevation,
        ...              'Radius': 2.0, 'Channel': channel}
        ...             for azimuth, elevation, channel in (
        ...                 (0, 0, 2), (30, 0, 1), (110, 0, 5), (-110, 0, 4),
        ...                 (-30, 0, 3), (0, 90, 6))]
        >>> speakers.append({'Azimuth': 0, 'Elevation': -90, 'Radius': 2.0,
        ...                  'IsImaginary': True})
        >>> path = os.path.join(tempfile.mkdtemp(), 'layout.json')
        >>> with open(path, 'w') as file:
        ...     json.dump({'LoudspeakerLayout': {'Loudspeakers': speakers}},
        ...               file)
        >>> loudspeakers, imaginary = pyspar.io.load_loudspeaker_array(
        ...     path, return_imaginary=True, plot=True)
        >>> np.rad2deg(loudspeakers.azimuth[:5]).round()
        array([ 30.,   0., 330., 250., 110.])
        >>> imaginary.cshape
        (1,)
    """
    if not isinstance(return_imaginary, bool):
        raise TypeError("return_imaginary must be a bool; got "
                        f"{type(return_imaginary).__name__}.")
    if not isinstance(plot, bool):
        raise TypeError(f"plot must be a bool; got {type(plot).__name__}.")

    with open(json_path, 'r') as file:
        data = json.load(file)

    try:
        speakers = data['LoudspeakerLayout']['Loudspeakers']
    except (KeyError, TypeError):
        raise ValueError(
            f"{json_path} is not an IEM loudspeaker layout: it has no "
            "'LoudspeakerLayout' with 'Loudspeakers'.") from None

    imaginary = [s for s in speakers if s.get('IsImaginary', False)]
    speakers = [s for s in speakers if not s.get('IsImaginary', False)]
    if not speakers:
        raise ValueError(f"{json_path} contains no real loudspeakers.")

    channels = [s['Channel'] for s in speakers if 'Channel' in s]
    if channels:
        if len(channels) < len(speakers):
            raise ValueError(
                f"Some loudspeakers in {json_path} have a 'Channel' field "
                "and some have none; give it for all or for none.")
        for channel in channels:
            if (isinstance(channel, bool) or not isinstance(channel, int)
                    or channel < 1):
                raise ValueError(
                    f"The channel numbers in {json_path} must be positive "
                    f"integers; got {channel!r}.")
        if len(set(channels)) < len(channels):
            raise ValueError(
                f"The channel numbers in {json_path} must be unique; got "
                f"{sorted(channels)}.")
        speakers = [speakers[i] for i in np.argsort(channels)]
        labels = sorted(channels)
    else:
        labels = range(1, len(speakers) + 1)

    entries = speakers + imaginary if return_imaginary or plot else speakers
    try:
        locations = np.array(
            [[s['Azimuth'], s['Elevation'], s['Radius']] for s in entries],
            dtype=float)
    except KeyError as missing:
        raise ValueError(
            f"A loudspeaker in {json_path} has no {missing} field.") from None

    # Normalize the azimuth to [-180, 180) regardless of the convention of
    # the file; some exports store it in [0, 360).
    locations[:, 0] = (locations[:, 0] + 180) % 360 - 180
    azimuth, elevation = np.deg2rad(locations[:, :2]).T
    count = len(speakers)

    if plot:
        points = np.column_stack([np.cos(elevation) * np.cos(azimuth),
                                  np.cos(elevation) * np.sin(azimuth),
                                  np.sin(elevation)])
        with pf.plot.context():
            ax = _map_axes(None)
            _draw_loudspeakers(ax, points[:count], points[count:])
            for label, xy in zip(labels, _project(points[:count]),
                                 strict=True):
                ax.annotate(str(label), xy, xytext=(4, 4), fontsize=8,
                            textcoords='offset points')

    loudspeakers = pf.Coordinates.from_spherical_elevation(
        azimuth[:count], elevation[:count], locations[:count, 2])
    if not return_imaginary:
        return loudspeakers
    if not imaginary:
        return loudspeakers, None
    return loudspeakers, pf.Coordinates.from_spherical_elevation(
        azimuth[count:], elevation[count:], locations[count:, 2])
