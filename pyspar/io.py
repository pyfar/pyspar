"""Reading loudspeaker layouts."""
import json

import numpy as np
import pyfar as pf

from pyspar.utils.plot import _draw_loudspeakers, _map_axes, _project


def load_loudspeaker_array(json_path, plot=False):
    """Load a loudspeaker layout from an IEM-convention JSON file.

    The file holds a ``LoudspeakerLayout`` object whose ``Loudspeakers``
    entries give ``Azimuth`` and ``Elevation`` in degrees and ``Radius`` in
    meters, the JSON layout format of the IEM Plug-in Suite.

    Loudspeakers marked ``IsImaginary`` are dropped: they have no output and
    exist only to close the triangulation of an AllRAD decoder. Channel
    numbers and gains are ignored, so the order of the file is kept.

    Parameters
    ----------
    json_path : str or path-like
        Path to the JSON file.
    plot : bool, optional
        If ``True``, the layout is plotted on a new figure, a Hammer-Aitoff
        map of the sphere seen from the listener with the front in the
        center and the left on the left. The real loudspeakers are dots
        numbered from 1 in the order of `loudspeakers`, the imaginary
        loudspeakers of the file open markers, both placed by their azimuth
        and elevation. The default is ``False``.

    Returns
    -------
    loudspeakers : pyfar.Coordinates
        Directions and distances of the real loudspeakers, in the order of
        the file.

    Examples
    --------
    Load and plot a layout of five loudspeakers at ear height and one
    above, closed below by an imaginary loudspeaker.

    .. plot::

        >>> import json
        >>> import os
        >>> import tempfile
        >>> import pyspar
        >>> speakers = [{'Azimuth': azimuth, 'Elevation': elevation,
        ...              'Radius': 2.0}
        ...             for azimuth, elevation in (
        ...                 (0, 0), (30, 0), (-30, 0), (110, 0), (-110, 0),
        ...                 (0, 90))]
        >>> speakers.append({'Azimuth': 0, 'Elevation': -90, 'Radius': 2.0,
        ...                  'IsImaginary': True})
        >>> path = os.path.join(tempfile.mkdtemp(), 'layout.json')
        >>> with open(path, 'w') as file:
        ...     json.dump({'LoudspeakerLayout': {'Loudspeakers': speakers}},
        ...               file)
        >>> loudspeakers = pyspar.io.load_loudspeaker_array(path, plot=True)
    """
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

    try:
        locations = np.array(
            [[s['Azimuth'], s['Elevation'], s['Radius']] for s in speakers],
            dtype=float)
        if plot:
            imaginary = np.array(
                [[s['Azimuth'], s['Elevation']] for s in imaginary],
                dtype=float).reshape(-1, 2)
    except KeyError as missing:
        raise ValueError(
            f"A loudspeaker in {json_path} has no {missing} field.") from None

    # Normalize the azimuth to [-180, 180) regardless of the convention of
    # the file; some exports store it in [0, 360).
    locations[:, 0] = (locations[:, 0] + 180) % 360 - 180

    if plot:
        azimuth, elevation = np.deg2rad(
            np.concatenate([locations[:, :2], imaginary])).T
        points = np.column_stack([np.cos(elevation) * np.cos(azimuth),
                                  np.cos(elevation) * np.sin(azimuth),
                                  np.sin(elevation)])
        count = len(locations)
        with pf.plot.context():
            ax = _map_axes(None)
            _draw_loudspeakers(ax, points[:count], points[count:])
            for number, xy in enumerate(_project(points[:count]), start=1):
                ax.annotate(str(number), xy, xytext=(4, 4), fontsize=8,
                            textcoords='offset points')

    return pf.Coordinates.from_spherical_elevation(
        np.deg2rad(locations[:, 0]), np.deg2rad(locations[:, 1]),
        locations[:, 2])
