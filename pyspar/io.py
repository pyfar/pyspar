"""Reading loudspeaker layouts."""
import json

import numpy as np
import pyfar as pf


def load_loudspeaker_array(json_path):
    """Load a loudspeaker layout from an IEM-convention JSON file.

    The file holds a ``LoudspeakerLayout`` object whose ``Loudspeakers``
    entries give ``Azimuth`` and ``Elevation`` in degrees and ``Radius`` in
    meters, as written by the IEM Plug-in Suite and by the AmbiDEC plug-in
    of SPARTA, which uses the same format.

    Loudspeakers marked ``IsImaginary`` are dropped: they have no output and
    exist only to close the triangulation of an AllRAD decoder. Channel
    numbers and gains are ignored, since they belong to the room's wiring
    rather than to the rendering, so the order of the file is kept.

    Parameters
    ----------
    json_path : str or path-like
        Path to the JSON file.

    Returns
    -------
    loudspeakers : pyfar.Coordinates
        Directions and distances of the real loudspeakers, in the order of
        the file.

    Examples
    --------
    >>> import json
    >>> import os
    >>> import tempfile
    >>> import pyspar
    >>> layout = {'LoudspeakerLayout': {'Loudspeakers': [
    ...     {'Azimuth': az, 'Elevation': 0, 'Radius': 2.0}
    ...     for az in (0, 30, -30, 110, -110)]}}
    >>> path = os.path.join(tempfile.mkdtemp(), 'five.json')
    >>> with open(path, 'w') as file:
    ...     json.dump(layout, file)
    >>> pyspar.io.load_loudspeaker_array(path).cshape
    (5,)
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    try:
        speakers = data['LoudspeakerLayout']['Loudspeakers']
    except (KeyError, TypeError):
        raise ValueError(
            f"{json_path} is not an IEM loudspeaker layout: it has no "
            "'LoudspeakerLayout' with 'Loudspeakers'.") from None

    speakers = [s for s in speakers if not s.get('IsImaginary', False)]
    if not speakers:
        raise ValueError(f"{json_path} contains no real loudspeakers.")

    try:
        locations = np.array(
            [[s['Azimuth'], s['Elevation'], s['Radius']] for s in speakers],
            dtype=float)
    except KeyError as missing:
        raise ValueError(
            f"A loudspeaker in {json_path} has no {missing} field.") from None

    # Normalize the azimuth to [-180, 180) regardless of the convention of
    # the file; some exports store it in [0, 360).
    locations[:, 0] = (locations[:, 0] + 180) % 360 - 180
    return pf.Coordinates.from_spherical_elevation(
        np.deg2rad(locations[:, 0]), np.deg2rad(locations[:, 1]),
        locations[:, 2])
