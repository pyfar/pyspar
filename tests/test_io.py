"""Tests of reading loudspeaker layouts."""
import json

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.testing as npt  # noqa: E402
import pyfar as pf  # noqa: E402
import pytest  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402

import pyspar  # noqa: E402
from pyspar.utils.plot import _hammer_aitoff  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    """Close the figures a test opened."""
    yield
    plt.close('all')


def write_layout(path, speakers):
    """Write `speakers` as an IEM loudspeaker layout and return the path."""
    path.write_text(json.dumps(
        {'Name': 'test', 'LoudspeakerLayout': {
            'Name': 'test', 'Loudspeakers': speakers}}))
    return path


def speaker(azimuth, elevation, radius=1.0, channel=1, imaginary=False):
    """One entry as the IEM Plug-in Suite writes it."""
    return {'Azimuth': azimuth, 'Elevation': elevation, 'Radius': radius,
            'IsImaginary': imaginary, 'Channel': channel, 'Gain': 1.0}


def test_directions_and_distances_are_read(tmp_path):
    path = write_layout(tmp_path / 'layout.json', [
        speaker(0, 0, 2.0), speaker(90, 0, 1.5), speaker(0, 90, 3.0)])

    loudspeakers = pyspar.io.load_loudspeaker_array(path)

    expected = pf.Coordinates.from_spherical_elevation(
        np.deg2rad([0, 90, 0]), np.deg2rad([0, 0, 90]), [2.0, 1.5, 3.0])
    npt.assert_allclose(loudspeakers.cartesian, expected.cartesian,
                        atol=1e-12)


def test_azimuth_convention_of_the_file_does_not_matter(tmp_path):
    signed = write_layout(tmp_path / 'signed.json',
                          [speaker(az, 10) for az in (-150, -90, 0, 90)])
    unsigned = write_layout(tmp_path / 'unsigned.json',
                            [speaker(az, 10) for az in (210, 270, 0, 90)])

    npt.assert_allclose(pyspar.io.load_loudspeaker_array(unsigned).cartesian,
                        pyspar.io.load_loudspeaker_array(signed).cartesian,
                        atol=1e-12)


def test_imaginary_loudspeakers_are_dropped(tmp_path):
    path = write_layout(tmp_path / 'layout.json', [
        speaker(0, 0, channel=1), speaker(0, -90, channel=99,
                                          imaginary=True),
        speaker(90, 0, channel=2)])

    loudspeakers = pyspar.io.load_loudspeaker_array(path)

    expected = pf.Coordinates.from_spherical_elevation(
        np.deg2rad([0, 90]), [0, 0], 1)
    npt.assert_allclose(loudspeakers.cartesian, expected.cartesian,
                        atol=1e-12)


def test_file_order_is_kept_whatever_the_channel_numbers(tmp_path):
    path = write_layout(tmp_path / 'layout.json', [
        speaker(30, 0, channel=2), speaker(-30, 0, channel=1),
        speaker(110, 0, channel=12)])

    azimuth = np.rad2deg(pyspar.io.load_loudspeaker_array(path).azimuth)

    npt.assert_allclose(azimuth, [30, 330, 110], atol=1e-12)


def test_entries_without_the_optional_fields_are_read(tmp_path):
    path = write_layout(tmp_path / 'layout.json', [
        {'Azimuth': 0, 'Elevation': 0, 'Radius': 1.0},
        {'Azimuth': 90, 'Elevation': 0, 'Radius': 1.0}])

    assert pyspar.io.load_loudspeaker_array(path).cshape == (2,)


def test_path_may_be_a_string(tmp_path):
    path = write_layout(tmp_path / 'layout.json', [speaker(0, 0)])

    assert pyspar.io.load_loudspeaker_array(str(path)).cshape == (1,)


@pytest.mark.parametrize('content', [
    {'Name': 'no layout'},
    {'LoudspeakerLayout': {'Name': 'no loudspeakers'}},
    ['not', 'an', 'object']])
def test_error_if_the_file_is_not_an_iem_layout(tmp_path, content):
    path = tmp_path / 'other.json'
    path.write_text(json.dumps(content))

    with pytest.raises(ValueError, match='not an IEM loudspeaker layout'):
        pyspar.io.load_loudspeaker_array(path)


@pytest.mark.parametrize('field', ['Azimuth', 'Elevation', 'Radius'])
def test_error_if_an_entry_lacks_a_field(tmp_path, field):
    entry = speaker(0, 0)
    del entry[field]
    path = write_layout(tmp_path / 'layout.json', [speaker(90, 0), entry])

    with pytest.raises(ValueError, match=f"has no '{field}' field"):
        pyspar.io.load_loudspeaker_array(path)


def test_error_if_only_imaginary_loudspeakers_remain(tmp_path):
    path = write_layout(tmp_path / 'layout.json',
                        [speaker(0, -90, imaginary=True)])

    with pytest.raises(ValueError, match='no real loudspeakers'):
        pyspar.io.load_loudspeaker_array(path)


def test_plot_marks_real_and_imaginary_loudspeakers(tmp_path):
    path = write_layout(tmp_path / 'layout.json', [
        speaker(30, 0), speaker(-30, 10), speaker(0, 90),
        speaker(0, -90, radius=0.0, imaginary=True)])

    pyspar.io.load_loudspeaker_array(path, plot=True)

    ax = plt.gca()
    real, imaginary = ax.collections
    npt.assert_allclose(
        real.get_offsets(),
        _hammer_aitoff(np.deg2rad([30, -30, 0]), np.deg2rad([0, 10, 90])),
        atol=1e-12)
    npt.assert_allclose(imaginary.get_offsets(), [[0, -0.5]], atol=1e-12)
    assert len(imaginary.get_facecolor()) == 0
    labels = [text.get_text() for text in ax.texts
              if isinstance(text, Annotation)]
    assert labels == ['1', '2', '3']


def test_nothing_is_plotted_by_default(tmp_path):
    path = write_layout(tmp_path / 'layout.json', [speaker(0, 0)])

    pyspar.io.load_loudspeaker_array(path)

    assert plt.get_fignums() == []


@pytest.mark.parametrize(('azimuth', 'elevation', 'expected'), [
    (0, 0, [0, 0]), (0, 90, [0, 0.5]), (0, -90, [0, -0.5]),
    (90, 0, [-np.sin(np.pi / 4) / np.sqrt(1 + np.cos(np.pi / 4)), 0])])
def test_hammer_aitoff_shows_the_front_in_the_center_and_left_on_the_left(
        azimuth, elevation, expected):
    npt.assert_allclose(
        _hammer_aitoff(np.deg2rad(azimuth), np.deg2rad(elevation)),
        expected, atol=1e-15)


@pytest.mark.parametrize('field', ['Azimuth', 'Elevation'])
def test_error_if_an_imaginary_entry_to_plot_lacks_a_field(tmp_path, field):
    entry = speaker(0, -90, imaginary=True)
    del entry[field]
    path = write_layout(tmp_path / 'layout.json', [speaker(0, 0), entry])

    assert pyspar.io.load_loudspeaker_array(path).cshape == (1,)
    with pytest.raises(ValueError, match=f"has no '{field}' field"):
        pyspar.io.load_loudspeaker_array(path, plot=True)


@pytest.mark.parametrize('plot', ['yes', 1, None])
def test_error_if_plot_is_not_a_bool(tmp_path, plot):
    path = write_layout(tmp_path / 'layout.json', [speaker(0, 0)])

    with pytest.raises(TypeError, match='plot must be a bool'):
        pyspar.io.load_loudspeaker_array(path, plot=plot)
