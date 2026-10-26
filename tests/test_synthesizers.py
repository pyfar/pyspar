"""Tests of the loudspeaker synthesizer."""
import warnings

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
from conftest import encode, layout

from pyspar import parametric


def nls_direct(directions, loudspeakers):
    """Index of the nearest loudspeaker by great-circle distance.

    The angles in degrees between every direction and every loudspeaker,
    and the first smallest one.
    """
    loudspeakers = loudspeakers / np.linalg.norm(
        loudspeakers, axis=1, keepdims=True)
    angle = np.degrees(np.arccos(np.clip(directions @ loudspeakers.T, -1, 1)))
    return np.argmin(angle, axis=1)


def render(result, loudspeakers, **kwargs):
    """Loudspeaker signals as ``(n_samples, n_loudspeakers)``."""
    signal = parametric.LoudspeakerSynthesizer(loudspeakers, **kwargs)(result)
    return np.asarray(signal.time).T


def result_with(analysis, directions):
    """`analysis` with its directions replaced, given as ``(n_samples, 3)``."""
    return parametric.TimeResult(
        pressure=analysis.pressure,
        doa=pf.Coordinates(*np.asarray(directions).T),
        signal=analysis.signal)


# ----- nearest loudspeaker -------------------------------------------------

@pytest.mark.parametrize('layout_name', [
    'layout_horizontal', 'layout_surround', 'layout_hemisphere'])
def test_nls_reproduces_the_pressure_exactly(analysis, layout_name, request):
    data = render(analysis, request.getfixturevalue(layout_name))

    npt.assert_array_equal(data.sum(axis=1), analysis.pressure.time[0])


def test_nls_feeds_one_loudspeaker_per_sample(analysis, layout_surround):
    data = render(analysis, layout_surround)

    assert np.all(np.count_nonzero(data, axis=1) <= 1)


@pytest.mark.parametrize('layout_name', [
    'layout_horizontal', 'layout_surround', 'layout_hemisphere'])
def test_nls_matches_the_direct_computation(analysis, layout_name, request):
    loudspeakers = request.getfixturevalue(layout_name)
    data = render(analysis, loudspeakers)
    sounding = analysis.pressure.time[0] != 0

    expected = nls_direct(analysis.doa.cartesian[sounding],
                          loudspeakers.cartesian)

    npt.assert_array_equal(np.argmax(np.abs(data[sounding]), axis=1),
                           expected)


@pytest.mark.parametrize('index', range(8))
def test_nls_sends_a_plane_wave_to_its_loudspeaker(index, layout_surround):
    pressure = np.zeros(16)
    pressure[3] = 1.0
    direction = pf.Coordinates(*layout_surround.cartesian[index])
    result = parametric.PseudoIntensityVectorAnalyzer()(
        encode(pressure, direction))

    data = render(result, layout_surround)

    assert np.flatnonzero(data[3]).tolist() == [index]


def test_output_follows_the_order_of_the_loudspeakers(analysis,
                                                      layout_surround):
    order = [3, 0, 7, 1, 5, 2, 6, 4]
    reordered = pf.Coordinates(*layout_surround.cartesian[order].T)

    npt.assert_array_equal(render(analysis, reordered),
                           render(analysis, layout_surround)[:, order])


def test_loudspeaker_distance_is_ignored(analysis, layout_surround):
    farther = pf.Coordinates(*(3.5 * layout_surround.cartesian).T)

    npt.assert_array_equal(render(analysis, farther),
                           render(analysis, layout_surround))


def test_output_is_a_signal_per_loudspeaker(analysis, layout_surround):
    signal = parametric.LoudspeakerSynthesizer(layout_surround)(analysis)

    assert isinstance(signal, pf.Signal)
    assert signal.cshape == (8,)
    assert signal.n_samples == analysis.pressure.n_samples
    assert signal.sampling_rate == analysis.sampling_rate


# ----- two-dimensional weighting --------------------------------------------

def test_2d_weights_the_pressure_by_the_cosine_of_elevation(
        analysis, layout_horizontal):
    # Samples without a direction are placed straight ahead.
    elevation = np.nan_to_num(np.arcsin(analysis.doa.cartesian[:, 2]))

    data = render(analysis, layout_horizontal, dimensionality=2)

    npt.assert_allclose(data.sum(axis=1),
                        analysis.pressure.time[0] * np.abs(np.cos(elevation)),
                        atol=1e-15)


def test_2d_attenuates_an_elevated_plane_wave(layout_horizontal):
    pressure = np.zeros(16)
    pressure[3] = 1.0
    elevated = pf.Coordinates.from_spherical_elevation(0, np.deg2rad(60), 1)
    result = parametric.PseudoIntensityVectorAnalyzer()(
        encode(pressure, elevated))

    data = render(result, layout_horizontal, dimensionality=2)

    npt.assert_allclose(data[3].sum(), 0.5 * result.pressure.time[0][3],
                        atol=1e-15)


# ----- samples without a direction -----------------------------------------

def test_silent_samples_without_a_direction_are_rendered_silently(
        analysis, layout_surround):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        data = render(analysis, layout_surround)

    silent = analysis.pressure.time[0] == 0
    npt.assert_array_equal(data[silent], 0.0)


def test_sounding_sample_without_a_direction_warns(analysis, layout_surround):
    directions = analysis.doa.cartesian.copy()
    loudest = np.argmax(np.abs(analysis.pressure.time[0]))
    directions[loudest] = np.nan

    with pytest.warns(UserWarning,
                      match='1 of 2048 samples .* carry pressure'):
        render(result_with(analysis, directions), layout_surround)


def test_quiet_samples_without_a_direction_do_not_warn(analysis,
                                                       layout_surround):
    # As near zero crossings of a measured response: 1e-4 of the energy
    # without a direction is not worth a warning.
    directions = analysis.doa.cartesian.copy()
    energy = analysis.pressure.time[0] ** 2
    quiet = np.flatnonzero(
        np.cumsum(np.sort(energy)) <= 1e-4 * np.sum(energy))
    directions[np.argsort(energy)[quiet]] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        data = render(result_with(analysis, directions), layout_surround)

    npt.assert_allclose(np.sum(data, axis=1), analysis.pressure.time[0],
                        atol=1e-15)


def test_sample_without_a_direction_is_placed_in_front(analysis,
                                                       layout_surround):
    directions = analysis.doa.cartesian.copy()
    loudest = np.argmax(np.abs(analysis.pressure.time[0]))
    directions[loudest] = np.nan

    with pytest.warns(UserWarning, match='straight ahead'):
        data = render(result_with(analysis, directions), layout_surround)

    front = np.argmax(layout_surround.cartesian @ [1.0, 0.0, 0.0])
    assert np.flatnonzero(data[loudest]).tolist() == [front]


# ----- parameters and errors -----------------------------------------------

def test_default_parameters(layout_surround):
    synthesizer = parametric.LoudspeakerSynthesizer(layout_surround)

    assert synthesizer.loudspeakers is layout_surround
    assert synthesizer.panning == 'nls'
    assert synthesizer.dimensionality == 3


def test_error_if_the_loudspeakers_are_not_coordinates():
    with pytest.raises(TypeError, match='must be a pyfar.Coordinates'):
        parametric.LoudspeakerSynthesizer(np.zeros((5, 3)))


def test_error_if_there_are_no_loudspeakers():
    with pytest.raises(ValueError, match='at least one direction'):
        parametric.LoudspeakerSynthesizer(pf.Coordinates())


def test_error_if_a_loudspeaker_is_at_the_origin():
    loudspeakers = pf.Coordinates([1, 0], [0, 0], [0, 0])

    with pytest.raises(ValueError, match='nonzero radius'):
        parametric.LoudspeakerSynthesizer(loudspeakers)


def test_error_if_the_panning_is_unknown(layout_surround):
    with pytest.raises(ValueError, match="panning is 'ambisonic' but must"):
        parametric.LoudspeakerSynthesizer(layout_surround, panning='ambisonic')


@pytest.mark.parametrize('dimensionality', [1, 4, '3'])
def test_error_if_the_dimensionality_is_unknown(layout_surround,
                                                dimensionality):
    with pytest.raises(ValueError, match='dimensionality is .* but must'):
        parametric.LoudspeakerSynthesizer(layout_surround,
                                          dimensionality=dimensionality)


def test_error_if_the_input_is_not_an_analysis_result(layout_surround):
    with pytest.raises(TypeError,
                       match='requires a pyspar.parametric.TimeResult'):
        parametric.LoudspeakerSynthesizer(layout_surround)(np.zeros(16))


def test_single_loudspeaker_receives_the_whole_pressure(analysis):
    data = render(analysis, layout([0], [0]))

    npt.assert_array_equal(data[:, 0], analysis.pressure.time[0])
