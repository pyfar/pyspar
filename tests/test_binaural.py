"""Tests of the binaural synthesizer."""
import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
from conftest import layout, sh_signal
from spharpy import SphericalHarmonicSignal

from pyspar import parametric


def convolution_direct(feeds, hrirs):
    """Ear signals summed over the loudspeakers, one convolution at a time.

    Each loudspeaker signal is convolved with its response for each ear,
    cropped to the length of the input, and summed.
    """
    n_samples = feeds.shape[0]
    out = np.zeros((n_samples, 2))
    for loudspeaker in range(feeds.shape[1]):
        for ear in range(2):
            out[:, ear] += np.convolve(
                feeds[:, loudspeaker], hrirs[loudspeaker, ear])[:n_samples]
    return out


def ears(signal):
    """Ear signals as ``(n_samples, 2)``."""
    return np.asarray(signal.time).T


# ----- rendering -----------------------------------------------------------

def test_transparent_head_reproduces_the_pressure(analysis, delta_hrirs):
    hrirs, directions = delta_hrirs

    brir = ears(parametric.BinauralSynthesizer(hrirs, directions)(analysis))

    n_samples = analysis.pressure.n_samples
    npt.assert_allclose(brir[:n_samples],
                        analysis.pressure.time[0][:, None] * [1, 1],
                        atol=1e-15)
    npt.assert_allclose(brir[n_samples:], 0.0, atol=1e-15)


@pytest.mark.parametrize('index', [0, 5, 13, 25])
def test_one_loudspeaker_reproduces_its_hrir_per_ear(hrir_set, index):
    hrirs, directions = hrir_set
    synthesizer = parametric.BinauralSynthesizer(hrirs, directions)
    feeds = np.zeros((26, 40))
    feeds[index, 3] = 1.0

    brir = ears(synthesizer.convolve(pf.Signal(feeds, 48000)))

    npt.assert_allclose(brir[3:35], np.asarray(hrirs.time)[index].T,
                        atol=1e-14)


def test_matches_the_direct_computation(analysis, hrir_set):
    hrirs, directions = hrir_set
    synthesizer = parametric.BinauralSynthesizer(hrirs, directions)
    feeds = np.asarray(synthesizer.render_loudspeakers(analysis).time).T

    brir = ears(synthesizer(analysis))

    expected = convolution_direct(feeds, np.asarray(hrirs.time))
    npt.assert_allclose(brir[:feeds.shape[0]], expected, atol=1e-13)


def test_output_is_the_full_linear_convolution(analysis, hrir_set):
    hrirs, directions = hrir_set

    brir = parametric.BinauralSynthesizer(hrirs, directions)(analysis)

    assert isinstance(brir, pf.Signal)
    assert brir.cshape == (2,)
    assert brir.n_samples == analysis.pressure.n_samples + hrirs.n_samples - 1
    assert brir.sampling_rate == 48000


def test_rendering_is_linear(analysis, hrir_set):
    hrirs, directions = hrir_set
    synthesizer = parametric.BinauralSynthesizer(hrirs, directions)
    scaled = parametric.TimeResult(
        pressure=2.5 * analysis.pressure, doa=analysis.doa,
        signal=analysis.signal)

    npt.assert_allclose(ears(synthesizer(scaled)),
                        2.5 * ears(synthesizer(analysis)), atol=1e-14)


def test_call_equals_the_two_stages(analysis, hrir_set):
    hrirs, directions = hrir_set
    synthesizer = parametric.BinauralSynthesizer(hrirs, directions)

    npt.assert_array_equal(
        synthesizer(analysis).time,
        synthesizer.convolve(synthesizer.render_loudspeakers(analysis)).time)


def test_renders_on_the_measurement_grid_by_default(analysis, hrir_set):
    hrirs, directions = hrir_set

    default = parametric.BinauralSynthesizer(hrirs, directions)
    explicit = parametric.BinauralSynthesizer(hrirs, directions,
                                              virtual_loudspeakers=directions)

    assert default.virtual_loudspeakers is None
    assert explicit.virtual_loudspeakers is directions
    npt.assert_array_equal(default(analysis).time, explicit(analysis).time)


def test_virtual_loudspeakers_use_the_nearest_hrir(analysis, hrir_set):
    hrirs, directions = hrir_set
    # Slightly off the measured directions 0, 8, 16 and 24.
    nearby = layout([2, 3, -2, 0], [1, -44, 44, 88])
    subset = [8, 0, 16, 24]

    rendered = parametric.BinauralSynthesizer(hrirs, directions,
                                              virtual_loudspeakers=nearby)

    feeds = rendered.render_loudspeakers(analysis)
    expected = convolution_direct(
        np.asarray(feeds.time).T, np.asarray(hrirs.time)[subset])
    npt.assert_allclose(ears(rendered.convolve(feeds))[:feeds.n_samples],
                        expected, atol=1e-13)


def test_distances_do_not_affect_the_hrir_choice(analysis, hrir_set):
    hrirs, directions = hrir_set
    near = layout([2, 3, -2, 0], [1, -44, 44, 88])
    far = pf.Coordinates(*(10.0 * near.cartesian).T)

    npt.assert_array_equal(
        parametric.BinauralSynthesizer(hrirs, directions, near)(analysis).time,
        parametric.BinauralSynthesizer(hrirs, directions, far)(analysis).time)


def test_properties_reflect_the_virtual_layout(hrir_set):
    hrirs, directions = hrir_set
    virtual = layout([0, 90, 180, -90], [0, 0, 0, 0])

    synthesizer = parametric.BinauralSynthesizer(
        hrirs, directions, virtual, panning='nls', dimensionality=2)

    assert synthesizer.virtual_loudspeakers is virtual
    assert synthesizer.panning == 'nls'
    assert synthesizer.dimensionality == 2


# ----- errors --------------------------------------------------------------

def test_error_if_the_hrirs_are_an_array(hrir_set):
    hrirs, directions = hrir_set

    with pytest.raises(TypeError, match='hrirs must be a pyfar.Signal'):
        parametric.BinauralSynthesizer(np.asarray(hrirs.time), directions)


def test_error_if_the_hrirs_are_spherical_harmonics(hrir_set):
    _, directions = hrir_set
    signal = SphericalHarmonicSignal(np.zeros((4, 8)), 48000, 'real', 'N3D',
                                     'ACN', False)

    with pytest.raises(TypeError, match='hrirs must be a pyfar.Signal'):
        parametric.BinauralSynthesizer(signal, directions)


@pytest.mark.parametrize('cshape', [(26,), (26, 3), (26, 2, 1)])
def test_error_if_the_hrirs_are_not_two_eared(hrir_set, cshape):
    _, directions = hrir_set

    with pytest.raises(ValueError, match=r'cshape \(n_directions, 2\)'):
        parametric.BinauralSynthesizer(
            pf.Signal(np.zeros(cshape + (8,)), 48000), directions)


def test_error_if_the_directions_are_not_coordinates(hrir_set):
    hrirs, directions = hrir_set

    with pytest.raises(TypeError, match='hrir_directions must be'):
        parametric.BinauralSynthesizer(hrirs, directions.cartesian)


def test_error_if_directions_and_hrirs_disagree(hrir_set):
    hrirs, _ = hrir_set

    with pytest.raises(ValueError, match='26 directions but hrir_directions'):
        parametric.BinauralSynthesizer(hrirs, layout([0, 90], [0, 0]))


def test_error_if_the_virtual_loudspeakers_are_not_coordinates(hrir_set):
    hrirs, directions = hrir_set

    with pytest.raises(TypeError,
                       match='virtual_loudspeakers must be a pyfar'):
        parametric.BinauralSynthesizer(hrirs, directions, np.zeros((4, 3)))


def test_error_if_a_direction_is_at_the_origin(hrir_set):
    hrirs, directions = hrir_set
    xyz = directions.cartesian.copy()
    xyz[0] = 0

    with pytest.raises(ValueError,
                       match='hrir_directions must have a nonzero'):
        parametric.BinauralSynthesizer(hrirs, pf.Coordinates(*xyz.T))
    with pytest.raises(ValueError,
                       match='virtual_loudspeakers must have a nonzero'):
        parametric.BinauralSynthesizer(
            hrirs, directions,
            virtual_loudspeakers=pf.Coordinates([1, 0], [0, 0], [0, 0]))


def test_error_if_the_sampling_rates_disagree(hrir_set):
    hrirs, directions = hrir_set
    signal = sh_signal(np.random.default_rng(1).standard_normal((4, 64)),
                       sampling_rate=44100)
    result = parametric.PseudoIntensityVectorAnalyzer()(signal)

    with pytest.raises(ValueError, match='resample the hrirs'):
        parametric.BinauralSynthesizer(hrirs, directions)(result)


def test_error_if_convolve_gets_the_wrong_channel_count(hrir_set):
    hrirs, directions = hrir_set
    synthesizer = parametric.BinauralSynthesizer(hrirs, directions)

    with pytest.raises(ValueError, match=r'cshape \(26,\)'):
        synthesizer.convolve(pf.Signal(np.zeros((4, 16)), 48000))


@pytest.mark.parametrize('signals', [
    np.zeros((26, 16)), sh_signal(np.zeros((25, 16)))],
    ids=['array', 'spherical-harmonic-signal'])
def test_convolve_error_if_the_input_is_not_a_signal(hrir_set, signals):
    synthesizer = parametric.BinauralSynthesizer(*hrir_set)

    with pytest.raises(TypeError, match='must be a pyfar.Signal'):
        synthesizer.convolve(signals)


# ----- pipeline ------------------------------------------------------------

def test_sdm_binaural_equals_the_two_stages(impulse_response, hrir_set):
    hrirs, directions = hrir_set

    rendered = parametric.sdm_binaural(impulse_response, hrirs, directions,
                                       window_length=5)

    result = parametric.PseudoIntensityVectorAnalyzer(window_length=5)(
        impulse_response)
    expected = parametric.BinauralSynthesizer(hrirs, directions)(result)
    npt.assert_array_equal(rendered.time, expected.time)


@pytest.mark.filterwarnings('ignore:.*samples have no direction')
def test_sdm_binaural_with_microphones_equals_the_two_stages(
        array_response, microphones, hrir_set):
    hrirs, directions = hrir_set

    rendered = parametric.sdm_binaural(array_response, hrirs, directions,
                                       window_length=40,
                                       microphones=microphones,
                                       reference_channel=4)

    result = parametric.TimeDifferenceOfArrivalAnalyzer(
        microphones, reference_channel=4, window_length=40)(array_response)
    expected = parametric.BinauralSynthesizer(hrirs, directions)(result)
    npt.assert_array_equal(rendered.time, expected.time)


def test_sdm_binaural_renders_on_given_loudspeakers(impulse_response,
                                                    hrir_set):
    hrirs, directions = hrir_set
    virtual = layout([0, 90, 180, -90], [0, 0, 0, 0])

    rendered = parametric.sdm_binaural(impulse_response, hrirs, directions,
                                       window_length=5,
                                       virtual_loudspeakers=virtual,
                                       dimensionality=2)

    result = parametric.PseudoIntensityVectorAnalyzer(window_length=5)(
        impulse_response)
    expected = parametric.BinauralSynthesizer(hrirs, directions, virtual,
                                              dimensionality=2)(result)
    npt.assert_array_equal(rendered.time, expected.time)
