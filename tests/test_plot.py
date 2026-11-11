"""Tests of the plots."""
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.testing as npt  # noqa: E402
import pyfar as pf  # noqa: E402
import pytest  # noqa: E402

from pyspar import parametric  # noqa: E402
from pyspar.parametric import plot  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    """Close the figures a test opened."""
    yield
    plt.close('all')


def make_result(pressure, directions, sampling_rate=48000):
    """Analysis result from a pressure and ``(n_samples, 3)`` directions."""
    pressure = np.asarray(pressure, dtype=float)
    directions = np.asarray(directions, dtype=float)
    return parametric.TimeResult(
        pressure=pf.Signal(pressure, sampling_rate),
        signal=pf.Signal(pressure, sampling_rate),
        doa=pf.Coordinates(*directions.T))


@pytest.fixture
def impulses():
    """Unit pressure at sample 100 from the front and at 400 from the left."""
    pressure = np.zeros(1000)
    pressure[[100, 400]] = 1.0
    directions = np.full((1000, 3), np.nan)
    directions[100] = [1, 0, 0]
    directions[400] = [0, 1, 0]
    return make_result(pressure, directions)


def curves(ax):
    """Data of the lines of `ax` in drawing order, one row per line."""
    return np.array([line.get_ydata() for line in ax.lines])


# -- the quantities behind the plots ------------------------------------------

@pytest.mark.parametrize(('integration', 'first', 'last'), [
    ('forward', 52, 292), ('backward', 292, 999)])
def test_window_includes_the_sample_at_its_limit(integration, first, last):
    # Onset 1 ms (48 samples) before the impulse; the limit is 5 ms (240
    # samples) after the onset.
    pressure = np.zeros(1000)
    pressure[100] = 1.0

    window = plot._window(pressure, 5, 48000, integration)

    assert np.arange(1000)[window][[0, -1]].tolist() == [first, last]


def test_window_onset_is_the_first_sample_above_a_fifth_of_the_peak():
    pressure = np.zeros(1000)
    pressure[[300, 320, 340]] = [0.19, 0.21, 1.0]

    window = plot._window(pressure, 0, 48000, 'forward')

    assert window.start == 320 - 48


def test_window_stays_within_the_response():
    pressure = np.zeros(100)
    pressure[10] = 1.0

    assert plot._window(pressure, 50, 48000, 'forward') == slice(0, 100)
    assert plot._window(pressure, 50, 48000, 'backward') == slice(99, 100)


@pytest.mark.parametrize(('plane', 'direction', 'angle'), [
    ('lateral', [1, 0, 0], 0), ('lateral', [0, 1, 0], 90),
    ('lateral', [0, -1, 0], -90), ('median', [1, 0, 0], 0),
    ('median', [0, 0, 1], 90), ('median', [0, 0, -1], -90),
    ('transverse', [0, -1, 0], 0), ('transverse', [0, 0, 1], 90),
    ('transverse', [0, 0, -1], -90), ('lateral', [-1, 0, 0], -180),
    ('lateral', [-1, -1e-9, 0], -180)])
def test_polar_response_puts_a_direction_at_its_angle(plane, direction,
                                                      angle):
    response = plot._polar_response(
        np.array([direction], float), np.array([2.0]), 1, plane)

    # One bin per degree from -180 to 179; the back is the bin at -180.
    expected = np.zeros(360)
    expected[angle + 180] = 4.0
    npt.assert_allclose(response, expected, atol=1e-15)


def test_polar_response_weights_by_the_cosine_to_the_plane():
    elevation = np.deg2rad(60)
    direction = [[np.cos(elevation), 0, np.sin(elevation)]]

    response = plot._polar_response(np.array(direction), np.array([1.0]), 1,
                                    'lateral')

    npt.assert_allclose(response[180], 0.5)


def test_polar_response_bins_have_the_width_of_the_resolution():
    azimuth = np.deg2rad([4.9, 5.1, 14.9])
    directions = np.column_stack([np.cos(azimuth), np.sin(azimuth),
                                  np.zeros(3)])

    response = plot._polar_response(directions, np.ones(3), 10, 'lateral')

    assert len(response) == 36
    npt.assert_array_equal(response[18:20], [1, 2])


def test_polar_response_skips_samples_without_a_direction(impulses):
    pressure, directions, _ = plot._data(impulses)
    pressure = pressure.copy()
    pressure[200] = 5.0

    response = plot._polar_response(directions, pressure, 1, 'lateral')

    npt.assert_allclose(np.sum(response), 2.0)


def test_circular_smooth_centers_and_wraps_the_average():
    response = np.zeros(8)
    response[0] = 3.0

    smoothed = plot._circular_smooth(response, 3)

    npt.assert_allclose(smoothed, [1, 1, 0, 0, 0, 0, 0, 1])


def test_circular_smooth_of_one_bin_changes_nothing():
    response = np.arange(5.0)

    assert plot._circular_smooth(response, 1) is response


def test_octave_smooth_averages_a_band_around_each_bin():
    spectrum = np.arange(64.0)
    factor = 2 ** (1 / 2)

    smoothed = plot._octave_smooth(spectrum, 1)

    for index in (0, 1, 10, 40, 63):
        lower = min(int(np.floor(index / factor + 0.5)), 63)
        upper = min(int(np.floor(index * factor + 0.5)), 63)
        npt.assert_allclose(smoothed[index],
                            np.mean(spectrum[lower:upper + 1]))


def test_octave_smooth_keeps_a_flat_spectrum():
    npt.assert_allclose(plot._octave_smooth(np.full(128, 2.5), 3), 2.5)


# -- what is drawn ------------------------------------------------------------

def test_spatio_temporal_draws_one_curve_per_time_on_polar_axes(impulses):
    ax = plot.spatio_temporal(impulses, times=(1, 5, 20))

    assert ax.name == 'polar'
    assert len(ax.lines) == 3


def test_spatio_temporal_draws_the_longest_window_first(impulses):
    # Only the 20 ms window reaches the impulse from the left.
    ax = plot.spatio_temporal(impulses, times=(1, 20), smoothing=1,
                              colors=['r', 'b'])

    first, second = curves(ax)
    assert ax.lines[0].get_color() == 'b'
    npt.assert_allclose(first[[180, 270]], 0)
    assert second[180] == 0
    assert second[270] == -45


def test_spatio_temporal_labels_backward_windows_to_the_end(impulses):
    ax = plot.spatio_temporal(impulses, integration='backward',
                              times=(0, 5))

    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        '0 - 21 ms', '5 - 21 ms']


def test_spatio_temporal_limits_levels_to_the_dynamic_range(impulses):
    ax = plot.spatio_temporal(impulses, times=(20,), dynamic_range=30)

    assert ax.get_ylim() == (-30, 0)
    assert np.min(curves(ax)) == -30
    assert np.max(curves(ax)) == 0


@pytest.mark.parametrize(('level_spacing', 'levels'), [
    (10, [-20, -10, 0]), ([-24, -12], [-24, -12])])
def test_level_grid_from_a_spacing_or_from_levels(impulses, level_spacing,
                                                  levels):
    ax = plot.spatio_temporal(impulses, dynamic_range=30,
                              level_spacing=level_spacing)

    npt.assert_allclose(ax.get_yticks(), levels)


def test_spatio_temporal_fills_the_curves(impulses):
    ax = plot.spatio_temporal(impulses, times=(1, 5, 20), fill=True)

    assert len(ax.lines) == 0
    assert len(ax.patches) == 3


def test_spatio_temporal_plots_on_given_axes(impulses):
    _, given = plt.subplots(subplot_kw={'projection': 'polar'})

    assert plot.spatio_temporal(impulses, ax=given) is given


def test_spatio_temporal_sums_the_energy_of_several_results(impulses):
    pressure = np.zeros(600)
    pressure[100] = 1.0
    directions = np.full((600, 3), np.nan)
    directions[100] = [-1, 0, 0]
    behind = make_result(pressure, directions)

    ax = plot.spatio_temporal([impulses, behind], times=(5,), smoothing=1)

    # The front is bin 180, the back bin 0 and the closing point 360.
    npt.assert_allclose(curves(ax)[0, [0, 180, 360]], 0, atol=1e-12)


def test_spatio_temporal_names_the_directions_in_the_tick_labels(impulses):
    ax = plot.spatio_temporal(impulses, plane='median')

    labels = [label.get_text() for label in ax.get_xticklabels()]
    assert labels[0] == '0$^\\circ$ Front'
    assert labels[3] == 'Top\n90$^\\circ$'
    assert labels[6] == 'Back 180$^\\circ$'
    assert labels[9] == '-90$^\\circ$\nBottom'
    assert not ax.texts


def test_time_frequency_draws_one_curve_per_time(impulses):
    ax = plot.time_frequency(impulses, times=(1, 5, 20))

    assert len(ax.lines) == 3
    assert ax.get_ylim() == (-45, 3)


def test_time_frequency_levels_are_relative_to_the_maximum(impulses):
    # One impulse has a flat spectrum, two have peaks of twice the energy.
    ax = plot.time_frequency(impulses, times=(1, 20), num_fractions=24)

    longest, shortest = curves(ax)
    npt.assert_allclose(np.max(longest), 0, atol=1e-12)
    npt.assert_allclose(shortest[1:-1], -10 * np.log10(4), atol=1e-9)


def test_time_frequency_fills_below_the_curves(impulses):
    ax = plot.time_frequency(impulses, times=(1, 5, 20), fill=True)

    assert len(ax.lines) == 3
    assert len(ax.collections) == 3


def test_time_frequency_uses_the_length_of_the_first_result(impulses):
    short = make_result(impulses.pressure.time[0][:600],
                        impulses.doa.cartesian[:600])

    ax = plot.time_frequency([impulses, short], times=(5,))

    assert len(ax.lines[0].get_xdata()) == 501


def test_time_frequency_plots_on_given_axes(impulses):
    _, given = plt.subplots()

    assert plot.time_frequency(impulses, ax=given) is given


def test_doa_map_returns_the_map_and_the_level_axes(impulses):
    ax = plot.doa_map(impulses, (0, 5), dynamic_range=30)

    assert ax.shape == (2,)
    assert len(ax[0].patches) == 1
    npt.assert_allclose(ax[0].patches[0].center, [0, 0], atol=1e-15)
    npt.assert_allclose(ax[0].patches[0].radius, 0.035)
    assert ax[1].get_ylim() == (-30, 0)


def test_doa_map_selects_samples_by_ceiling_the_limits():
    # Ceil of the limits as one-based, inclusive sample numbers: 5 to 10 ms
    # are the samples 239 to 479.
    pressure = np.ones(1000)
    result = make_result(pressure, np.tile([1.0, 0, 0], (1000, 1)))

    ax = plot.doa_map(result, (5, 10))

    assert len(ax[0].patches) == 479 - 239 + 1


def test_doa_map_skips_samples_without_a_direction_or_below_the_range():
    pressure = np.array([1.0, 1.0, 1e-3, 1.0])
    directions = np.array([[1, 0, 0], [np.nan, np.nan, np.nan], [1, 0, 0],
                           [0, 1, 0]])

    ax = plot.doa_map(make_result(pressure, directions), (0, 1),
                      dynamic_range=40)

    assert len(ax[0].patches) == 2


def test_doa_map_plots_on_given_axes(impulses):
    _, given = plt.subplots(1, 2)

    ax = plot.doa_map(impulses, (0, 5), ax=given)

    assert ax[0] is given[0]
    assert ax[1] is given[1]


def test_doa_map_marks_the_direction_at_the_peak(impulses):
    # Two equal peaks: the first, from the front, is marked.
    ax = plot.doa_map(impulses, (5, 10))

    npt.assert_allclose(ax[0].lines[-1].get_xydata(), [[0, 0]], atol=1e-15)


def test_warning_if_the_peak_has_no_direction():
    directions = np.array([[1, 0, 0], [np.nan] * 3, [1, 0, 0]])
    result = make_result([0.5, 1.0, 0.5], directions)

    with pytest.warns(UserWarning, match='direction at the peak'):
        ax = plot.doa_map(result, (0, 1))

    assert all(line.get_marker() == 'None' for line in ax[0].lines)


def test_doa_map_colours_the_dots_by_level(impulses):
    ax = plot.doa_map(impulses, (5, 10), cmap='turbo')

    # The loudest dot, at the peak, takes the top of the colormap.
    loudest = max(ax[0].patches, key=lambda dot: dot.radius)
    npt.assert_allclose(loudest.get_facecolor()[:3],
                        plt.get_cmap('turbo')(1.0)[:3])
    assert ax.shape == (3,)


def test_doa_map_colorbar_spans_the_sphere(impulses):
    ax = plot.doa_map(impulses, (5, 10), cmap='turbo')
    ax[0].figure.canvas.draw()

    to_figure = ax[0].transData + ax[0].figure.transFigure.inverted()
    poles = to_figure.transform([(0, -0.5), (0, 0.5)])[:, 1]
    bar = ax[2].get_position()
    npt.assert_allclose([bar.y0, bar.y1], poles, atol=1e-12)


def test_doa_map_without_the_direct_sound(impulses):
    ax = plot.doa_map(impulses, (5, 10), direct_sound=False)

    assert all(line.get_marker() == 'None' for line in ax[0].lines)
    assert ax[0].get_legend() is None


def test_doa_map_ends_the_level_axis_at_the_last_visible_sample():
    # Visible samples end at sample 480 (10 ms); zeros follow to 100 ms.
    pressure = np.zeros(4800)
    pressure[[100, 480]] = [1.0, 0.1]
    result = make_result(pressure, np.tile([1.0, 0, 0], (4800, 1)))

    npt.assert_allclose(plot.doa_map(result, (0, 5))[1].get_xlim(),
                        (0, 10.5))
    # The window starts at the onset, sample 52.
    npt.assert_allclose(plot.doa_map(result, (0, 50))[1].get_xlim(),
                        (0, 1.05 * (52 / 48 + 50)))


def test_doa_map_counts_the_window_from_the_onset():
    # The pressure crosses a fifth of the peak at sample 99; the onset is
    # 1 ms earlier, at sample 51, so (0, 1) ms selects samples 50 to 98.
    pressure = np.full(1000, 0.1)
    pressure[99:] = 1.0
    result = make_result(pressure, np.tile([1.0, 0, 0], (1000, 1)))

    ax = plot.doa_map(result, (0, 1))

    assert len(ax[0].patches) == 49
    npt.assert_allclose(ax[1].patches[-1].get_x(), 51 / 48)


# -- errors -------------------------------------------------------------------

@pytest.mark.parametrize('function', [plot.spatio_temporal,
                                      plot.time_frequency])
@pytest.mark.parametrize('result', [np.zeros(8), []])
def test_error_if_result_is_not_an_analysis_result(function, result):
    with pytest.raises(TypeError,
                       match='requires a pyspar.parametric.TimeResult'):
        function(result)


def test_error_if_doa_map_gets_several_results(impulses):
    with pytest.raises(TypeError, match='doa_map requires a pyspar'):
        plot.doa_map([impulses], (0, 5))


def test_error_if_results_differ_in_sampling_rate(impulses):
    other = make_result(impulses.pressure.time[0], impulses.doa.cartesian,
                        44100)

    with pytest.raises(ValueError, match='different sampling rates'):
        plot.spatio_temporal([impulses, other])


def test_error_if_pressure_is_silent():
    result = make_result(np.zeros(100), np.full((100, 3), np.nan))

    with pytest.raises(ValueError, match='pressure of the result is silent'):
        plot.time_frequency(result)


def test_error_if_plane_is_unknown(impulses):
    with pytest.raises(ValueError, match="plane is 'frontal'"):
        plot.spatio_temporal(impulses, plane='frontal')


@pytest.mark.parametrize('function', [plot.spatio_temporal,
                                      plot.time_frequency])
def test_error_if_integration_is_unknown(impulses, function):
    with pytest.raises(ValueError, match="integration is 'both'"):
        function(impulses, integration='both')


@pytest.mark.parametrize('times', [[], [-5, 10], [[5, 10]]])
def test_error_if_times_are_invalid(impulses, times):
    with pytest.raises(ValueError, match='times must be'):
        plot.time_frequency(impulses, times=times)


@pytest.mark.parametrize('dynamic_range', [0, -30])
def test_error_if_dynamic_range_is_not_positive(impulses, dynamic_range):
    with pytest.raises(ValueError, match='dynamic_range is'):
        plot.spatio_temporal(impulses, dynamic_range=dynamic_range)


@pytest.mark.parametrize('dynamic_range', [True, '45'])
def test_error_if_dynamic_range_is_not_a_number(impulses, dynamic_range):
    with pytest.raises(TypeError, match='dynamic_range must be a number'):
        plot.spatio_temporal(impulses, dynamic_range=dynamic_range)


def test_error_if_resolution_does_not_divide_180_degrees(impulses):
    with pytest.raises(ValueError, match='resolution is 7'):
        plot.spatio_temporal(impulses, resolution=7)


def test_error_if_spatio_temporal_axes_are_not_polar(impulses):
    _, given = plt.subplots()

    with pytest.raises(ValueError, match='ax must be a polar axes'):
        plot.spatio_temporal(impulses, ax=given)


def test_error_if_colors_do_not_match_the_times(impulses):
    with pytest.raises(ValueError, match='2 colors for 3 times'):
        plot.time_frequency(impulses, times=(1, 5, 20), colors=['r', 'b'])


def test_error_if_num_fractions_is_below_one(impulses):
    with pytest.raises(ValueError, match='num_fractions is 0'):
        plot.time_frequency(impulses, num_fractions=0)


@pytest.mark.parametrize('num_fractions', [1.5, True])
def test_error_if_num_fractions_is_not_an_integer(impulses, num_fractions):
    with pytest.raises(TypeError, match='num_fractions must be an integer'):
        plot.time_frequency(impulses, num_fractions=num_fractions)


@pytest.mark.parametrize('time_window', [5, (5, 10, 20), (20, 5)])
def test_error_if_time_window_is_not_a_start_and_an_end(impulses,
                                                        time_window):
    with pytest.raises(ValueError, match='time_window must hold'):
        plot.doa_map(impulses, time_window)


def test_error_if_doa_map_does_not_get_two_axes(impulses):
    _, given = plt.subplots()

    with pytest.raises(ValueError, match='ax must hold two axes'):
        plot.doa_map(impulses, (0, 5), ax=given)


def test_error_if_direct_sound_is_not_a_bool(impulses):
    with pytest.raises(TypeError, match='direct_sound must be a bool'):
        plot.doa_map(impulses, (0, 5), direct_sound=1)
