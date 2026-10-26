"""Tests of the data contract."""
import numpy as np
import pyfar as pf
import pytest

from pyspar import parametric


def silence(n_samples=64, sampling_rate=48000):
    return pf.Signal(np.zeros(n_samples), sampling_rate)


def test_time_result_holds_the_estimated_parameters(frontal_impulse):
    doa = parametric.PseudoIntensityVectorAnalyzer()(frontal_impulse).doa
    pressure = silence()
    result = parametric.TimeResult(
        pressure=pressure, doa=doa, signal=frontal_impulse)

    assert result.pressure is pressure
    assert result.doa is doa
    assert result.signal is frontal_impulse
    assert result.sampling_rate == 48000


def test_time_result_takes_the_sampling_rate_from_the_signal():
    doa = pf.Coordinates(*np.ones((3, 64)))

    result = parametric.TimeResult(
        pressure=silence(sampling_rate=44100), doa=doa,
        signal=silence(sampling_rate=44100))

    assert result.sampling_rate == 44100.0


def test_time_result_error_if_the_signal_is_not_a_signal():
    doa = pf.Coordinates(*np.ones((3, 64)))

    with pytest.raises(TypeError, match='signal must be a pyfar.Signal'):
        parametric.TimeResult(pressure=silence(), doa=doa,
                              signal=np.zeros(64))


@pytest.mark.parametrize('base', [
    parametric.AnalyzerBase, parametric.SynthesizerBase,
    parametric.Postprocessor])
def test_bases_cannot_be_instantiated(base):
    with pytest.raises(TypeError, match='abstract'):
        base()


def test_postprocessor_acts_on_the_output_by_default():
    assert parametric.Postprocessor.stage == 'output'


def test_time_result_error_if_the_pressure_is_not_a_signal(frontal_impulse):
    doa = parametric.PseudoIntensityVectorAnalyzer()(frontal_impulse).doa

    with pytest.raises(TypeError, match='pressure must be a pyfar.Signal'):
        parametric.TimeResult(pressure=np.zeros(64), doa=doa,
                              signal=frontal_impulse)


def test_time_result_error_if_the_pressure_has_several_channels(
        frontal_impulse):
    doa = parametric.PseudoIntensityVectorAnalyzer()(frontal_impulse).doa

    with pytest.raises(ValueError,
                       match=r'pressure must have one channel, got cshape '
                             r'\(2,\)'):
        parametric.TimeResult(pressure=pf.Signal(np.zeros((2, 64)), 48000),
                              doa=doa, signal=frontal_impulse)


def test_time_result_error_if_the_doa_is_not_coordinates(frontal_impulse):
    with pytest.raises(TypeError, match='doa must be a pyfar.Coordinates'):
        parametric.TimeResult(pressure=silence(), doa=np.zeros((64, 3)),
                              signal=frontal_impulse)


def test_time_result_error_if_a_direction_is_missing(frontal_impulse):
    doa = pf.Coordinates(*np.ones((3, 63)))

    with pytest.raises(ValueError,
                       match=r'cshape \(63,\) for 64 pressure samples'):
        parametric.TimeResult(pressure=silence(), doa=doa,
                              signal=frontal_impulse)


def test_time_result_error_for_a_zero_vector(frontal_impulse):
    directions = np.ones((64, 3))
    directions[[5, 9]] = 0.0

    with pytest.raises(ValueError,
                       match='2 zero vectors; samples without a direction '
                             'must be NaN'):
        parametric.TimeResult(pressure=silence(),
                              doa=pf.Coordinates(*directions.T),
                              signal=frontal_impulse)


def test_time_result_error_for_a_time_frequency_field(frontal_impulse):
    # 8 bins x 8 frames holds as many directions as there are samples, so
    # only the shape tells the two apart.
    doa = pf.Coordinates(*np.ones((3, 8, 8)))

    with pytest.raises(ValueError, match=r'cshape \(8, 8\) for 64'):
        parametric.TimeResult(pressure=silence(), doa=doa,
                              signal=frontal_impulse)


def test_time_result_error_if_the_signal_length_disagrees(frontal_impulse):
    doa = pf.Coordinates(*np.ones((3, 32)))

    with pytest.raises(ValueError, match='signal has 64 samples for 32'):
        parametric.TimeResult(pressure=silence(32), doa=doa,
                              signal=frontal_impulse)


def test_time_result_error_if_the_sampling_rates_disagree(frontal_impulse):
    doa = pf.Coordinates(*np.ones((3, 64)))

    with pytest.raises(
            ValueError,
            match='sampling rate of 48000 Hz but the pressure 44100 Hz'):
        parametric.TimeResult(pressure=silence(sampling_rate=44100),
                              doa=doa, signal=frontal_impulse)
