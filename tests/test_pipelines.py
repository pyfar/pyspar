"""Tests of the pipelines."""
import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
from conftest import sh_signal

from pyspar import parametric


class Scale(parametric.Postprocessor):
    """Dummy output-stage equalizer multiplying by a constant."""

    def __init__(self, factor, calls):
        self.factor = factor
        self.calls = calls

    def __call__(self, signal, result=None):
        """Record the call and scale `signal`."""
        self.calls.append((self.factor, result))
        # pyfar's Signal arithmetic runs in the frequency domain; scaling the
        # time data keeps the exact zeros of the render exact.
        return pf.Signal(signal.time * self.factor, signal.sampling_rate)


class FeedEqualizer(parametric.Postprocessor):
    """Dummy loudspeaker-stage equalizer."""

    stage = 'loudspeaker'

    def __call__(self, signal, result=None):  # noqa: ARG002
        """Return `signal` unchanged."""
        return signal


def test_pipeline_analyzes_then_renders(impulse_response, layout_surround):
    analyzer = parametric.PseudoIntensityVectorAnalyzer(window_length=5)
    synthesizer = parametric.LoudspeakerSynthesizer(layout_surround)

    rendered = parametric.SDMPipeline(analyzer, synthesizer)(impulse_response)

    expected = synthesizer(analyzer(impulse_response))
    npt.assert_array_equal(rendered.time, expected.time)


def test_postprocessors_run_in_order(impulse_response, layout_surround):
    calls = []
    pipeline = parametric.SDMPipeline(
        parametric.PseudoIntensityVectorAnalyzer(),
        parametric.LoudspeakerSynthesizer(layout_surround),
        [Scale(2.0, calls), Scale(3.0, calls)])

    rendered = pipeline(impulse_response)

    assert [factor for factor, _ in calls] == [2.0, 3.0]
    unscaled = parametric.SDMPipeline(
        parametric.PseudoIntensityVectorAnalyzer(),
        parametric.LoudspeakerSynthesizer(layout_surround))(impulse_response)
    npt.assert_allclose(rendered.time, 6.0 * unscaled.time)


def test_postprocessors_receive_the_analysis(impulse_response,
                                             layout_surround):
    calls = []
    pipeline = parametric.SDMPipeline(
        parametric.PseudoIntensityVectorAnalyzer(),
        parametric.LoudspeakerSynthesizer(layout_surround),
        [Scale(1.0, calls)])

    pipeline(impulse_response)

    result = calls[0][1]
    assert isinstance(result, parametric.TimeResult)
    npt.assert_array_equal(result.pressure.time[0],
                           np.asarray(impulse_response.time)[0, 0])


def test_error_if_a_loudspeaker_stage_postprocessor_is_given(
        layout_surround):
    with pytest.raises(ValueError, match="'loudspeaker'-stage postprocessor"):
        parametric.SDMPipeline(
            parametric.PseudoIntensityVectorAnalyzer(),
            parametric.LoudspeakerSynthesizer(layout_surround),
            [FeedEqualizer()])


def test_error_if_a_postprocessor_is_not_one(layout_surround):
    with pytest.raises(TypeError,
                       match='must be pyspar.parametric.Postprocessor'):
        parametric.SDMPipeline(
            parametric.PseudoIntensityVectorAnalyzer(),
            parametric.LoudspeakerSynthesizer(layout_surround),
            [lambda signal, result=None: signal])  # noqa: ARG005


def test_error_if_the_analyzer_is_not_one(layout_surround):
    with pytest.raises(TypeError, match='analyzer must be'):
        parametric.SDMPipeline(
            None, parametric.LoudspeakerSynthesizer(layout_surround))


def test_error_if_the_synthesizer_is_not_one():
    with pytest.raises(TypeError, match='synthesizer must be'):
        parametric.SDMPipeline(
            parametric.PseudoIntensityVectorAnalyzer(), None)


def test_sdm_loudspeakers_equals_the_two_stages(impulse_response,
                                                layout_surround):
    rendered = parametric.sdm_loudspeakers(
        impulse_response, layout_surround, window_length=5)

    result = parametric.PseudoIntensityVectorAnalyzer(window_length=5)(
        impulse_response)
    expected = parametric.LoudspeakerSynthesizer(layout_surround)(result)
    npt.assert_array_equal(rendered.time, expected.time)


def test_sdm_loudspeakers_applies_the_dimensionality(impulse_response,
                                                     layout_horizontal):
    rendered = parametric.sdm_loudspeakers(
        impulse_response, layout_horizontal, window_length=5,
        dimensionality=2)

    result = parametric.PseudoIntensityVectorAnalyzer(window_length=5)(
        impulse_response)
    expected = parametric.LoudspeakerSynthesizer(
        layout_horizontal, dimensionality=2)(result)
    npt.assert_array_equal(rendered.time, expected.time)


def test_sdm_loudspeakers_takes_the_sampling_rate_from_the_signal(
        layout_surround):
    signal = sh_signal(np.random.default_rng(1).standard_normal((4, 64)),
                       sampling_rate=44100)

    rendered = parametric.sdm_loudspeakers(signal, layout_surround,
                                           window_length=3)

    assert rendered.sampling_rate == 44100


@pytest.mark.filterwarnings('ignore:.*samples have no direction')
def test_sdm_loudspeakers_with_microphones_equals_the_two_stages(
        array_response, microphones, layout_surround):
    rendered = parametric.sdm_loudspeakers(
        array_response, layout_surround, window_length=40,
        microphones=microphones, reference_channel=4)

    result = parametric.TimeDifferenceOfArrivalAnalyzer(
        microphones, reference_channel=4, window_length=40)(array_response)
    expected = parametric.LoudspeakerSynthesizer(layout_surround)(result)
    npt.assert_array_equal(rendered.time, expected.time)


def test_sdm_loudspeakers_error_if_microphones_are_given_for_a_sh_signal(
        impulse_response, microphones, layout_surround):
    with pytest.raises(TypeError, match='requires the capsule signals'):
        parametric.sdm_loudspeakers(impulse_response, layout_surround,
                                    window_length=3, microphones=microphones,
                                    reference_channel=4)


def test_sdm_loudspeakers_error_if_capsule_signals_come_without_microphones(
        array_response, layout_surround):
    with pytest.raises(TypeError, match='requires a spharpy'):
        parametric.sdm_loudspeakers(array_response, layout_surround,
                                    window_length=3)


def test_sdm_loudspeakers_error_for_a_reference_channel_without_microphones(
        impulse_response, layout_surround):
    with pytest.raises(ValueError, match='without microphones'):
        parametric.sdm_loudspeakers(impulse_response, layout_surround,
                                    window_length=3, reference_channel=4)


def test_sdm_loudspeakers_error_if_microphones_come_without_a_capsule(
        array_response, microphones, layout_surround):
    with pytest.raises(TypeError,
                       match='reference_channel must be an integer; got None'):
        parametric.sdm_loudspeakers(array_response, layout_surround,
                                    window_length=40, microphones=microphones)


def test_sdm_loudspeakers_error_if_the_input_is_an_array(layout_surround):
    with pytest.raises(TypeError, match='requires a spharpy'):
        parametric.sdm_loudspeakers(np.zeros((64, 4)), layout_surround,
                                    window_length=3)
