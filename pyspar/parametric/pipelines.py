"""Pipelines chaining an analyzer, a synthesizer and equalizers."""
from __future__ import annotations

from .analyzers import PseudoIntensityVectorAnalyzer
from .classes import AnalyzerBase, Postprocessor, SynthesizerBase
from .synthesizers import LoudspeakerSynthesizer


class SDMPipeline:
    """Run an SDM analysis, a synthesizer and then equalizers in order.

    The analysis runs once and its result is handed to the synthesizer and to
    every equalizer, so an equalizer that needs a reference, such as the
    measured pressure, takes it from there.

    Parameters
    ----------
    analyzer : pyspar.parametric.AnalyzerBase
        Estimates the direction of arrival per sample, for example a
        :class:`~pyspar.parametric.PseudoIntensityVectorAnalyzer`.
    synthesizer : pyspar.parametric.SynthesizerBase
        Renders the analysis result.
    postprocessors : sequence of pyspar.parametric.Postprocessor, optional
        Equalizers applied to the rendered signal in the given order. Only
        output-stage equalizers belong here; loudspeaker-stage ones act on the
        loudspeaker feeds inside a synthesizer and raise.

    Attributes
    ----------
    analyzer : pyspar.parametric.AnalyzerBase
        The analyzer.
    synthesizer : pyspar.parametric.SynthesizerBase
        The synthesizer.
    postprocessors : list of pyspar.parametric.Postprocessor
        The equalizers.
    """

    def __init__(self, analyzer, synthesizer, postprocessors=()):
        if not isinstance(analyzer, AnalyzerBase):
            raise TypeError(
                "analyzer must be a pyspar.parametric.AnalyzerBase; got "
                f"{type(analyzer).__name__}.")
        if not isinstance(synthesizer, SynthesizerBase):
            raise TypeError(
                "synthesizer must be a pyspar.parametric.SynthesizerBase; got "
                f"{type(synthesizer).__name__}.")

        postprocessors = list(postprocessors)
        for postprocessor in postprocessors:
            if not isinstance(postprocessor, Postprocessor):
                raise TypeError(
                    "postprocessors must be pyspar.parametric.Postprocessor "
                    f"objects; got {type(postprocessor).__name__}.")
            if postprocessor.stage != 'output':
                # Applied here it would act on whatever the synthesizer
                # returns, which for a binaural rendering is the BRIR.
                raise ValueError(
                    f"{type(postprocessor).__name__} is a "
                    f"'{postprocessor.stage}'-stage postprocessor and cannot "
                    "run after the synthesizer.")

        self.analyzer = analyzer
        self.synthesizer = synthesizer
        self.postprocessors = postprocessors

    def __call__(self, signal):
        """Analyze, render and equalize `signal`.

        Parameters
        ----------
        signal : spharpy.SphericalHarmonicSignal, pyfar.Signal
            Measured signal, as accepted by the analyzer.

        Returns
        -------
        signal : pyfar.Signal
            Rendered and equalized signal.
        """
        result = self.analyzer(signal)
        rendered = self.synthesizer(result)
        for postprocessor in self.postprocessors:
            rendered = postprocessor(rendered, result=result)
        return rendered


def sdm_loudspeakers(signal, loudspeakers, *, window_length, panning='nls',
                     dimensionality=3, lfe_channels=None, lfe_cutoff=80):
    """Render a measured impulse response to loudspeakers with SDM.

    Analysis and rendering in one call; for several renders of the same
    measurement, analyze once with a
    :class:`~pyspar.parametric.PseudoIntensityVectorAnalyzer` and reuse the
    result.

    Parameters
    ----------
    signal : spharpy.SphericalHarmonicSignal
        Measured impulse response, a spherical harmonic signal of order one
        or higher. Its sampling rate configures the analysis.
    loudspeakers : pyfar.Coordinates
        Loudspeaker directions.
    window_length : int
        Window of the analysis, in samples, see
        :class:`~pyspar.parametric.PseudoIntensityVectorAnalyzer`.
    panning : ``'nls'``, optional
        Panning law, see :class:`~pyspar.parametric.LoudspeakerSynthesizer`.
        The default is ``'nls'``.
    dimensionality : ``3``, ``2``, optional
        Dimensionality of the layout, see
        :class:`~pyspar.parametric.LoudspeakerSynthesizer`. The default is
        ``3``.
    lfe_channels : int, sequence of int, optional
        Indices of the LFE channels in `loudspeakers`, see
        :class:`~pyspar.parametric.LoudspeakerSynthesizer`. The default is
        ``None``.
    lfe_cutoff : float, optional
        Crossover frequency of the LFE channels in Hz. The default is ``80``.

    Returns
    -------
    loudspeaker_signals : pyfar.Signal
        Loudspeaker signals with ``cshape (n_loudspeakers,)``.
    """
    pipeline = SDMPipeline(
        PseudoIntensityVectorAnalyzer(window_length=window_length),
        LoudspeakerSynthesizer(loudspeakers, panning=panning,
                               dimensionality=dimensionality,
                               lfe_channels=lfe_channels,
                               lfe_cutoff=lfe_cutoff))
    return pipeline(signal)
