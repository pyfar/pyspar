"""Parametric spatial audio rendering.

Signal-dependent methods: an analyzer estimates spatial parameters of a
measurement, a synthesizer renders them, and postprocessors correct the
loudspeaker signals inside the synthesizer or the rendered signal.
"""
from .classes import (
    AnalyzerBase,
    Postprocessor,
    SynthesizerBase,
    TimeResult,
)
from .analyzers import (
    PseudoIntensityVectorAnalyzer, TimeDifferenceOfArrivalAnalyzer)
from .postprocessors import (
    LoudspeakerSpectrumEqualization, SpectralDecayCorrection)
from .synthesizers import (
    AmbisonicSynthesizer, BinauralSynthesizer, LoudspeakerSynthesizer)
from . import plot


__all__ = [
    'AmbisonicSynthesizer',
    'AnalyzerBase',
    'BinauralSynthesizer',
    'LoudspeakerSpectrumEqualization',
    'LoudspeakerSynthesizer',
    'plot',
    'Postprocessor',
    'PseudoIntensityVectorAnalyzer',
    'SpectralDecayCorrection',
    'SynthesizerBase',
    'TimeDifferenceOfArrivalAnalyzer',
    'TimeResult',
]
