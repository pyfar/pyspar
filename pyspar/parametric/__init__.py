"""Parametric spatial audio rendering.

Signal-dependent methods: an analyzer estimates spatial parameters of a
measurement, a synthesizer renders them, and postprocessors correct the
loudspeaker signals inside the synthesizer or the rendered signal.
"""
from .classes import TimeResult
from .analyzers import (
    PseudoIntensityVectorAnalyzer, TimeDifferenceOfArrivalAnalyzer)
from .synthesizers import (
    AmbisonicSynthesizer, BinauralSynthesizer, LoudspeakerSynthesizer)
from . import plot


__all__ = [
    'AmbisonicSynthesizer',
    'BinauralSynthesizer',
    'LoudspeakerSynthesizer',
    'plot',
    'PseudoIntensityVectorAnalyzer',
    'TimeDifferenceOfArrivalAnalyzer',
    'TimeResult',
]
