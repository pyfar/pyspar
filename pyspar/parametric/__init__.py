"""Parametric spatial audio rendering.

Signal-dependent methods: an analyzer estimates spatial parameters of a
measurement, a synthesizer renders them, and postprocessors correct the
loudspeaker signals inside the synthesizer or the rendered signal.
"""
from .classes import TimeResult
from .analyzers import PseudoIntensityVectorAnalyzer
from .synthesizers import LoudspeakerSynthesizer


__all__ = [
    'LoudspeakerSynthesizer',
    'PseudoIntensityVectorAnalyzer',
    'TimeResult',
]
