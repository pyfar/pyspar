"""Data contract and abstract bases of the parametric processing chain.

Methods with one direction of arrival per sample, the Spatial Decomposition
Method and its variants, are split into three stages, so that an analysis
can be reused by several synthesizers and a correction can be swapped without
touching either:

``Analyzer``
    Estimates spatial parameters from a measured signal and returns them
    as a result, such as :class:`TimeResult`.
``Synthesizer``
    Turns the result of an analysis into loudspeaker, binaural or Ambisonic
    signals.
``Postprocessor``
    Equalizes a rendered signal.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pyfar as pf


@dataclass
class TimeResult:
    """Spatial analysis with one direction of arrival per pressure sample.

    Attributes
    ----------
    pressure : pyfar.Signal
        Omnidirectional pressure impulse response, one channel.
    signal : pyfar.Signal
        The analyzed signal, kept for synthesizers that need more than the
        pressure: the first-order spharpy.SphericalHarmonicSignal in ACN/N3D
        for the pseudo-intensity vector.
    doa : pyfar.Coordinates, cshape (n_samples,)
        Direction of arrival per sample of `pressure`; samples without a
        direction are NaN.
    sampling_rate : float
        Sampling rate of `signal` in Hz.
    """

    pressure: pf.Signal
    signal: pf.Signal
    doa: pf.Coordinates

    def __post_init__(self):
        """Check that pressure, directions and signal belong together.

        `pressure` must be a single-channel signal, `doa` must hold one
        direction per pressure sample, with a cshape of ``(n_samples,)`` and
        NaN rather than a zero vector where a sample has none, and `signal`
        must have the sampling rate and the number of samples of the
        pressure.
        """
        if not isinstance(self.pressure, pf.Signal):
            raise TypeError(
                "pressure must be a pyfar.Signal; got "
                f"{type(self.pressure).__name__}.")
        if self.pressure.cshape != (1,):
            raise ValueError(
                "pressure must have one channel, got cshape "
                f"{self.pressure.cshape}.")
        n_samples = self.pressure.n_samples
        if not isinstance(self.doa, pf.Coordinates):
            raise TypeError(
                "doa must be a pyfar.Coordinates; got "
                f"{type(self.doa).__name__}.")
        # Shape, not size: directions per frame and frequency bin can
        # number as many as the samples.
        if self.doa.cshape != (n_samples,):
            raise ValueError(
                "TimeResult needs one direction per pressure sample: doa "
                f"has cshape {self.doa.cshape} for {n_samples} pressure "
                "samples.")
        # pyfar reads a zero vector as straight ahead, so it cannot mark a
        # sample without a direction.
        norm = np.linalg.norm(
            np.atleast_2d(self.doa.cartesian).reshape(-1, 3), axis=1)
        if np.any(norm == 0):
            raise ValueError(
                f"doa has {int(np.sum(norm == 0))} zero vectors; "
                "samples without a direction must be NaN.")
        if not isinstance(self.signal, pf.Signal):
            raise TypeError(
                "signal must be a pyfar.Signal; got "
                f"{type(self.signal).__name__}.")
        if self.signal.n_samples != n_samples:
            raise ValueError(
                f"signal has {self.signal.n_samples} samples for "
                f"{n_samples} pressure samples.")
        if self.signal.sampling_rate != self.pressure.sampling_rate:
            raise ValueError(
                "signal has a sampling rate of "
                f"{self.signal.sampling_rate} Hz but the pressure "
                f"{self.pressure.sampling_rate} Hz.")

    @property
    def sampling_rate(self):
        """Sampling rate of `signal` in Hz."""
        return float(self.signal.sampling_rate)


class AnalyzerBase(ABC):
    """Abstract base class of the spatial analyzers.

    Its subclasses return a result that the synthesizers render, such as a
    :class:`TimeResult`.
    """

    @abstractmethod
    def __call__(self, signal):
        """Analyze `signal` and return its result."""


class SynthesizerBase(ABC):
    """Abstract base class of the synthesizers."""

    @abstractmethod
    def __call__(self, result):
        """Render `result` and return the rendered signal."""


class Postprocessor(ABC):
    """Abstract base class of the equalizers.

    Attributes
    ----------
    stage : str
        Signal domain the equalizer acts on. It is a fixed property of each
        subclass rather than a user choice: ``'loudspeaker'`` for corrections
        applied to the loudspeaker feeds, which run inside a synthesizer, and
        ``'output'`` for methods defined on the rendered signal.
    """

    stage = 'output'

    @abstractmethod
    def __call__(self, signal, result=None):
        """Equalize `signal` and return the result."""
