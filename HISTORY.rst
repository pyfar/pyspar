=======
History
=======

[Unreleased]
------------

Added
^^^^^
* Subpackage ``pyspar.parametric`` for parametric rendering, with the data
  contract of its processing chain: ``TimeResult`` and the
  ``AnalyzerBase``, ``SynthesizerBase`` and ``Postprocessor`` bases.
* ``PseudoIntensityVectorAnalyzer`` estimating one direction of arrival per
  sample from the pseudo-intensity vector of a first-order signal, smoothed
  by a centred Hann window of odd length.
* ``LoudspeakerSynthesizer`` rendering an analysis to a loudspeaker array with
  nearest-loudspeaker panning, optionally weighted for horizontal layouts.
* ``SDMPipeline`` chaining analysis, rendering and equalization, and
  ``sdm_loudspeakers`` doing all three in one call for a spherical harmonic
  signal.
* ``pyspar.io.load_loudspeaker_array`` reading IEM-convention JSON layouts,
  dropping imaginary loudspeakers.
* Crossover to low-frequency effects (LFE) channels in
  ``LoudspeakerSynthesizer`` and ``sdm_loudspeakers``, set by
  ``lfe_channels`` and ``lfe_cutoff``.

Changed
^^^^^^^

Deprecated
^^^^^^^^^^

Removed
^^^^^^^

Fixed
^^^^^

0.1.0 (2026-07-20)
------------------

Added
^^^^^
* First release on PyPI.
