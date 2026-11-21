=======
History
=======

[Unreleased]
------------

Added
^^^^^
* Subpackage ``pyspar.parametric`` for parametric rendering, with the base
  classes of its processing chain, ``AnalyzerBase``, ``SynthesizerBase``
  and ``Postprocessor``, and the analysis result ``TimeResult``.
* ``PseudoIntensityVectorAnalyzer`` estimating one direction of arrival per
  sample from the pseudo-intensity vector of a first-order signal, smoothed
  by a centred Hann window of odd length.
* ``LoudspeakerSynthesizer`` rendering an analysis to a loudspeaker array with
  nearest-loudspeaker panning, optionally weighted for horizontal layouts.
* ``pyspar.io.load_loudspeaker_array`` reading IEM-convention JSON layouts
  in the order of their channel numbers, returning the imaginary
  loudspeakers on request with ``return_imaginary``, and plotting the
  layout with ``plot=True``.
* Crossover to low-frequency effects (LFE) channels in
  ``LoudspeakerSynthesizer``, set by ``lfe_channels`` and ``lfe_cutoff``.
* ``BinauralSynthesizer`` rendering an analysis to ear signals through
  virtual loudspeakers.

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
