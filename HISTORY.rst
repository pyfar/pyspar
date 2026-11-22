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
