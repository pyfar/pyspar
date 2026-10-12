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
* ``TimeDifferenceOfArrivalAnalyzer`` estimating one direction of arrival per
  sample from the time differences of arrival at an open microphone array,
  with the frame of the array by default and an error for shorter ones.
* ``AmbisonicSynthesizer`` upmixing an analysis to higher-order Ambisonics
  with the Ambisonic Spatial Decomposition Method, and ``passband`` on
  ``PseudoIntensityVectorAnalyzer`` estimating the directions from the
  band-passed signal.
* ``pyspar.parametric.plot`` with ``doa_map``, ``spatio_temporal`` and
  ``time_frequency``, plotting an analysis as a map of the directions of
  arrival, as directional energy in growing time windows, and as spectra in
  growing time windows.
* ``SpectralDecayCorrection``, the spectral decay correction of ASDM
  upmixing, matching the third-octave energy decay of every order to that
  of the analyzed signal, applied by ``AmbisonicSynthesizer`` unless
  ``spectral_decay_correction=False``.
* ``LoudspeakerSpectrumEqualization``, the post equalization of the
  loudspeaker spectra in octave bands and short-time frames, matching the
  sum of the loudspeaker spectra to the pressure, available as
  ``correction='loudspeaker_spectrum'`` of the loudspeaker and binaural
  synthesizers.
* Each synthesizer accepts a correction only where it applies.
* ``SpectralEqualization`` matching the spectrum of the loudspeaker signals to
  that of the pressure over time with minimum- or linear-phase filters,
  available as ``correction='minphase'`` or ``'linphase'`` of the
  loudspeaker and binaural synthesizers.
* ``RoughnessCompensation`` smoothing the late part of the sparse
  loudspeaker signals of nearest-loudspeaker panning, from the estimated or
  a given mixing time, or from the onset, with Schroeder allpasses or with a
  velvet filter per loudspeaker, available as the ``roughness`` argument of
  the loudspeaker and binaural synthesizers.

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
