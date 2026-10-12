"""Which corrections each synthesizer accepts.

Every correction is tried in every place it can be passed: those that
belong there are accepted, the others are rejected when the synthesizer is
built, before anything is rendered.
"""
import pytest

from pyspar import parametric


# -- the correction of the loudspeaker synthesizer ----------------------------

@pytest.mark.parametrize(('correction', 'kind'), [
    (None, type(None)),
    ('minphase', parametric.SpectralEqualization),
    ('linphase', parametric.SpectralEqualization),
    ('loudspeaker_spectrum', parametric.LoudspeakerSpectrumEqualization),
    (parametric.SpectralEqualization(), parametric.SpectralEqualization),
    (parametric.LoudspeakerSpectrumEqualization(),
     parametric.LoudspeakerSpectrumEqualization)])
def test_loudspeaker_synthesizer_accepts(layout_horizontal, correction, kind):
    synthesizer = parametric.LoudspeakerSynthesizer(layout_horizontal,
                                                    correction=correction)

    assert isinstance(synthesizer.correction, kind)


def test_loudspeaker_spectrum_string_gives_the_default_equalizer(
        layout_horizontal):
    correction = parametric.LoudspeakerSynthesizer(
        layout_horizontal, correction='loudspeaker_spectrum').correction

    assert (correction.periods, correction.max_boost_db) == (7, 6.0)


@pytest.mark.parametrize(('correction', 'error', 'match'), [
    ('flat', ValueError,
     "correction is 'flat' but must be 'minphase', 'linphase', "
     "'loudspeaker_spectrum'"),
    (parametric.SpectralDecayCorrection(), ValueError,
     'corrects Ambisonic signals'),
    (parametric.RoughnessCompensation(), TypeError, 'correction must be'),
    (0.5, TypeError, 'correction must be')])
def test_loudspeaker_synthesizer_rejects(layout_horizontal, correction,
                                         error, match):
    with pytest.raises(error, match=match):
        parametric.LoudspeakerSynthesizer(layout_horizontal,
                                          correction=correction)


# -- the correction of the binaural synthesizer -------------------------------

@pytest.mark.parametrize(('correction', 'kind'), [
    (None, type(None)),
    ('minphase', parametric.SpectralEqualization),
    ('linphase', parametric.SpectralEqualization),
    ('loudspeaker_spectrum', parametric.LoudspeakerSpectrumEqualization),
    (parametric.LoudspeakerSpectrumEqualization(),
     parametric.LoudspeakerSpectrumEqualization)])
def test_binaural_synthesizer_accepts(hrir_set, correction, kind):
    synthesizer = parametric.BinauralSynthesizer(*hrir_set,
                                                 correction=correction)

    assert isinstance(synthesizer.correction, kind)


@pytest.mark.parametrize(('correction', 'error', 'match'), [
    ('flat', ValueError, "correction is 'flat' but must be"),
    (parametric.SpectralDecayCorrection(), ValueError,
     'corrects Ambisonic signals'),
    (parametric.RoughnessCompensation(), TypeError, 'correction must be')])
def test_binaural_synthesizer_rejects(hrir_set, correction, error, match):
    with pytest.raises(error, match=match):
        parametric.BinauralSynthesizer(*hrir_set, correction=correction)


# -- the roughness compensation of both ---------------------------------------

@pytest.mark.parametrize('roughness', [
    None, 'schroeder', 'velvet', parametric.RoughnessCompensation()])
def test_synthesizers_accept_a_roughness_compensation(
        layout_horizontal, hrir_set, roughness):
    for synthesizer in (
            parametric.LoudspeakerSynthesizer(layout_horizontal,
                                              roughness=roughness),
            parametric.BinauralSynthesizer(*hrir_set, roughness=roughness)):
        assert isinstance(synthesizer.roughness,
                          (type(None), parametric.RoughnessCompensation))


@pytest.mark.parametrize(('roughness', 'error', 'match'), [
    ('minphase', ValueError, "mode is 'minphase'"),
    (parametric.SpectralEqualization(), TypeError, 'roughness must be'),
    (parametric.SpectralDecayCorrection(), TypeError, 'roughness must be')])
def test_synthesizers_reject_as_roughness(layout_horizontal, hrir_set,
                                          roughness, error, match):
    with pytest.raises(error, match=match):
        parametric.LoudspeakerSynthesizer(layout_horizontal,
                                          roughness=roughness)
    with pytest.raises(error, match=match):
        parametric.BinauralSynthesizer(*hrir_set, roughness=roughness)
