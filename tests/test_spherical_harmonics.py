"""Tests of the check of spherical harmonic signals."""
import numpy as np
import pyfar as pf
import pytest
from conftest import sh_signal
from spharpy import SphericalHarmonicSignal

from pyspar.utils.spherical_harmonics import _check_sh_input


@pytest.mark.parametrize(('signal', 'error', 'message'), [
    (pf.Signal(np.zeros((4, 16)), 48000), TypeError,
     'Caller requires a spharpy.SphericalHarmonicSignal; got Signal'),
    (SphericalHarmonicSignal(
        np.zeros((4, 16), dtype=complex), 48000, 'complex', 'N3D', 'ACN',
        False, is_complex=True), ValueError,
     'Caller requires a real signal without the Condon-Shortley phase; '
     'got complex, condon_shortley=False'),
    (SphericalHarmonicSignal(
        np.zeros((4, 16)), 48000, 'real', 'N3D', 'ACN', True), ValueError,
     'got real, condon_shortley=True'),
    (sh_signal(np.zeros((2, 4, 16))), ValueError,
     r'Caller expects a single spherical harmonic signal, got cshape '
     r'\(2, 4\)'),
    (sh_signal(np.zeros((1, 16))), ValueError,
     'signal has order 0 but Caller needs at least order 1')])
def test_check_sh_input_errors(signal, error, message):
    with pytest.raises(error, match=message):
        _check_sh_input(signal, 'Caller', 1)


def test_check_sh_input_error_if_the_coefficients_are_not_on_the_last_axis(
        monkeypatch):
    signal = sh_signal(np.zeros((4, 16)))
    monkeypatch.setattr(type(signal), 'caxis_spherical_harmonics', (-2,),
                        raising=False)

    with pytest.raises(
            ValueError,
            match=r'Caller expects the spherical harmonic coefficients on '
                  r'the last channel axis; got '
                  r'caxis_spherical_harmonics=\(-2,\)'):
        _check_sh_input(signal, 'Caller', 1)


@pytest.mark.parametrize('channel_convention', ['ACN', 'FuMa'])
@pytest.mark.parametrize('normalization', [
    'N3D', 'SN3D', 'NM', 'SNM', 'maxN'])
def test_check_sh_input_accepts_every_convention(
        channel_convention, normalization):
    signal = sh_signal(np.zeros((4, 16)))
    signal.channel_convention = channel_convention
    signal.normalization = normalization

    _check_sh_input(signal, 'Caller', 1)
