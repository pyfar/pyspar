"""Conversion of spherical harmonic signals."""
import numpy as np
from spharpy import SphericalHarmonicSignal


def sh_to_n3d(signal, caller):
    """Validate a spherical harmonic signal and return its data in ACN/N3D.

    The conventions of `signal` are read from the object, so no method ever
    takes a normalization or channel convention argument. The order is kept;
    a method needing fewer channels truncates the result itself.

    Parameters
    ----------
    signal : spharpy.SphericalHarmonicSignal
        A single signal of order one or higher in any channel convention and
        normalization.
    caller : str
        Name used in the error messages.

    Returns
    -------
    data : numpy.ndarray, shape (n_samples, n_coefficients)
        Time data in ACN/N3D, samples along the first axis.
    """
    if not isinstance(signal, SphericalHarmonicSignal):
        raise TypeError(
            f"{caller} requires a spharpy.SphericalHarmonicSignal; got "
            f"{type(signal).__name__}.")
    if signal.basis_type != 'real':
        raise ValueError(
            f"{caller} requires a real spherical harmonic basis, got "
            f"{signal.basis_type!r}.")
    if signal.condon_shortley:
        # Every basis evaluated in pyspar comes from spharpy's real basis,
        # which omits the Condon-Shortley phase.
        raise ValueError(
            f"{caller} requires data without the Condon-Shortley phase.")
    if signal.n_max < 1:
        raise ValueError(
            f"{caller} needs at least order 1, got order {signal.n_max}.")
    # The last axis of cshape counts the spherical harmonic coefficients.
    if int(np.prod(signal.cshape[:-1])) != 1:
        raise ValueError(
            f"{caller} expects a single spherical harmonic signal, got cshape "
            f"{signal.cshape}.")

    sh = signal.copy()
    sh.channel_convention = 'ACN'
    sh.normalization = 'N3D'
    # Contiguous, so that every caller computes on the same memory layout
    # and results do not depend on the conventions of the input.
    return np.ascontiguousarray(
        np.asarray(sh.time, dtype=float).reshape(-1, sh.n_samples).T)
