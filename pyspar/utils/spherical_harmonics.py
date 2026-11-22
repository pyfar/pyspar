"""Checks of spherical harmonic signals."""
import numpy as np
from spharpy import SphericalHarmonicSignal


def _check_sh_input(signal, caller, order):
    """Raise an error unless `signal` can be read by `caller`.

    Parameters
    ----------
    signal : spharpy.SphericalHarmonicSignal
        One real signal without the Condon-Shortley phase, of order `order`
        or higher, with the coefficients on the last channel axis.
    caller : str
        Name of the calling class, for the messages.
    order : int
        Lowest spherical harmonic order `caller` needs.
    """
    if not isinstance(signal, SphericalHarmonicSignal):
        raise TypeError(
            f"{caller} requires a spharpy.SphericalHarmonicSignal; got "
            f"{type(signal).__name__}.")
    if signal.basis_type != 'real' or signal.condon_shortley:
        raise ValueError(
            f"{caller} requires a real signal without the Condon-Shortley "
            f"phase; got {signal.basis_type}, condon_shortley="
            f"{signal.condon_shortley}.")
    # spharpy 1.1 can hold the coefficients on another channel axis.
    caxis = getattr(signal, 'caxis_spherical_harmonics', (-1,))
    if caxis != (-1,):
        raise ValueError(
            f"{caller} expects the spherical harmonic coefficients on the "
            f"last channel axis; got caxis_spherical_harmonics={caxis}.")
    if np.prod(signal.cshape[:-1]) != 1:
        raise ValueError(
            f"{caller} expects a single spherical harmonic signal, got "
            f"cshape {signal.cshape}.")
    if signal.n_max < order:
        raise ValueError(
            f"signal has order {signal.n_max} but {caller} needs at least "
            f"order {order}.")
