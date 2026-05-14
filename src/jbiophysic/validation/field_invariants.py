"""Field-invariant utilities with explicit current-density layout handling."""

from __future__ import annotations

from typing import Literal

import numpy as np


def normalize_current_density_layout(
    J_e, *, layout: Literal["channel_first", "channel_last"] = "channel_first"
) -> np.ndarray:
    """Return current density in channel-first layout ``[T,3,Nx,Ny,Nz]`` or ``[3,Nx,Ny,Nz]``.

    The default TFNE manifest layout is channel-first. Accelerator kernels may use
    channel-last, but derivative/CSD operators must explicitly declare and convert it.
    """
    arr = np.asarray(J_e)
    if layout == "channel_first":
        if arr.ndim == 4 and arr.shape[0] == 3:
            return arr
        if arr.ndim == 5 and arr.shape[1] == 3:
            return arr
        raise ValueError("channel_first J_e must have shape [3,Nx,Ny,Nz] or [T,3,Nx,Ny,Nz]")
    if layout == "channel_last":
        if arr.ndim == 4 and arr.shape[-1] == 3:
            return np.moveaxis(arr, -1, 0)
        if arr.ndim == 5 and arr.shape[-1] == 3:
            return np.moveaxis(arr, -1, 1)
        raise ValueError("channel_last J_e must have shape [Nx,Ny,Nz,3] or [T,Nx,Ny,Nz,3]")
    raise ValueError("layout must be channel_first or channel_last")
