from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class DimensionlessParams(NamedTuple):
    """Dimensionless parameters for the ADE analytical solution.

    Attributes
    ----------
    Z : ndarray
        Dimensionless depth (-), Z = z / L.
    T : ndarray
        Dimensionless time (-), T = t * v / L.
    T0 : float
        Dimensionless pulse duration (-), T0 = t_pulse * v / L.
    P : float
        Péclet number (-), P = v * L / D.
    ws : float or None
        Damköhler number for kinetic sorption (-).
        None when kinetic=False.
    """

    Z: NDArray[np.float64]
    T: NDArray[np.float64]
    T0: float
    P: float
    ws: float | None


def compute_dimensionless_params(
    grid,
    boundary_conditions,
    hydro_properties,
    adsorption=None,
    kinetic: bool = False,
) -> DimensionlessParams:
    """Compute dimensionless parameters for the ADE analytical solution.

    Converts physical grid, flow, and transport parameters into the
    dimensionless form required by the equilibrium and kinetic solvers.

    Parameters
    ----------
    grid : SimulationGrid
        Spatial and temporal discretization. Must have `.depth` and `.time`
        arrays (m and s respectively).
    boundary_conditions : BoundaryConditions
        Contaminant source boundary conditions. Must have `.pulse_time` (s).
    hydro_properties : HydrologicalProperties
        Hydrological properties. Must have `.pore_velocity` (m/s) and
        `.dispersion_coefficient` (m²/s).
    adsorption : Adsorption, optional
        Adsorption parameters. Required when kinetic=True. Must have
        `.rate_const`, `.betas`, `.sp_retardation`.
    kinetic : bool, optional
        If True, also compute the Damköhler number `ws`. Default is False.

    Returns
    -------
    DimensionlessParams
        Named tuple containing Z, T, T0, P, and ws (None if kinetic=False).

    Raises
    ------
    ValueError
        If pore_velocity or dispersion_coefficient is zero.
    ValueError
        If kinetic=True but adsorption is None.

    Examples
    --------
    >>> params = compute_dimensionless_params(grid, bcs, hydro, kinetic=False)
    >>> C1, C_tot = equilibrium_solver(R, params.Z, params.T, params.P, params.T0, ...)
    """
    v = hydro_properties.pore_velocity
    D = hydro_properties.dispersion_coefficient

    if v == 0:
        raise ValueError("Pore velocity must be non-zero.")
    if D == 0:
        raise ValueError("Dispersion coefficient must be non-zero.")
    if kinetic and adsorption is None:
        raise ValueError("Adsorption parameters required when kinetic=True.")

    L = grid.depth[-1]

    Z = grid.depth / L
    T = grid.time * (v / L)
    T0 = boundary_conditions.pulse_time * (v / L)
    P = v * L / D

    ws = None
    if kinetic:
        ws = (
            adsorption.rate_const
            * (1 - adsorption.betas)
            * (1 + adsorption.sp_retardation)
            * L
            / v
        )

    return DimensionlessParams(Z=Z, T=T, T0=T0, P=P, ws=ws)