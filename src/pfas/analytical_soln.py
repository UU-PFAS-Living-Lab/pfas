# noqa: N806
"""Analytical solution module for PFAS transport modeling.

This module provides data structures and solvers for simulating contaminant
transport through porous media, including equilibrium and kinetic sorption.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pfas.solver_utils import compute_dimensionless_params
from pfas.solvers import equilibrium_solver, kinetic_solver


@dataclass
class SimulationGrid:
    """Grid for spatial and temporal discretization of the simulation domain.

    Parameters
    ----------
    depth : ndarray
        Spatial grid points (z-coordinates) in the domain.
    time : ndarray
        Temporal grid points for the simulation.
    """

    depth: NDArray[np.float64]
    time: NDArray[np.float64]

    @property
    def domain_length(self) -> float:
        """Calculate total domain length.

        Returns
        -------
        float
            Total length of the simulation domain.
        """
        return self.depth[-1] + self.depth[0]


@dataclass
class BoundaryConditions:
    """Boundary conditions for contaminant transport.

    Parameters
    ----------
    C_list : list of float
        Inlet concentrations for each interval [M L⁻³]. ``C_list[j]`` is
        the concentration active from ``T_list[j]`` until ``T_list[j+1]``.
        The last interval extends to infinity. Examples:
        - Continuous step:   ``C_list=[C0],           T_list=[0]``
        - Pulse from t=0:    ``C_list=[C0, 0],        T_list=[0, t1]``
        - Delayed pulse:     ``C_list=[0, C0, 0],     T_list=[0, t1, t2]``
        - Multiple pulses:   ``C_list=[f1, 0, f2],    T_list=[0, t1, t2]``
    T_list : list of float
        Switching times [T] at which the inlet concentration changes.
        Must have the same length as ``C_list``. ``T_list[0]`` must be 0.
    """

    C_list: list[float]
    T_list: list[float]

@dataclass
class HydrologicalProperties:
    """Hydrological properties of the porous medium.

    Parameters
    ----------
    water_content : float
        Volumetric water content (theta) (dimensionless).
    pore_velocity : float
        Average pore water velocity (v) (m/s).
    dispersion_coefficient : float
        Hydrodynamic dispersion coefficient (D) (m²/s).
    """

    water_content: float
    pore_velocity: float
    dispersion_coefficient: float


@dataclass
class Adsorption:
    """Adsorption parameters for contaminant-solid interactions.

    Parameters
    ----------
    Kd : float
        Solid-phase partition coefficient (m3/kg).
    rate_const : float
        Rate constant for kinetic sorption (alphas) (1/s).
    frac_int : float
        Fraction of instantaneous sorption sites (Fs) (dimensionless).
    sp_retardation : float
        Solid-phase retardation factor (dimensionless).
    awi_retardation : float
        Air-water interface retardation factor (Raw) (dimensionless).
    """
    Kd: float  # noqa: N815
    rate_const: float
    frac_int: float
    sp_retardation: float
    awi_retardation: float

    @property
    def total_retardation(self) -> float:
        """Calculate total retardation factor.

        Returns
        -------
        float
            Sum of solid-phase and air-water interface retardation.
        """
        return 1 + self.sp_retardation + self.awi_retardation

    @property
    def beta_s(self) -> float:
        """Calculate solid-phase sorption parameter.

        Returns
        -------
        float
            Dimensionless sorption parameter for solid phase.
        """
        return (1 + self.frac_int * self.sp_retardation) / (1 + self.sp_retardation)

    @property
    def beta(self) -> float:
        """Calculate combined sorption parameter.

        Returns
        -------
        float
            Dimensionless parameter accounting for both solid-phase and AWI sorption.
        """
        return (
            (self.beta_s * (1 + self.sp_retardation) + self.awi_retardation)
            / (1 + self.sp_retardation + self.awi_retardation)
        )


def analytical_soln(  # noqa: PLR0913, PLR0917
    grid,
    bulk_density: float,
    boundary_conditions,
    initial_contaminant_concentration: NDArray[np.float64],
    hydro_properties,
    adsorption,
    kinetic: bool = False,
    volume_averaged: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64]]:
    """Solve contaminant transport using analytical solutions.

    Computes aqueous and sorbed phase concentrations for PFAS transport through
    the vadose zone using analytical solutions to the advection-dispersion equation
    (ADE) with retardation. Dimensionless parameters are computed via
    :func:`compute_dimensionless_params` and passed to the appropriate solver.

    Parameters
    ----------
    grid : SimulationGrid
        Spatial and temporal discretization grid. Must have `.depth` (m)
        and `.time` (s) arrays.
    bulk_density : float
        Bulk density of the porous medium (kg/L).
    boundary_conditions : BoundaryConditions
        Contaminant source boundary conditions. Must have `.C_list` [M L⁻³]
        and `.T_list` [T] defining the inlet concentration history.
    initial_contaminant_concentration : ndarray of shape (n_depth,)
        Initial aqueous concentration distribution in the domain [M L⁻³].
    hydro_properties : HydrologicalProperties
        Hydrological properties. Must have `.pore_velocity` [L T⁻¹],
        `.dispersion_coefficient` [L² T⁻¹], and `.water_content` [-].
    adsorption : Adsorption
        Adsorption parameters. Must have `.Kd`, `.total_retardation`,
        `.sp_retardation`, `.frac_int`, `.beta`, and `.beta_s`. When
        kinetic=True, also requires `.rate_const`.
    kinetic : bool, optional
        If True, use the kinetic sorption model (:func:`kinetic_solver`),
        which returns a separate sorbed phase C2. If False (default), use
        the equilibrium model (:func:`equilibrium_solver`), and C2 is None.
    volume_averaged : bool, optional
        If True, use volume-averaged (resident) concentrations in the kinetic
        BVP kernel. If False (default), use flux-averaged concentrations.
        Only used by the kinetic solver.

    Returns
    -------
    C1 : ndarray
        Aqueous phase concentration [M L⁻³].
    C2 : ndarray or None
        Sorbed phase concentration [M M⁻¹]. None when kinetic=False.
    C_tot : ndarray
        Total concentration [M L⁻³] bulk volume.

    Raises
    ------
    ValueError
        If pore_velocity or dispersion_coefficient is zero.
    """
    dim = compute_dimensionless_params(
        grid,
        hydro_properties,
        T_list=boundary_conditions.T_list,
        adsorption=adsorption,
        kinetic=kinetic,
    )

    C2 = None

    if kinetic:
        C1, C2, C_tot = kinetic_solver(
            adsorption.total_retardation,
            dim,
            boundary_conditions.C_list,
            initial_contaminant_concentration,
            adsorption.beta_s,
            adsorption.beta,
            volume_averaged,
            adsorption.sp_retardation,
            adsorption.frac_int,
            adsorption.Kd,
            hydro_properties.water_content,
            bulk_density,
        )
    else:
        C1, C_tot = equilibrium_solver(
            adsorption.total_retardation,
            dim,
            boundary_conditions.C_list,
            initial_contaminant_concentration,
            hydro_properties.water_content,
        )

    return C1, C2, C_tot
