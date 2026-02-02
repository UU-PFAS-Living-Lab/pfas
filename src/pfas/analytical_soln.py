# noqa: N806
"""Analytical solution module for PFAS transport modeling.

This module provides data structures and solvers for simulating contaminant
transport through porous media, including equilibrium and kinetic sorption.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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
    """Boundary conditions for contaminant source.

    Parameters
    ----------
    pulse_time : float
        Duration of the contaminant pulse at the boundary (s).
    contaminant_release_rate : float or ndarray
        Rate of contaminant release at the boundary (C10) (mg/L·s).
    """

    pulse_time: float
    contaminant_release_rate: NDArray[np.float64] | float


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
        Distribution coefficient for solid-phase partitioning (L/kg).
    rate_const : float
        Rate constant for kinetic sorption (alphas) (1/s).
    frac_int : float
        Fraction of instantaneous sorption sites (Fs) (dimensionless).
    sp_retardation : float
        Solid-phase retardation factor (dimensionless).
    awi_retardation : float
        Air-water interface retardation factor (Raw) (dimensionless).
    """

    Kd: float
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
        return self.sp_retardation + self.awi_retardation

    @property
    def betas(self) -> float:
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
            (self.betas * (1 + self.sp_retardation) + self.awi_retardation)
            / (1 + self.sp_retardation + self.awi_retardation)
        )


def analytical_soln( #noqa:PLR0913
    grid: SimulationGrid,
    bulk_density: float,
    boundary_conditions: BoundaryConditions,
    initial_contaminant_concentration: NDArray[np.float64],
    hydro_properties: HydrologicalProperties,
    adsorption: Adsorption,
    kinetic: bool = False,
    volume_averaged: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64]]:
    """Solve contaminant transport using analytical solutions.

    Computes aqueous and sorbed phase concentrations for PFAS transport through
    the vadose zone using analytical solutions to the advection-dispersion equation
    with retardation. The solution is computed using dimensionless variables:
    Z (dimensionless depth), T (dimensionless time), P (Péclet number),
    and ws (Damköhler number for kinetic sorption).

    Parameters
    ----------
    grid : SimulationGrid
        Spatial and temporal discretization grid.
    bulk_density : float
        Bulk density of the porous medium (kg/L).
    boundary_conditions : BoundaryConditions
        Contaminant source boundary conditions.
    initial_contaminant_concentration : ndarray
        Initial concentration distribution in the domain (mg/L).
    hydro_properties : HydrologicalProperties
        Hydrological properties of the medium.
    adsorption : Adsorption
        Adsorption parameters for the contaminant.
    kinetic : bool, optional
        If True, use kinetic sorption model; otherwise use equilibrium model.
        Default is False.
    volume_averaged : bool, optional
        If True, return volume-averaged concentrations. Default is False.

    Returns
    -------
    C1 : ndarray
        Aqueous phase concentration (mg/L).
    C2 : ndarray or None
        Sorbed phase concentration (mg/kg). None for equilibrium sorption.
    C_tot : ndarray
        Total concentration (mg/L bulk volume).
    """
    # Compute dimensionless variables
    L = grid.depth[-1]
    v = hydro_properties.pore_velocity
    Z = grid.depth / L
    T = grid.time * (v / L)
    T0 = boundary_conditions.pulse_time * (v / L)
    P = v * L / hydro_properties.dispersion_coefficient

    ws = (
        adsorption.rate_const
        * (1 - adsorption.betas)
        * (1 + adsorption.sp_retardation)
        * L
        / v
    )
    C2 = None

    if kinetic:
        C1, C2, C_tot = kinetic_solver(
            adsorption.total_retardation,
            Z,
            T,
            P,
            T0,
            boundary_conditions.contaminant_release_rate,
            initial_contaminant_concentration,
            ws,
            adsorption.betas,
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
            Z,
            T,
            P,
            T0,
            boundary_conditions.contaminant_release_rate,
            initial_contaminant_concentration,
            hydro_properties.water_content,
        )

    return C1, C2, C_tot
