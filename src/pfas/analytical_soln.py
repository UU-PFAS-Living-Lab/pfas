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

    :param depth: Spatial grid points (z-coordinates) in the domain
    :type depth: NDArray[float]
    :param time: Temporal grid points for the simulation
    :type time: NDArray[float]
    """

    depth: NDArray[float]
    time: NDArray[float]

    @property
    def domain_length(self) -> float:
        """Calculate total domain length.

        :return: Total length of the simulation domain
        :rtype: float
        """
        return self.depth[-1] + self.depth[0]


@dataclass
class BoundaryConditions:
    """Boundary conditions for contaminant source.

    :param pulse_time: Duration of the contaminant pulse at the boundary
    :type pulse_time: float
    :param contaminant_release_rate: Rate of contaminant release at the boundary (C10)
    :type contaminant_release_rate: NDArray[float]
    """

    pulse_time: float
    contaminant_release_rate: NDArray[float]


@dataclass
class HydrologicalProperties:
    """Hydrological properties of the porous medium.

    :param water_content: Volumetric water content (theta)
    :type water_content: float
    :param pore_velocity: Average pore water velocity (v)
    :type pore_velocity: float
    :param dispersion_coefficient: Hydrodynamic dispersion coefficient (D)
    :type dispersion_coefficient: float
    """

    water_content: float
    pore_velocity: float
    dispersion_coefficient: float


@dataclass
class Adsorption:
    """Adsorption parameters for contaminant-solid interactions.

    :param Kd: Distribution coefficient for solid-phase partitioning
    :type Kd: float
    :param rate_const: Rate constant for kinetic sorption (alphas)
    :type rate_const: float
    :param frac_int: Fraction of instantaneous sorption sites (Fs)
    :type frac_int: float
    :param sp_retardation: Solid-phase retardation factor
    :type sp_retardation: float
    :param awi_retardation: Air-water interface retardation factor (Raw)
    :type awi_retardation: float
    """

    Kd: float
    rate_const: float
    frac_int: float
    sp_retardation: float
    awi_retardation: float

    @property
    def total_retardation(self) -> float:
        """Calculate total retardation factor.

        :return: Sum of solid-phase and air-water interface retardation
        :rtype: float
        """
        return self.sp_retardation + self.awi_retardation

    @property
    def betas(self) -> float:
        """Calculate solid-phase sorption parameter.

        :return: Dimensionless sorption parameter for solid phase
        :rtype: float
        """
        return (1 + self.frac_int * self.sp_retardation) / (1 + self.sp_retardation)

    @property
    def beta(self) -> float:
        """Calculate combined sorption parameter.

        :return: Dimensionless parameter accounting for both solid-phase and AWI sorption
        :rtype: float
        """
        return (
            (self.betas * (1 + self.sp_retardation) + self.awi_retardation)
            / (1 + self.sp_retardation + self.awi_retardation)
        )


def analytical_soln(
    grid: SimulationGrid,
    bulk_density: float,
    boundary_conditions: BoundaryConditions,
    initial_contaminant_concentration: NDArray[float],
    hydro_properties: HydrologicalProperties,
    adsorption: Adsorption,
    kinetic: bool = False,
    volume_averaged: bool = False,
) -> tuple[NDArray[float], NDArray[float] | None, NDArray[float]]:
    """Solve contaminant transport using analytical solutions.

    The solution is computed using dimensionless variables:
    Z (dimensionless depth), T (dimensionless time), P (Péclet number),
    and ws (Damköhler number for kinetic sorption).

    :param grid: Spatial and temporal discretization grid
    :type grid: SimulationGrid
    :param bulk_density: Bulk density of the porous medium
    :type bulk_density: float
    :param boundary_conditions: Contaminant source boundary conditions
    :type boundary_conditions: BoundaryConditions
    :param initial_contaminant_concentration: Initial concentration distribution in the domain
    :type initial_contaminant_concentration: NDArray[float]
    :param hydro_properties: Hydrological properties of the medium
    :type hydro_properties: HydrologicalProperties
    :param adsorption: Adsorption parameters for the contaminant
    :type adsorption: Adsorption
    :param kinetic: If True, use kinetic sorption model; otherwise use equilibrium model, defaults to False
    :type kinetic: bool, optional
    :param volume_averaged: If True, return volume-averaged concentrations, defaults to False
    :type volume_averaged: bool, optional
    :return: Tuple containing aqueous phase concentration (C1), sorbed phase concentration (C2, None for equilibrium), and total concentration (C_tot)
    :rtype: tuple[NDArray[float], NDArray[float] | None, NDArray[float]]
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