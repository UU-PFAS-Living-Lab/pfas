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
