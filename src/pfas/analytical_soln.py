from dataclasses import dataclass

import numpy as np

from pfas_leach_screening.solvers import equilibrium_solver, kinetic_solver


@dataclass
class SimulationGrid():
    depth: np.array[float]  # z
    time: np.array[float]  # t

    @property
    def total_depth(self) -> float:
        return self.depth[-1] + self.depth[0]


@dataclass
class BoundaryConditions():
    pulse_time: float
    # solute_concentration: float
    contaminant_release_rate: np.NDArray[float] # C10


@dataclass
class HydrologicalProperties():
    water_content: float # theta
    pore_velocity: float # v
    dispersion_coefficient: float  # D


@dataclass
class Adsorption():
    Kd: float
    rate_const: float  # alphas
    frac_int: float # Fs
    sp_retardation: float # adsorption.sp_retardation
    awi_retardation: float # Raw

    @property
    def total_retardation(self) -> float:
        return self.sp_retardation + self.awi_retardation

    @property
    def betas(self) -> float:
        return (1+self.frac_int*self.sp_retardation)/(1+self.sp_retardation)

    @property
    def beta(self) -> float:
        return ((self.betas*(1+self.sp_retardation)+self.awi_retardation)
                / (1+self.sp_retardation+self.awi_retardation))

def analytical_soln(
        grid: SimulationGrid,
        bulk_density: float,
        boundary_conditions: BoundaryConditions,
        initial_contaminant_concentration: np.NDArray[float],
        hydro_properties: HydrologicalProperties,
        adsorption: Adsorption,
        kinetic: bool = False,
        volume_averaged: bool = False):

    # Compute dimensionless variables
    L = grid.depth[-1]
    v = hydro_properties.pore_velocity
    Z = grid.depth/L            # dimensionless length
    T = grid.time*(v/L)        # dimensionless time
    T0 = boundary_conditions.pulse_time*(v/L)      # dimensionless duration of contamination period
    P = v*L/hydro_properties.dispersion_coefficient        # Peclect number

    ws = adsorption.rate_const*(1-adsorption.betas)*(1+adsorption.sp_retardation)*L/v   # Damköhler number
    C2 = None  # TODO

    if kinetic:  # SPA == solid phase adsorption
        C1, C2, C_tot = kinetic_solver(
            adsorption.total_retardation, Z, T, P, T0, boundary_conditions.contaminant_release_rate,
            initial_contaminant_concentration, ws, adsorption.betas, adsorption.beta,
            volume_averaged, adsorption.sp_retardation, adsorption.frac_int,
            adsorption.Kd, hydro_properties.water_content, bulk_density)
    else:
        C1, C_tot = equilibrium_solver(
            adsorption.total_retardation,
            Z, T, P, T0, boundary_conditions.contaminant_release_rate,
            initial_contaminant_concentration, hydro_properties.water_content)

    return C1, C2, C_tot
