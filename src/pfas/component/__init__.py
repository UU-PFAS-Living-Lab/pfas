# -*- coding: utf-8 -*-
"""
Components for building PFAS transport model. 

"""

from pfas.component.awi import GuoTracer, SWCsorption
from pfas.component.kd import LinearSPsorption, FreundlichSPsorption
from pfas.component.kaw import Le2021_asymptote, Le2021_langmuir, Szyszkowski   
from pfas.component.retardation import Retardation
from pfas.component.solver import EquilibriumSolver, KineticSolver
__all__ = [
    "GuoTracer",
    "SWCsorption",
    "LinearSPsorption",
    "FreundlichSPsorption",
    "Retardation",
    "EquilibriumSolver",
    "KineticSolver",
    "Le2021_asymptote",
    "Le2021_langmuir",
    "Szyszkowski",
]

ALL_COMPONENTS = [GuoTracer, SWCsorption, LinearSPsorption, FreundlichSPsorption, Retardation,
                  EquilibriumSolver, KineticSolver, Le2021_asymptote, Le2021_langmuir, Szyszkowski]
