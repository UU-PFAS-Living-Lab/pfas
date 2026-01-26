The *pfas* API Reference
=============================

This chapter documents the Python API for the *pfas* package.

The "model" module
------------------

.. module:: pfas.model

The model module provides orchestration for sequential execution of preprocessing
and solving steps using a builder pattern.

.. autoclass:: Model
   :members: __init__, add
   :member-order: bysource

The "preprocessing" module
---------------------------

.. module:: pfas.preprocessing

The preprocessing module provides classes for preparing input parameters
before running the analytical solver. Each preprocessor class follows a
common pattern: it accepts parameters on initialization, validates them
using Pydantic, and provides a ``compute()`` method that returns a
dictionary of computed values.

.. autoclass:: WaterPreprocessor
   :members: compute
   :member-order: bysource

.. autoclass:: BoundaryPreprocessor
   :members: compute
   :member-order: bysource

.. autoclass:: GridGenerator
   :members: compute
   :member-order: bysource

.. autoclass:: SpRetardationPreprocessor
   :members: compute
   :member-order: bysource

.. autoclass:: SWCAdsorptionPreprocessor
   :members: compute
   :member-order: bysource

.. autoclass:: SorptionKawiDirectInput
   :members: compute
   :member-order: bysource

.. autoclass:: SimulationRunner
   :members: compute
   :member-order: bysource

The "analytical_soln" module
----------------------------

.. module:: pfas.analytical_soln

The analytical_soln module provides the core data structures and solvers for
simulating contaminant transport through porous media with equilibrium or
kinetic sorption. It takes preprocessed input from the preprocessing module
and runs the appropriate solver based on the selected sorption model.

Data Structures
~~~~~~~~~~~~~~~

.. autoclass:: SimulationGrid
   :members:
   :member-order: bysource

.. autoclass:: BoundaryConditions
   :members:
   :member-order: bysource

.. autoclass:: HydrologicalProperties
   :members:
   :member-order: bysource

.. autoclass:: Adsorption
   :members:
   :member-order: bysource

Main Solution Function
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: analytical_soln

The "solvers" module
--------------------

.. module:: pfas.solvers

The solvers module provides numerical solver functions for the analytical
solution of the advection-dispersion equation with retardation and kinetic
sorption for PFAS transport through the vadose zone.

Helper Functions
~~~~~~~~~~~~~~~~

.. autofunction:: eqbvpfunc

.. autofunction:: eqivpfunc

.. autofunction:: neqivpfunc

.. autofunction:: Hfunc

.. autofunction:: Hs2func

Main Solver Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: equilibrium_solver

.. autofunction:: kinetic_solver

The "utils" module
------------------

.. module:: pfas.utils

The utils module provides utility functions for kinetic sorption calculations,
air-water interface area estimation, and numerical integration support.

.. autofunction:: ABfunc

.. autofunction:: Aaw_func_thermo

.. autofunction:: Aaw_func_tracer