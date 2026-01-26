The *pfas* API Reference
=============================

This chapter documents the Python API for the *pfas* package.

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