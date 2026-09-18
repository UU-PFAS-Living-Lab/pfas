Introduction
============

``pfas`` is an open-source Python package for simulating the transport of
per- and polyfluoroalkyl substances (PFAS) in the unsaturated zone.

The package provides semi-analytical solutions for one-dimensional PFAS
transport under a range of retention processes, including:

* linear and nonlinear solid-phase sorption;
* equilibrium and kinetic sorption;
* air--water interfacial adsorption.

The transport framework builds on the analytical advection--dispersion
solutions implemented in CXTFIT v. 2.0 by Toride et al. (1995) [1]_ with extensions for
PFAS-specific retention processes, including air--water interfacial
adsorption, following Guo et al. (2022) [2]_. We acknowledge the original authors of
both papers for their contributions.

Project information
--------------------

``pfas`` is developed and maintained by the PFAS Living Lab at Utrecht
University. It is released as open-source software under the
`MIT License <https://opensource.org/license/mit>`_.

The project aims to make state-of-the-art representations of PFAS transport
in the vadose zone accessible, transparent, and reproducible. Its modular
design allows users to apply the included solutions and extend the package
with additional process representations or solvers.

Scientific references
----------------------

References are provided in the documentation and, where relevant, in the
API documentation for individual models and parameterisations. Users should
cite the original methodological publications when applying a specific
transport or sorption formulation.

.. [1] Toride, N., Leij, F.J., & van Genuchten, M.Th. (1995). The
   CXTFIT Code for Estimating Transport Parameters from Laboratory or Field
   Tracer Experiments, Version 2.1. Research Report No. 137, USDA-ARS,
   Riverside, CA.

.. [2] Guo, B., Zeng, J., Brusseau, M.L., & Zhang, Y. (2022). A
   screening model for quantifying PFAS leaching in the vadose zone and
   mass discharge to groundwater. *Advances in Water Resources*, 160,
   104102. https://doi.org/10.1016/j.advwatres.2021.104102