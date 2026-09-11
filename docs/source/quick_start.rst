Quickstart
==========

Welcome to PFAS! This package provides tools for modeling the transport of per- and polyfluoroalkyl substances (PFAS) through the unsaturated zone.

Installation
------------

Install the package using pip:

.. code-block:: bash

    pip install pfas

Or, if you're developing locally:

.. code-block:: bash

    git clone https://github.com/UU-PFAS-Living-Lab/pfas.git
    cd pfas
    pip install -e .

Basic Usage
-----------

This example follows ``examples/gen_example.py``. It models a 60 cm domain
with a 10 mg/L PFAS pulse for 2,000 seconds and runs the simulation until
10,000 seconds. The example uses linear solid-phase sorption, soil-water
characteristic based air-water interfacial adsorption, and an equilibrium
solver.

**Step 1: Import the model and components**

.. code-block:: python

    from matplotlib import pyplot as plt

    from pfas.component import (
        BoundaryPreprocessor,
        EquilibriumSolver,
        GridGenerator,
        LinearSPsorption,
        Retardation,
        SWCsorption,
        WaterPreprocessor,
    )
    from pfas.model import Model

**Step 2: Build and run the model**

.. code-block:: python

    model = Model()

    # Generate the spatial and temporal grid.
    model.compute(
        GridGenerator,
        domain_length=60,
        spatial_resolution=1.0,
        time_resolution=100,
        time_total=10000,
    )

    # Compute water flow properties.
    model.compute(
        WaterPreprocessor,
        average_infiltration_rate=1.5,
        hydraulic_conductivity=6,
        porosity=0.34,
        dispersivity=1.5,
        van_genuchten_n=1.31,
        residual_water_content=0.04,
    )

    # Set the 10 mg/L pulse at the upper boundary.
    model.compute(
        BoundaryPreprocessor,
        C_list=[10.0, 0],
        T_list=[0, 2000],
    )

    # Configure linear solid-phase sorption.
    sorption_solid = {
        "kinetic_sorption": True,
        "sorption_isotherm": "linear",
        "linear": {
            "Kd_method": "direct_input",
            "Kd": 5.0,
        },
    }
    model.compute(LinearSPsorption, sorption_solid=sorption_solid)

    # Compute air-water interfacial adsorption and retardation.
    model.compute(
        SWCsorption,
        sigma0=71,
        scaling_factor_awi=1.0,
        van_genuchten_alpha=0.019,
    )
    model.compute(Retardation, Kaw=0.5, bulk_density=1.6)

    # Run the equilibrium transport simulation.
    model.compute(EquilibriumSolver)

    simulation_grid = model.grid
    concentration = model.C1

Visualize Results
-----------------

**Concentration depth profiles at different times:**

.. code-block:: python

    time_indices = [0, 10, 20, 22, 30]

    plt.figure(figsize=(8, 6))
    for time_index in time_indices:
        plt.plot(
            concentration[:, time_index],
            simulation_grid.depth,
            label=f"t = {simulation_grid.time[time_index]:.0f} s",
        )

    plt.xlabel("Total PFAS Concentration (mg/L)")
    plt.ylabel("Depth (cm)")
    plt.title("PFAS Concentration Depth Profile at Different Times")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

**Breakthrough curve at the bottom of the domain:**

.. code-block:: python

    bottom_concentration = concentration[-1, :]

    plt.figure(figsize=(8, 5))
    plt.plot(simulation_grid.time, bottom_concentration, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("PFAS Concentration at Bottom (mg/L)")
    plt.title("PFAS Breakthrough Curve at Bottom of Model")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

Next Steps
----------

- Explore the ``examples/`` directory for more complex scenarios.
- See the :doc:`api` reference for all available modules, functions, and classes.

For issues or questions, please visit the GitHub repository `<https://github.com/UU-PFAS-Living-Lab/pfas>`_