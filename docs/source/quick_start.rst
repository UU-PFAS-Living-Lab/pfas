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

This example demonstrates a basic forward modeling exercise of PFAS leaching in the vadose zone using linear sorption.

**Step 1: Load configuration**

.. code-block:: python

    from pfas.configuration import read_toml
    
    config = read_toml("examples/data/config.toml")

**Step 2: Initialize the model and add preprocessors**

.. code-block:: python

    from pfas.preprocessing import (
        WaterPreprocessor,
        BoundaryPreprocessor,
        GridGenerator,
        SpRetardationPreprocessor,
        SWCAdsorptionPreprocessor,
        SorptionKawiDirectInput,
        SimulationRunner
    )
    from pfas.model import Model
    
    # Create model instance
    model = Model(config)
    
    # Add preprocessing steps
    model.add(WaterPreprocessor, porosity=0.4)
    model.add(BoundaryPreprocessor)
    model.add(GridGenerator)
    model.add(SpRetardationPreprocessor)
    model.add(SWCAdsorptionPreprocessor)
    model.add(SorptionKawiDirectInput)
    model.add(SimulationRunner)

**Step 3: Access generated data**

.. code-block:: python

    # Access all model output
    data = model.generated_data
    C_tot = data["C_tot"]  # Total PFAS concentration
    grid = data["grid"]    # Grid information

Visualize Results
-----------------

**Breakthrough curve at the bottom of the domain:**

.. code-block:: python

    import matplotlib.pyplot as plt
    
    plt.plot(grid.time, C_tot[0, :], label=f"Depth = {grid.depth} cm", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Total PFAS Concentration (mg/L)")
    plt.title("PFAS Concentration Over Time")
    plt.legend()
    plt.show()

**Concentration depth profiles at different times:**

.. code-block:: python

    import matplotlib.pyplot as plt
    
    t_len = C_tot.shape[1]
    time_indices = [0, t_len//4, t_len//2, 3*t_len//4, -1]
    
    plt.figure(figsize=(8, 6))
    for t_idx in time_indices:
        plt.plot(C_tot[:, t_idx], grid.depth, label=f"t = {grid.time[t_idx]:.0f} s")
    
    plt.xlabel("Total PFAS Concentration (mg/L)")
    plt.ylabel("Depth (cm)")
    plt.title("PFAS Concentration Depth Profile at Different Times")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

Configuration File
------------------

Create a ``config.toml`` file with your model parameters:

.. code-block:: toml

    # See examples/data/config.toml for a complete example

Next Steps
----------

- Check the :doc:`user_guide` for detailed documentation
- Explore ``examples/`` directory for more complex scenarios
- See :doc:`api_reference` for all available functions and classes

For issues or questions, please visit the GitHub repository `<https://github.com/UU-PFAS-Living-Lab/pfas>`_