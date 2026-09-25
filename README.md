[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Documentation Status](https://readthedocs.org/projects/pfas/badge/?version=latest)](https://pfas.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/pfas)](https://pypi.org/project/pfas/)
[![Python versions](https://img.shields.io/pypi/pyversions/pfas)](https://pypi.org/project/pfas/)
# PFAS - *a package for semi-analytical modeling of PFAS transport in the vadose zone*

![PFAS Logo](docs/source/images/logo_pfas.png)

A Python package for modeling the transport of per- and polyfluoroalkyl substances (PFAS) through the unsaturated zone.

## Overview

PFAS is a toolkit for simulating the movement and fate of PFAS contaminants in the unsaturated zone. It provides a flexible, modular framework for constructing transport models with configurable preprocessing steps and analytical solvers. The package is designed for researchers and engineers studying PFAS contamination and remediation.

## Features

- **Modular Architecture**: Build complex transport models using pluggable preprocessors and solvers.
- **Flexible Configuration**: Define simulations using intuitive TOML configuration files or directly in code.
- **Sorption Modeling**: Support for linear and non-linear sorption processes to soil particles and Air-Water Interface, in combination with flexible approaches for defining these processes.
- **Vadose Zone Transport**: Simulate PFAS movement through the unsaturated zone under steady-state flow conditions.
- **Embedded PFAS data**: The package contains modules with sorption data for Air-Water interface computation, Air-Water interfacial sorption coefficients and solid phase sorption coefficients.
- **Grid Generation**: Automatic mesh generation for spatial domains
- **Boundary Condition Management**: Flexible handling of domain boundaries

## Requirements

- Python >= 3.9
- NumPy >= 2
- SciPy
- Matplotlib
- Pydantic
- Marimo (for tutorials)

## Installation

### From PyPI

```bash
pip install pfas
```

### From Source

```bash
git clone https://github.com/UU-PFAS-Living-Lab/pfas.git
cd pfas
pip install -e .
```

### For Development

Install with additional testing, documentation, and example dependencies:

```bash
pip install -e ".[dev]"
```

Or install individual extras:

```bash
pip install -e ".[test]"      # For testing
pip install -e ".[docs]"      # For building documentation
pip install -e ".[examples]"  # For running examples
```

## Quick Start

Here's a minimal example to get started:

```python
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

model = Model()

model.compute(
    GridGenerator,
    domain_length=60,
    spatial_resolution=1.0,
    time_resolution=100,
    time_total=10000,
)
model.compute(
    WaterPreprocessor,
    average_infiltration_rate=1.5,
    hydraulic_conductivity=6,
    porosity=0.34,
    dispersivity=1.5,
    van_genuchten_n=1.31,
    residual_water_content=0.04,
)
model.compute(BoundaryPreprocessor, C_list=[10.0, 0], T_list=[0, 2000])
model.compute(
    LinearSPsorption,
    sorption_solid={
        "kinetic_sorption": True,
        "sorption_isotherm": "linear",
        "linear": {"Kd_method": "direct_input", "Kd": 5.0},
    },
)
model.compute(
    SWCsorption,
    sigma0=71,
    scaling_factor_awi=1.0,
    van_genuchten_alpha=0.019,
)
model.compute(Retardation, Kaw=0.5, bulk_density=1.6)
model.compute(EquilibriumSolver)

# Access results
grid = model.grid
concentration = model.C_tot
```

## Documentation

Full documentation is available at [Read the Docs](https://pfas.readthedocs.io/). 

## Examples

Several example scripts are provided in the `examples/` directory, demonstrating:

- `data_structure.py` - Data structure handling
- `initial_value_problem.py` - Setting up initial value problems
- `Kd_sorption_component.py` - Linear and component-based sorption modeling
- `gen_example.py` - Basic PFAS transport simulation
- `kin_vs_eq.py` - Kinetic versus equilibrium sorption
- `mass_balance.py` - Mass-balance checking
- `RunningModelDifferentSoils.py` - Running models for different soils
- `loop_for_Staring_soils_and_PFASs.py` - Looping over soils and PFAS compounds

To run these examples, you need Marimo.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PFAS in your research, please cite:


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please contact: Valerie de Rijk (v.derijk@uu.nl)

## Acknowledgments

This package was developed at Utrecht University as part of the PFAS Living Lab.

