[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
# PFAS - *a package for semi-analytical modeling of PFAS transport in the vadose zone*
# PFAS Transport Modeling Package

A Python package for modeling the transport of per- and polyfluoroalkyl substances (PFAS) through the unsaturated zone.

## Overview

PFAS is a toolkit for simulating the movement and fate of PFAS contaminants in soil and groundwater systems. It provides a flexible, modular framework for constructing transport models with configurable preprocessing steps and analytical solvers. The package is designed for researchers and engineers studying PFAS contamination and remediation.

## Features

- **Modular Architecture**: Build complex transport models using pluggable preprocessors and solvers
- **Flexible Configuration**: Define simulations using intuitive TOML configuration files
- **Sorption Modeling**: Support for linear and non-linear sorption processes (Kawi model)
- **Vadose Zone Transport**: Simulate PFAS movement through the unsaturated zone
- **Soil-Water Content Effects**: Account for soil-water characteristic curves in moisture dynamics
- **Grid Generation**: Automatic mesh generation for spatial domains
- **Boundary Condition Management**: Flexible handling of domain boundaries
- **Type-Safe**: Full type annotations for better IDE support and code reliability

## Requirements

- Python >= 3.9
- NumPy >= 2
- SciPy
- Matplotlib
- Pydantic
- Marimo (for interactive documentation)

## Installation

### From PyPI (coming soon)

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
from pfas.configuration import read_toml
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

# Load configuration from file
config = read_toml("examples/data/config.toml")

# Create and configure the model
model = (Model(config)
    .add(WaterPreprocessor, porosity=0.4)
    .add(BoundaryPreprocessor)
    .add(GridGenerator)
    .add(SpRetardationPreprocessor)
    .add(SWCAdsorptionPreprocessor)
    .add(SorptionKawiDirectInput)
    .add(SimulationRunner)
)

# Access results
data = model.generated_data
C_tot = data["C_tot"]  # Total PFAS concentration
grid = data["grid"]    # Grid information
```

## Documentation

Full documentation is available at [Read the Docs](https://pfas.readthedocs.io/) and includes:

- [Installation Guide](docs/source/installation.rst)
- [Quick Start Tutorial](docs/source/quick_start.rst)
- [Detailed Tutorials](docs/source/tutorials.rst)
- [API Reference](docs/source/api.rst)
- [FAQ](docs/source/faq.rst)

## Examples

Several example scripts are provided in the `examples/` directory, demonstrating:

- `data_structure.py` - Data structure handling
- `initial_value_problem.py` - Setting up initial value problems
- `Kd_sorption.py` - Linear sorption (Kd) modeling
- `factory.py` - Factory patterns for model creation
- `gen_example.py` - Configuration generation



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PFAS in your research, please cite:


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please contact:

## Acknowledgments

This package was developed at Utrecht University as part of the PFAS Living Lab initiative.

