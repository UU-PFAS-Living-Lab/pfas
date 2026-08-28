"""Model orchestration module for PFAS transport simulations.

This module provides the Model class for orchestrating the sequential execution
of preprocessing and solving steps with a fluent builder pattern.
"""

from collections import defaultdict

from pydantic_core import PydanticUndefined

from pfas.component import ALL_COMPONENTS


class Model:
    """Orchestrate sequential execution of preprocessors and solvers.

    The Model class implements a builder pattern for constructing and executing
    a sequence of preprocessing and solving components. It manages the flow of
    data from configuration through various transformers and ultimately to the
    analytical solver.

    Parameters
    ----------
    config : object
        Configuration object containing parameters for the simulation. Parameters
        are extracted from this object based on the annotated fields of each
        model class.

    Attributes
    ----------
    config : object
        The configuration object passed at initialization.
    generated_data : dict
        Dictionary storing outputs from each computation step, making them
        available as inputs for subsequent steps.

    Examples
    --------
    >>> from pfas.preprocessing import WaterPreprocessor, BoundaryPreprocessor
    >>> from pfas.configuration import SimulationConfig
    >>> config = SimulationConfig(...)
    >>> result = (Model(config)
    ...     .add(WaterPreprocessor)
    ...     .add(BoundaryPreprocessor)
    ...     .add(GridGenerator)
    ...     .add(SimulationRunner)
    ... )
    """

    def __init__(self):
        """Initialize Model with configuration.

        Parameters
        ----------
        config : object
            Configuration object containing simulation parameters.
        """
        self.generated_data = {}
        self.input_data = {}
        self.default_values = {}

    def compute(self, model_class, **kwargs):
        """Add and execute a preprocessing or solving component.

        Dynamically instantiates a model class with parameters extracted from
        the configuration object and previously generated data. Executes the
        model's compute() method and stores results for downstream use.

        Parameters
        ----------
        model_class : type
            A preprocessor or solver class with type-annotated __init__ parameters.
            The class must have a compute() method.
        **kwargs : dict, optional
            Additional keyword arguments to override config or generated data values.

        Returns
        -------
        self : Model
            Returns self for method chaining (builder pattern).
        """
        extra_keys = set(kwargs) - set(model_class.model_fields.keys())
        if len(extra_keys) > 0:
            raise ValueError(f"Unknown keyword arguments supplied: {extra_keys}")
        fields = model_class.model_fields
        default_vals = {key: fields[key].default for key in fields
                        if fields[key].default != PydanticUndefined}
        self.default_values.update(default_vals)
        kwargs_avail = set(kwargs).intersection(set(self.input_data) | set(self.generated_data))
        if len(kwargs_avail) != 0:
            raise ValueError(f"You cannot supply the same key twice: {kwargs_avail}"
                             "is/are already available")

        all_data = self.default_values | kwargs | self.input_data | self.generated_data
        try:
            class_kwargs = {key: all_data[key] for key in model_class.model_fields}
        except KeyError:
            missing_keys = {key for key in model_class.model_fields if key not in all_data}
            # all_keys = set(all_data)
            candidates = []
            # candidates = list(self._find_add_components(missing_keys, all_keys))
            if len(candidates) > 0:
                raise ValueError("Multiple possible ways to compute the missing arguments,"
                                 f" add them manually: {candidates}")
            elif len(candidates) == 0:
                messages = defaultdict(list)
                for missing in missing_keys:
                    for component in ALL_COMPONENTS:
                        if missing in component.outputs:
                            messages[missing].append(component.__class__.__name__)
                missing_keys_message = f"Missing following arguments: {missing_keys}"
                suggest_message = ", ".join(f"Please supply argument {arg} directly or through "
                                            f"component(s) {', '.join(messages[arg])}"
                                            for arg in messages)
                raise ValueError(missing_keys_message + suggest_message)
            else:
                for comp in candidates:
                    self.compute(self, comp)
                class_kwargs = {key: all_data[key] for key in model_class.model_fields}

        model = model_class(**class_kwargs)
        self.generated_data.update(model.compute())
        self.input_data.update(kwargs)
        return self

    def __getattr__(self, key):
        all_data = self.default_values | self.generated_data | self.input_data
        if key in all_data:
            return all_data[key]
        return super().__getattribute__(key)

    # def _find_add_components(self, missing_arguments, all_keys):
    #     if len(missing_arguments) == 0:
    #         yield []
    #         return

    #     for comp in ALL_COMPONENTS:
    #         if len(set(missing_arguments).union(comp.outputs)) > 0:
    #             new_keys = all_keys | set(comp.outputs)
    #             new_missing_arguments = set(missing_arguments) + comp.inputs - new_keys
    #             for comp_list in self._find_add_components(new_missing_arguments, new_keys):
    #                 yield comp_list + [comp]
