"""Model orchestration module for PFAS transport simulations.

This module provides the Model class for orchestrating the sequential execution
of preprocessing and solving steps with a fluent builder pattern.
"""


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
    def __init__(self, config):
        """Initialize Model with configuration.
        
        Parameters
        ----------
        config : object
            Configuration object containing simulation parameters.
        """
        self.config = config
        self.generated_data = {}

    def add(self, model_class, **kwargs):
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
        config_data = {key: getattr(self.config, key) for key in model_class.__annotations__
                       if hasattr(self.config, key)}
        gen_data = {key: self.generated_data[key] for key in model_class.__annotations__
                    if key in self.generated_data}
        config_data.update(gen_data)
        config_data.update(kwargs)

        model = model_class(**config_data)
        self.generated_data.update(model.compute())
        return self
