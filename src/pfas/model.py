class Model():
    def __init__(self, config):
        self.config = config
        self.generated_data = {}

    def add(self, model_class, **kwargs):
        config_data = {key: getattr(self.config, key) for key in model_class.__annotations__
                       if hasattr(self.config, key)}
        gen_data = {key: self.generated_data[key] for key in model_class.__annotations__
                    if key in self.generated_data}
        config_data.update(gen_data)
        config_data.update(kwargs)

        model = model_class(**config_data)
        self.generated_data.update(model.compute())
        return self
