from pfas.configuration import validate_config


def test_config(configuration):
    # Pass the underlying dict to the validation function
    validate_config(configuration.config_dict)
