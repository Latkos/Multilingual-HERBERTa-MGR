import yaml


def get_config(config_path):
    with open(config_path, "r") as stream:
        return yaml.safe_load(stream)


def get_training_args(config_path,model_type):
    config = get_config(config_path)
    return config["train"][model_type]
