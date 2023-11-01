import yaml

def get_training_args(config_path, model_type):
    with open(config_path, "r") as stream:
        config=yaml.safe_load(stream)
    return config["train"][model_type]
