import yaml


def read_yaml(file_name: str):
    with open(file_name, "r") as yml:
        config = yaml.safe_load(yml)

    return config
