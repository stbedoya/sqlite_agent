import yaml
import os


def load_config(config_path="config/config.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
