# src/biomass2pred/config.py

from pathlib import Path
import yaml

def load_config(config_path: str | Path):
    """
    Load YAML configuration file.

    Args:
        config_path : Path to the YAML config file.

    Returns:
        Nested dictionary with configuration values
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config