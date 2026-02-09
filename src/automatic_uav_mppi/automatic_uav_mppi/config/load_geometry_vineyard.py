#!/usr/bin/env python3
"""
load_geometry_vineyard.py
Loads YAML configuration and converts it to a Python dict with JAX arrays.
"""

import pathlib

import jax.numpy as jnp
import numpy as np
import yaml
import os

from ament_index_python.packages import get_package_share_directory

def load_mppi_config(config_path=None):
    """
    Load MPPI configuration from YAML file and convert to Python dict.

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        dict: Configuration dictionary with JAX arrays
    """
    if config_path is None:
        pkg_share = get_package_share_directory("automatic_uav_mppi")
        config_path = pathlib.Path(pkg_share) / "config" / "geometry_vineyard.yaml"

    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Convert to Python config with JAX arrays
    config = {
        "home": jnp.array(yaml_config["home"], dtype=jnp.float32),
        "first_row": jnp.array(yaml_config["first_row"], dtype=jnp.float32),
        "altitude": yaml_config["altitude"],
        "row_length": yaml_config["row_length"],
        "row_spacing": yaml_config["row_spacing"],
        "num_rows": yaml_config["num_rows"],
        
        "v_ref": yaml_config["v_ref"],
        "radius_turn": yaml_config["radius_turn"],

        "publish_rate": yaml_config["publish_rate"],
        "horizon": yaml_config["horizon"],
        "dt": yaml_config["dt"],

        "takeoff_threshold": yaml_config["takeoff_threshold"],

        "hold_time": yaml_config["hold_time"],
    }

    return config


def save_config_as_python(yaml_path, output_path):
    """
    Convert YAML config to a Python file for easy editing.

    Args:
        yaml_path: Path to input YAML file
        output_path: Path to output Python file
    """
    config = load_mppi_config(yaml_path)

    with open(output_path, "w") as f:
        f.write("# Auto-generated from YAML config\n")
        f.write("import jax.numpy as jnp\n\n")
        f.write("config = {\n")

        for key, value in config.items():
            if isinstance(value, (jnp.ndarray, np.ndarray)):
                f.write(f'    "{key}": jnp.array({value.tolist()}),\n')
            else:
                f.write(f'    "{key}": {value},\n')

        f.write("}\n")

    print(f"Python config saved to: {output_path}")


if __name__ == "__main__":
    # Test loading config
    config = load_mppi_config()
    print("Configuration loaded successfully!")
    print(f"home: {config['home']}")
    print(f"first_row: {config['first_row']}")
    print(f"altitude: {config['altitude']}")
    print(f"row_length: {config['row_length']}")
    print(f"row_spacing: {config['row_spacing']}")
    print(f"num_rows: {config['num_rows']}")
    print(f"v_ref: {config['v_ref']}")
    print(f"radius_turn: {config['radius_turn']}")
    print(f"publish_rate: {config['publish_rate']}")
    print(f"horizon: {config['horizon']}")
    print(f"dt: {config['dt']}")
    print(f"takeoff_threshold: {config['takeoff_threshold']}")
    print(f"hold_time: {config['hold_time']}")