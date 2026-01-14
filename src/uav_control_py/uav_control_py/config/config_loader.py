#!/usr/bin/env python3
"""
config_loader.py
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
        pkg_share = get_package_share_directory("uav_control_py")
        config_path = pathlib.Path(pkg_share) / "config" / "mppi_config.yaml"

    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Convert to Python config with JAX arrays
    config = {
        # Simulation parameters
        "dt": yaml_config["dt"],
        "n_samples": yaml_config["n_samples"],
        "horizon": yaml_config["horizon"],
        "temperature": yaml_config["temperature"],
        # Physics parameters
        "mass": yaml_config["mass"],
        "g": yaml_config["g"],
        "tau": yaml_config["tau"],
        "inertia": jnp.array(yaml_config["inertia"], dtype=jnp.float32),
        # Control constraints
        "min_thrust": yaml_config["min_thrust"],
        "max_thrust": yaml_config["max_thrust"],
        "angular_rate_min": yaml_config["angular_rate_min"],
        "angular_rate_max": yaml_config["angular_rate_max"],
        "ctrl_noise_scale": jnp.array(
            yaml_config["ctrl_noise_scale"], dtype=jnp.float32
        ),
        # Cost weights
        "Q_diag": yaml_config["Q_diag"],
        "R_diag": yaml_config["R_diag"],
        "R_rate_diag": yaml_config["R_rate_diag"],
        "Q": jnp.diag(jnp.array(yaml_config["Q_diag"], dtype=jnp.float32)),
        "R": jnp.diag(jnp.array(yaml_config["R_diag"], dtype=jnp.float32)),
        "R_rate": jnp.diag(jnp.array(yaml_config["R_rate_diag"], dtype=jnp.float32)),
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
    print(f"dt: {config['dt']}")
    print(f"n_samples: {config['n_samples']}")
    print(f"horizon: {config['horizon']}")
    print(f"Q shape: {config['Q'].shape}")
    print(f"R shape: {config['R'].shape}")