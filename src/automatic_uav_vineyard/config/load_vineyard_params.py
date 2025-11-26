#!/usr/bin/env python3
import yaml
import os

def load_vineyard_params():
    """
    Load vineyard parameters from vineyard_params.yaml.
    Returns a dictionary containing all parameters under 'ros__parameters'.
    """
    # Percorso del file corrente
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Percorso del file YAML
    yaml_path = os.path.join(current_dir, "vineyard_params.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config not found at: {yaml_path}")

    # Carica YAML
    with open(yaml_path, "r") as f:
        params = yaml.safe_load(f)

    # Estrai i parametri corretti sotto ros__parameters
    if "vineyard_offboard" not in params or "ros__parameters" not in params["vineyard_offboard"]:
        raise KeyError("Chiave 'vineyard_offboard: ros__parameters' non trovata nel file YAML")

    data = params["vineyard_offboard"]["ros__parameters"]

    # Stampa per verifica
    print("=== PARAMETRI LETTI DAL YAML ===")
    for key, value in data.items():
        print(f"{key}: {value}")

    return data


if __name__ == "__main__":
    load_vineyard_params()
