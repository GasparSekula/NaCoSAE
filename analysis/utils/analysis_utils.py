"""This module provides functions for results analysis."""

from typing import Mapping, Sequence, Tuple
import os 
import json
import ast

import pandas as pd

def get_classes(classes_file_path: str) -> Sequence[str]:
    classes_raw = open(classes_file_path).read().splitlines()
    classes = [c.replace("_", " ") for c in classes_raw]
    return classes


def get_final_concepts(results_dir: str, classes_filepath: str) -> pd.DataFrame:
    predefined_classes = get_classes(classes_filepath)
    results = []

    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            params_filepath = os.path.join(folder_path, "params.txt")
            best_concepts_filepath = os.path.join(folder_path, "best_concepts.txt")

            if not os.path.exists(params_filepath) or not os.path.exists(
                best_concepts_filepath
            ):
                continue

            params = {}

            with open(params_filepath, "r") as params_file:
                for line in params_file:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    params[key.strip()] = value.strip()

            params["load_config"] = ast.literal_eval(params["load_config"])

            best_concepts = []
            with open(best_concepts_filepath, "r") as best_concepts_file:
                for line in best_concepts_file:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    best_concepts.append(
                        (obj["best_concept"], round(obj["best_score"], 2))
                    )
            concept = best_concepts[-1]

            result = {
                "model": params["load_config"]["explained_model_id"],
                "layer": params["model_layer_activations_path"].split("/")[-1],
                "neuron": params["neuron_id"],
                "concept": concept[0],
                "concept_type": (
                    "predefined" if concept[0].strip() in predefined_classes else "generated"
                ),
            }

            results.append(result)

    return pd.DataFrame(results)
