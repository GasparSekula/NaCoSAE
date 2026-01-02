"""This module provides functions for results analysis."""

from typing import Mapping, Sequence, Tuple
import os
import json
import ast

import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import logging


def get_classes(classes_file_path: str) -> Sequence[str]:
    classes_raw = open(classes_file_path).read().splitlines()
    classes = [c.replace("_", " ") for c in classes_raw]
    return classes


def join_clip_outputs(clip_dissect_outputs_path: str):
    dfs = []

    for file in os.listdir(clip_dissect_outputs_path):
        file_path = os.path.join(clip_dissect_outputs_path, file)
        model_name = "_".join(m for m in file.split("_")[:-1])
        df = pd.read_csv(file_path)
        df["model"] = model_name
        dfs.append(df)

    res = pd.concat(dfs, ignore_index=True)
    res.rename(columns={"unit": "neuron"}, inplace=True)
    return res


def get_final_concepts(results_dir: str, classes_filepath: str) -> pd.DataFrame:
    predefined_classes = get_classes(classes_filepath)
    results = []

    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            params_filepath = os.path.join(folder_path, "params.txt")
            best_concepts_filepath = os.path.join(
                folder_path, "best_concepts.txt"
            )

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
                "metric": params["metric"],
                "concept_type": (
                    "predefined"
                    if concept[0].strip() in predefined_classes
                    else "generated"
                ),
            }

            results.append(result)

    return pd.DataFrame(results)


def compare_methods(
    our_path: str, clip_dissect_path: str, classes_filepath: str
) -> pd.DataFrame:
    df_our = get_final_concepts(
        results_dir=our_path, classes_filepath=classes_filepath
    )
    df_our["neuron"] = df_our["neuron"].astype("int64")
    df_clip_dissect = join_clip_outputs(
        clip_dissect_outputs_path=clip_dissect_path
    )

    df = pd.merge(
        left=df_our, right=df_clip_dissect, on=["model", "layer", "neuron"]
    )
    df = df.rename(
        columns={
            "concept": "our",
            "description": "CLIP-Dissect",
            "concept_type": "our_concept_type",
        }
    )

    return df[
        ["model", "layer", "neuron", "our", "CLIP-Dissect", "our_concept_type"]
    ]


def find_top_activating_control_images(
    concept_activations_path: str,
    neuron_id: int,
    top_k: int = 5,
) -> Sequence[str]:
    """Return identifiers (filenames without extension) of top-k images.

    Args:
        concept_activations_path: Directory with .pt activations for a concept.
        neuron_id: Neuron index to score by.
        top_k: Number of top images to return.
    """
    concept_activations = {}

    for image_activations_file in os.listdir(concept_activations_path):
        if not image_activations_file.endswith(".pt"):
            continue
        tensor = torch.load(
            os.path.join(concept_activations_path, image_activations_file),
            map_location="cpu",
        )
        concept_activations[os.path.splitext(image_activations_file)[0]] = float(
            tensor[0, neuron_id]
        )

    return [
        key
        for key, _ in sorted(
            concept_activations.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
    ]


def visualize_top_activating_images(
    neuron_id: int,
    our_images_dir: str,
    concept_activations_path: str,
    concept_images_dir: str,
    our_concept: str = "Our Concept",
    control_concept: str = "Control Cocnept",
    image_height: int = 224,
) -> None:
    
    our_neuron_dir = os.path.join(our_images_dir, str(neuron_id))
    if not os.path.isdir(our_neuron_dir):
        raise FileNotFoundError(f"Our images directory not found: {our_neuron_dir}")

    our_files = [
        os.path.join(our_neuron_dir, f)
        for f in sorted(os.listdir(our_neuron_dir))
    ]
    if len(our_files) == 0:
        raise FileNotFoundError(f"No images found in {our_neuron_dir}")

    top_ids = find_top_activating_control_images(
        concept_activations_path=concept_activations_path,
        neuron_id=neuron_id,
        top_k=len(our_files),
    )

    def _find_image_by_identifier(images_dir: str, identifier: str) -> str | None:
        for f in os.listdir(images_dir):
            if os.path.splitext(f)[0] == identifier:
                return os.path.join(images_dir, f)
        return None

    control_files = []
    for ident in top_ids:
        p = _find_image_by_identifier(concept_images_dir, ident)
        if p is None:
            logging.warning(
                "Control image not found for identifier '%s' in %s", ident, concept_images_dir
            )
            continue
        control_files.append(p)

    N = len(our_files)
    control_files = control_files[:N]
    if len(control_files) < N:
        control_files += [None] * (N - len(control_files))

    def _resize_to_height(pil_image: Image.Image, height: int) -> Image.Image:
        """Resize image to target height while maintaining aspect ratio."""
        ratio = height / pil_image.height
        new_width = int(pil_image.width * ratio)
        return pil_image.resize((new_width, height), Image.LANCZOS)

    fig, axes = plt.subplots(2, N, figsize=(min(6 + 2*N, 3*N), 6))
    if N == 1:
        axes = axes.reshape(2, 1)  

    fig.suptitle(f"Neuron {neuron_id} top activating images comparison")
    
    row1_subtitle = f"Our concept: {our_concept.upper()} - top activating synthetic images"
    row2_subtitle = f"Control concept: {control_concept.upper()} - top activating Imagenet images"

    for idx in range(N):
        ax = axes[0, idx]
        ax.axis("off")
        try:
            with Image.open(our_files[idx]) as im:
                im_resized = _resize_to_height(im.copy(), image_height)
                ax.imshow(im_resized)
        except Exception as e:
            logging.error("Failed to load our image %s: %s", our_files[idx], e)
        if idx == N//2:
            ax.set_title(row1_subtitle, fontsize=10)

        plt.subplots_adjust(hspace=1.5)  

    for idx in range(N):
        ax = axes[1, idx]
        ax.axis("off")
        p = control_files[idx]
        if p is None:
            continue
        try:
            with Image.open(p) as im:
                im_resized = _resize_to_height(im.copy(), image_height)
                ax.imshow(im_resized)
        except Exception as e:
            logging.error("Failed to load control image %s: %s", p, e)
        if idx == N//2:
            ax.set_title(row2_subtitle, fontsize=10)

    plt.tight_layout()
    plt.show()






