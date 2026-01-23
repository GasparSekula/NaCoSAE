"""Main entry point for the NaCoSAE pipeline.

This module orchestrates the neuron concept attribution pipeline, which explains
the behavior of individual neurons by generating and refining concept descriptions
using language and image generation models.
"""

import os
import random

from absl import app
from absl import flags

import config
import pipeline
import scoring

_SEED = 42

_TEXT_TO_IMAGE_MODEL_ID = flags.DEFINE_string(
    "t2i_model", "stabilityai/sd-turbo", "model_id of the text2image model."
)
_EXPLAINED_MODEL_ID = flags.DEFINE_string(
    "explained_model", "resnet18", "model_id of the explained model."
)
_LANGUAGE_MODEL_ID = flags.DEFINE_string(
    "language_model",
    "meta-llama/Llama-3.2-1B-Instruct",
    "model_id of the language model.",
)
_MODEL_SWAPPING = flags.DEFINE_bool(
    "model_swapping",
    False,
    "If true, the models used will be send to CPU while not performing inference"
    " and send to GPU only for inference. Useful when the models don't fit into"
    " the VRAM all at once.",
)
_NUM_INFERENCE_STEPS = flags.DEFINE_integer(
    "num_inf_steps", "25", "Number of inference steps in diffusion."
)
_GUIDANCE_SCALE = flags.DEFINE_integer(
    "guidance_scale", "5", "Guidance scale for stable diffusion."
)
_NUM_IMAGES = flags.DEFINE_integer(
    "num_img", "5", "Number of images to generate per iteration."
)
_CONTROL_ACTIVATIONS_PATH = flags.DEFINE_string(
    "control_activations_path",
    "control_activations",
    "Path to directory with control images.",
)
_LAYER = flags.DEFINE_string(
    "layer", "avgpool", "Name of the layer to explain."
)
_NEURON_ID = flags.DEFINE_integer("neuron_id", 0, "ID of a neuron to explain.")
_N_ITERS = flags.DEFINE_integer("n_iters", "10", "Number of iterations.")
_METRIC = flags.DEFINE_enum_class(
    "metric", "AUC", scoring.Metric, "Metric to use to score the concepts."
)
_PROMPT_PATH = flags.DEFINE_string(
    "prompt",
    "src/prompts/templates/prompt_main.txt",
    "Path to prompt for the LLM.",
)
_SUMMARY_PROMPT_PATH = flags.DEFINE_string(
    "summary_prompt",
    "src/prompts/templates/prompt_summary.txt",
    "Path to summary prompt for the LLM.",
)
_SAVE_HISTORIES = flags.DEFINE_bool(
    "save_histories",
    False,
    "If true, generation history and final concept history of the LLM will be"
    "saved to a file.",
)
_SAVE_IMAGES = flags.DEFINE_bool(
    "save_images",
    False,
    "If true, images from each iteration of the pipeline will be saved.",
)
_SAVE_DIR = flags.DEFINE_string(
    "save_dir",
    os.environ["SAVE_DIR"],
    "Path where pipeline artifacts will be stored.",
)


def main(argv):
    """Run the neuron concept attribution pipeline.

    Initializes all configuration objects, creates the explanation pipeline,
    and runs it for the specified number of iterations to generate and refine
    concept descriptions for a target neuron.

    Args:
        argv: Command-line arguments (provided by absl).
    """
    random.seed(_SEED)

    load_config = config.LoadConfig(
        _LANGUAGE_MODEL_ID.value,
        _TEXT_TO_IMAGE_MODEL_ID.value,
        _EXPLAINED_MODEL_ID.value,
        {
            "max_new_tokens": 3000,
            "prompt_path": _PROMPT_PATH.value,
            "summary_prompt_path": _SUMMARY_PROMPT_PATH.value,
        },
        {
            "num_inference_steps": _NUM_INFERENCE_STEPS.value,
            "guidance_scale": _GUIDANCE_SCALE.value,
        },
        {"layer": _LAYER.value},
        _MODEL_SWAPPING.value,
    )
    image_generation_config = config.ImageGenerationConfig(
        _NUM_IMAGES.value, "A realstic photo of a"
    )
    concept_history_config = config.ConceptHistoryConfig(5, 5)  # temp
    history_managing_config = config.HistoryManagingConfig(
        _SAVE_IMAGES.value, _SAVE_HISTORIES.value, _SAVE_DIR.value
    )

    explanation_pipeline = pipeline.Pipeline(
        load_config,
        image_generation_config,
        concept_history_config,
        history_managing_config,
        _CONTROL_ACTIVATIONS_PATH.value,
        _LAYER.value,
        _NEURON_ID.value,
        _METRIC.value,
    )
    explanation_pipeline.run_pipeline(_N_ITERS.value)


if __name__ == "__main__":
    app.run(main)
