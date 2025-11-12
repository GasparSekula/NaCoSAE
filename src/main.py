import os

from absl import app
from absl import flags

import config
import pipeline
import scoring

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
_NUM_INFERENCE_STEPS = flags.DEFINE_integer(
    "num_inf_steps", "25", "Number of inference steps in diffusion."
)
_NUM_IMAGES = flags.DEFINE_integer(
    "num_img", "5", "Number of images to generate per iteration."
)
_CONTROL_ACTIVATIONS_PATH = flags.DEFINE_string(
    "control_activations_path",
    "control_activations",
    "Path to directory with control images.",
)
_NEURON_ID = flags.DEFINE_integer("neuron_id", 0, "ID of a neuron to explain.")
_N_ITERS = flags.DEFINE_integer("n_iters", "10", "Number of iterations.")
_METRIC = flags.DEFINE_enum_class(
    "metric", "AUC", scoring.Metric, "Metric to use to score the concepts."
)
_PROMPT_PATH = flags.DEFINE_string(
    "prompt",
    "src/prompts/templates/prompt_mils.txt",
    "Path to prompt for the LLM.",
)
_SAVE_HISTORIES = flags.DEFINE_bool(
    "save_histories",
    True,
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
    load_config = config.LoadConfig(
        _LANGUAGE_MODEL_ID.value,
        _TEXT_TO_IMAGE_MODEL_ID.value,
        _EXPLAINED_MODEL_ID.value,
        {
            "max_new_tokens": 30,
            "prompt_path": _PROMPT_PATH.value,
        },
        {
            "num_inference_steps": _NUM_INFERENCE_STEPS.value,
        },
        {},
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
        "avgpool",  # TODO(piechotam) parameterize explained layer
        _NEURON_ID.value,
        _METRIC.value,
    )
    explanation_pipeline.run_pipeline(_N_ITERS.value)


if __name__ == "__main__":
    app.run(main)
