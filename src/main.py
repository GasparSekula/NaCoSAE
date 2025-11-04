from absl import app
from absl import flags

import pipeline


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


def main(argv):
    load_config = pipeline.LoadConfig(
        _LANGUAGE_MODEL_ID.value,
        _TEXT_TO_IMAGE_MODEL_ID.value,
        _EXPLAINED_MODEL_ID.value,
        {
            "max_new_tokens": 30,
        },
        {
            "num_inference_steps": _NUM_INFERENCE_STEPS.value,
        },
        {},
    )
    image_generation_config = pipeline.ImageGenerationConfig(
        _NUM_IMAGES.value, "A realstic photo of a"
    )
    concept_history_config = pipeline.ConceptHistoryConfig(5, 5)  # temp

    explanation_pipeline = pipeline.Pipeline(
        load_config,
        image_generation_config,
        concept_history_config,
        _CONTROL_ACTIVATIONS_PATH.value,
        "avgpool",  # TODO(piechotam) parameterize explained layer
        _NEURON_ID.value,
    )
    explanation_pipeline.run_pipeline("auc", _N_ITERS.value)


if __name__ == "__main__":
    app.run(main)
