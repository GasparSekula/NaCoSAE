from src.model import image_model
from src.model import explained_model
from src import image_processing
from src import scoring

from absl import app
from absl import flags
import os
from PIL import Image

# TODO(piechotam) parameterize explained layer

_TEXT_TO_IMAGE_MODEL_ID = flags.DEFINE_string(
    "t2i_model", "stabilityai/sd-turbo", "model_id of the text2image model."
)
_EXPLAINED_MODEL_ID = flags.DEFINE_string(
    "explained_model", "resnet18", "model_id of the explained model."
)
_DEVICE = flags.DEFINE_string("device", "cuda", "Device to use")
_NUM_INFERENCE_STEPS = flags.DEFINE_integer(
    "num_inf_steps", "25", "Number of inference steps in diffusion."
)
_NUM_IMAGES = flags.DEFINE_integer(
    "num_img", "5", "Number of images to generate per iteration."
)
_CONTROL_IMAGES_PATH = flags.DEFINE_string(
    "control_images_path",
    "control_images",
    "Path to directory with control images.",
)
_NEURON_ID = flags.DEFINE_integer("neuron_id", 0, "ID of a neuron to explain.")


def main(argv):
    t2i_model = image_model.ImageModel(
        _TEXT_TO_IMAGE_MODEL_ID.value, _DEVICE.value, _NUM_INFERENCE_STEPS.value
    )
    expl_model = explained_model.ExplainedModel(
        _EXPLAINED_MODEL_ID.value, _DEVICE.value
    )

    synthetic_images = t2i_model.generate_images(
        _NUM_IMAGES.value, "a realistic photo of", "fruits"
    )
    image_processing.save_images_from_iteration(
        "synthetic_images", synthetic_images, "test_run_id", 1
    )
    input_batch_synthetic = image_processing.transform_images(
        _EXPLAINED_MODEL_ID.value, synthetic_images
    )
    synthetic_activations = expl_model.get_activations(input_batch_synthetic)

    # this will be precalculated and input_batch will be loaded from the start
    # for now its a rather inefficient, prototype version
    control_images = []
    control_images_directory = os.fsencode(_CONTROL_IMAGES_PATH.value)

    for control_image in os.listdir(control_images_directory):
        with Image.open(
            os.path.join(control_images_directory, control_image)
        ) as pil_image:
            control_images.append(pil_image.copy())

    input_batch_control = image_processing.transform_images(
        _EXPLAINED_MODEL_ID.value, control_images
    )
    control_activations = expl_model.get_activations(input_batch_control)

    metrics = scoring.calculate_metrics(
        control_activations[:, _NEURON_ID.value],
        synthetic_activations[:, _NEURON_ID.value],
    )

    auc = metrics["auc"]
    print(f"AUC: {auc}")


if __name__ == "__main__":
    app.run(main)
