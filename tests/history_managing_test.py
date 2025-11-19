from typing import Iterable, Mapping
import io

from PIL import Image
import pytest
import pytest_mock
from unittest.mock import call, mock_open

import history_managing


def _get_output(content: Iterable[str]) -> str:
    test_buffer = io.StringIO()
    history_managing._write_iterable_to_stream(test_buffer, content)

    return test_buffer.getvalue()


def test_writing_generation_history():
    generation_history = ["concept_1,0.31", "concept_2,0.13", "concept_3,0.78"]
    output = _get_output(generation_history)
    expected = "concept_1,0.31\nconcept_2,0.13\nconcept_3,0.78"

    assert output == expected


def test_writing_concept_history():
    concept_history = {"concept_1": 0.31, "concept_2": 0.13, "concept_3": 0.78}
    output = _get_output(
        [f"{concept},{score}" for concept, score in concept_history.items()]
    )
    expected = "concept_1,0.31\nconcept_2,0.13\nconcept_3,0.78"

    assert output == expected


def test_writing_params():
    params = {"run_id": "test-run-id", "neuron_id": 0, "metric": "AUC"}
    output = _get_output(
        (f"{param}: {value}") for param, value in params.items()
    )
    expected = "run_id: test-run-id\nneuron_id: 0\nmetric: AUC"

    assert output == expected


@pytest.fixture
def test_images() -> Mapping[str, Image.Image]:
    img1 = Image.new("RGB", (10, 10), color="red")
    img2 = Image.new("RGB", (20, 20), color="blue")
    images_list = [img1, img2]
    return {"img1": img1, "img2": img2, "images_list": images_list}


def test_write_image_to_stream_writes_jpeg(
    test_images: Mapping[str, Image.Image],
):
    img1 = test_images["img1"]
    buffer = io.BytesIO()
    history_managing._write_image_to_stream(buffer, img1)

    buffer.seek(0)
    data = buffer.read()

    assert len(data) > 0

    jpeg_prefix = b"\xff\xd8"
    assert data.startswith(jpeg_prefix)


def test_save_images_from_iteration(
    test_images: Mapping[str, Image.Image], mocker: pytest_mock.MockerFixture
):
    save_dir = "test/directory"
    iter_number = 1
    concept = "concept 1"

    mock_makedirs = mocker.patch("os.makedirs")
    mock_open_file = mocker.patch("builtins.open", new_callable=mock_open)
    mock_write_image_helper = mocker.patch(
        "history_managing._write_image_to_stream"
    )

    history_managing.save_images_from_iteration(
        test_images["images_list"], save_dir, iter_number, concept
    )

    expected_path = "test/directory/iteration_1_concept_1"
    mock_makedirs.assert_called_with(expected_path, exist_ok=True)

    assert mock_open_file.call_count == 2
    assert mock_write_image_helper.call_count == 2
