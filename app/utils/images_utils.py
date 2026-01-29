"""Utility functions for displaying generated images in the dashboard.

This module provides helper functions to render and display images from
experiment results in a formatted grid layout.
"""

import streamlit as st

from utils import results


def _show_images_row(images: list, cols: list, ncols: int) -> None:
    """Display a row of images in specified columns.

    Loads and displays images from file paths in the given columns,
    handling errors gracefully if an image cannot be read.

    Args:
        images: List of image file objects with a name attribute.
        cols: List of Streamlit column objects to display images in.
        ncols: Number of columns available for image layout.
    """
    for idx, img_file in enumerate(images):
        img_bytes = None
        with open(img_file.name, "rb") as f:
            img_bytes = f.read()

        if not img_bytes:
            st.warning(f"Skipping image {idx+1} because it could not be read.")
            continue

        col = cols[idx % ncols]
        col.image(
            img_bytes,
            caption=f"Image {idx+1}",
            width="stretch",
        )


def show_images(experiment_results: results.ExperimentResults) -> None:
    """Display all generated images from experiment results.

    Organizes and displays images by iteration and concept, with the summary
    iteration highlighted at the end. Images are arranged in a responsive grid.

    Args:
        experiment_results: The experiment results containing generated images.
    """
    st.markdown("# Generated images")

    images_list = experiment_results.images

    formatted_images_list = []

    for k, v in images_list.items():
        k_split = k.split("_")
        formatted_images_list.append(
            {
                "iteration": k_split[1],
                "concept": " ".join(k_split[2:]),
                "images": v,
            }
        )

    formatted_images_list.sort(key=lambda item: int(item["iteration"]))

    max_iteration = max(
        int(item["iteration"]) for item in formatted_images_list
    )

    for item in formatted_images_list:
        if (
            int(item["iteration"]) == max_iteration
            and item["concept"].strip().lower()
            == experiment_results.generation_history[-1][0].strip().lower()
        ):
            item["iteration"] = "Summary"

    if not formatted_images_list:
        st.write("No images to display.")
    else:

        summary_concept = None

        for iter_data in formatted_images_list:
            iter = iter_data["iteration"]
            concept = iter_data["concept"]
            images = iter_data["images"]

            if iter != "Summary":
                st.subheader(f"Iteration {iter} - {concept}")

                ncols = max(min(5, len(images)), 1)
                cols = st.columns(ncols)

                _show_images_row(images=images, cols=cols, ncols=ncols)
            else:
                summary_concept = concept

        st.subheader(f"Summary iteration - {summary_concept}")
        images = formatted_images_list[-1]["images"]
        ncols = max(min(5, len(images)), 1)
        cols = st.columns(ncols)

        _show_images_row(images=images, cols=cols, ncols=ncols)
