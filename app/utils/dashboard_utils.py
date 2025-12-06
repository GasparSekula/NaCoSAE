from typing import Sequence

import streamlit as st

from components import plots
from utils import results


def write_parameters(experiment_results: results.ExperimentResults) -> None:
    st.write("## Parameters")
    st.write(
        f"**Explained model:** {str(experiment_results.run_params["load_config"]["explained_model_id"])}"
    )
    st.write(
        f"- Layer: {str(experiment_results.run_params["model_layer_activations_path"].split("/")[-1])}"
    )
    st.write(f"- Neuron: {str(experiment_results.run_params["neuron_id"])}")
    st.write(
        f"**Language Model:** {str(experiment_results.run_params["load_config"]["language_model_id"])}"
    )
    st.write(
        f"**Text-To-Image Model:** {str(experiment_results.run_params["load_config"]["text_to_image_model_id"])}"
    )
    st.write(
        f"**Number of iterations:** {len(experiment_results.generation_history) - 1} + 1 summary run"
    )


def write_generated_concepts(
    experiment_results: results.ExperimentResults,
) -> None:
    st.write("## Generated concepts")
    st.dataframe(
        {
            "Iteration": range(
                1, len(experiment_results.generation_history) + 1
            ),
            "Generated Concept": [
                concept[0] for concept in experiment_results.generation_history
            ],
            "Score": [
                round(float(concept[1]), 2)
                for concept in experiment_results.generation_history
            ],
        },
        hide_index=True,
    )


def write_final_concept_set(
    experiment_results: results.ExperimentResults,
) -> None:
    st.write("## Final concepts set")
    st.write("Includes best concepts among initial ones and generated ones.")
    st.dataframe(
        {
            "Iteration": range(
                1, len(experiment_results.final_concept_history) + 1
            ),
            "Generated Concept": [
                concept[0]
                for concept in experiment_results.final_concept_history
            ],
            "Score": [
                round(float(concept[1]), 2)
                for concept in experiment_results.final_concept_history
            ],
        },
        hide_index=True,
    )


def generate_plots(generation_history_scores: Sequence[float]) -> None:
    st.write(" ")
    plots.plot_design_setup()

    plot_option = st.selectbox(
        "Visualize results",
        [
            "Score vs Iteration",
            "Relative Score vs Iteration",
            "Cumulative Best Score vs Iteration",
        ],
        placeholder="Select plot...",
    )
    if plot_option == "Score vs Iteration":
        plots.plot_score_vs_iteration(generation_history_scores)
    elif plot_option == "Relative Score vs Iteration":
        plots.plot_relative_score_over_iteration(generation_history_scores)
    elif plot_option == "Cumulative Best Score vs Iteration":
        plots.plot_best_score_over_iteration(generation_history_scores)
