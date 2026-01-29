from typing import Sequence

import streamlit as st

from components import plots
from utils import results


def write_final_concept(experiment_results: results.ExperimentResults) -> None:
    """Display the best proposed concept from the experiment.

    Writes the highest-scoring concept from the final concept history
    as a markdown heading to the Streamlit dashboard.

    Args:
        experiment_results: The experiment results containing concept history.
    """
    concept_history = experiment_results.final_concept_history
    best_concept = max(concept_history, key=lambda x: x[1])[0]

    st.write(f"### Proposed concept: {best_concept}")


def write_parameters(experiment_results: results.ExperimentResults) -> None:
    """Display experiment parameters and configuration.

    Writes all relevant experiment parameters to the Streamlit dashboard,
    including model IDs, layer/neuron information, and metric details.

    Args:
        experiment_results: The experiment results containing run parameters.
    """
    st.write("### Parameters")
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
    st.write(f"**Metric:** {str(experiment_results.run_params["metric"])}")


def write_generated_concepts(
    experiment_results: results.ExperimentResults,
) -> None:
    """Display all generated concepts across iterations.

    Creates and displays a table showing the iteration number, generated concept,
    and score for each iteration of the experiment.

    Args:
        experiment_results: The experiment results containing generation history.
    """
    st.markdown(
        """
        <style>
        thead tr th:first-child {display:none}
        tbody th {display:none}
        </style>
    """,
        unsafe_allow_html=True,
    )
    st.write("### Generated concepts")

    iters = [
        i for i in range(1, len(experiment_results.generation_history) + 1)
    ]
    iters[-1] = "Summary"
    st.table(
        {
            "Iteration": iters,
            "Generated Concept": [
                concept[0] for concept in experiment_results.generation_history
            ],
            "Score": [
                round(float(concept[1]), 2)
                for concept in experiment_results.generation_history
            ],
        }
        # hide_index=True,
    )


def write_final_concept_set(
    experiment_results: results.ExperimentResults,
) -> None:
    """Display the final set of best concepts.

    Shows a table of the final concepts including both initial and generated
    concepts, sorted by their scores.

    Args:
        experiment_results: The experiment results containing final concept history.
    """
    st.write("### Final concepts set")
    st.write("Includes best concepts among initial and generated ones.")
    st.table(
        {
            "Concept": [
                concept[0]
                for concept in experiment_results.final_concept_history
            ],
            "Score": [
                round(float(concept[1]), 2)
                for concept in experiment_results.final_concept_history
            ],
        },
        # hide_index=True,
    )


def generate_plots(generation_history_scores: Sequence[float]) -> None:
    """Generate and display interactive score visualization plots.

    Creates a selectbox for the user to choose between different plot types
    (Score vs Iteration, Relative Score, or Cumulative Best Score) and
    displays the selected plot.

    Args:
        generation_history_scores: Sequence of scores from each iteration.
    """
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
