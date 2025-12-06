import streamlit as st

from utils import results
from components import plots
from utils import history_processing
from utils import dashboard_utils

results_path = st.text_input("Results folder path")
if results_path:
    experiment_directory = st.selectbox(
        "Select experiment directory",
        options=results.get_experiment_directories(results_path),
        index=None,
    )
    if experiment_directory:
        experiment_results = results.load_experiment_results(
            results_path, experiment_directory
        )
        st.session_state[results.RESULTS_STATE_KEY] = experiment_results

        experiment_results: results.ExperimentResults = st.session_state.get(
            results.RESULTS_STATE_KEY
        )
        generation_history_scores = history_processing.get_scores_from_history(
            experiment_results.generation_history
        )

        st.write("# Analyze model output")

        dashboard_utils.write_parameters(experiment_results=experiment_results)
        dashboard_utils.write_generated_concepts(
            experiment_results=experiment_results
        )
        dashboard_utils.write_final_concept_set(
            experiment_results=experiment_results
        )
        dashboard_utils.generate_plots(
            generation_history_scores=generation_history_scores
        )
