"""Experiment results dashboard application.

This module implements a Streamlit dashboard for viewing and analyzing
experiment results, including generated concepts, images, and LLM reasoning.
"""

import os

import streamlit as st

from utils import results
from components import plots
from utils import history_processing
from utils import dashboard_utils
from utils import exceptions

results_path = st.text_input("Results folder path")
if results_path:

    if not os.path.isdir(results_path):
        st.error(f"Path {results_path} does not exit.")
    else:
        directories = results.get_experiment_directories(results_path)
        if not directories:
            st.warning(
                "The specified folder does not contain any subfolders with results."
            )
        else:
            experiment_directory = st.selectbox(
                "Select experiment directory.", options=directories, index=None
            )

        if experiment_directory:
            try:
                experiment_results = results.load_experiment_results(
                    results_path, experiment_directory
                )
                st.session_state[results.RESULTS_STATE_KEY] = experiment_results

                experiment_results: results.ExperimentResults = (
                    st.session_state.get(results.RESULTS_STATE_KEY)
                )
                generation_history_scores = (
                    history_processing.get_scores_from_history(
                        experiment_results.generation_history
                    )
                )

                dashboard_utils.write_final_concept(
                    experiment_results=experiment_results
                )

                st.write("## Analyze results")

                dashboard_utils.write_parameters(
                    experiment_results=experiment_results
                )
                dashboard_utils.write_generated_concepts(
                    experiment_results=experiment_results
                )
                dashboard_utils.write_final_concept_set(
                    experiment_results=experiment_results
                )
                dashboard_utils.generate_plots(
                    generation_history_scores=generation_history_scores
                )
            except exceptions.CorruptedExperimentError as e:
                st.error(f"Experiment data error: {e}")
            except Exception as e:
                st.error(f"Unexpected error while loading data: {e}")
