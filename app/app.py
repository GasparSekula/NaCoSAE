import streamlit as st

from utils import results

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
