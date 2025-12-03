import streamlit as st

from components import plots
from utils import results
from utils import history_processing

experiment_results: results.ExperimentResults = st.session_state.get(
    results.RESULTS_STATE_KEY
)
generation_history_scores = history_processing.get_scores_from_history(
    experiment_results.generation_history
)
plots.plot_score_vs_iteration(generation_history_scores)
