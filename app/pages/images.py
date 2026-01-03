import streamlit as st

from utils import results
from utils import images_utils

experiment_results: results.ExperimentResults = st.session_state.get(
    results.RESULTS_STATE_KEY
)

images_utils.show_images(experiment_results=experiment_results)
