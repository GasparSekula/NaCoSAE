"""Images page for displaying experiment results.

This module renders the images page of the Streamlit dashboard,
displaying images generated during the experiment.
"""

import streamlit as st

from utils import results
from utils import images_utils

experiment_results: results.ExperimentResults = st.session_state.get(
    results.RESULTS_STATE_KEY
)

images_utils.show_images(experiment_results=experiment_results)
