"""Reasoning page for displaying LLM's reasoning process.

This module renders the reasoning page of the Streamlit dashboard,
displaying the large language model's step-by-step reasoning and explanations
from the experiment results.
"""

import streamlit as st
import pandas as pd

from utils import results
from utils import reasoning_utils

experiment_results: results.ExperimentResults = st.session_state.get(
    results.RESULTS_STATE_KEY
)

st.markdown("# LLM's reasoning process")

reasoning_utils.write_reasoning(experiment_results=experiment_results)
