import streamlit as st
import pandas as pd

from utils import results
from utils import reasoning_utils

experiment_results: results.ExperimentResults = st.session_state.get(
    results.RESULTS_STATE_KEY
)

st.markdown("# LLM's reasoning process")

reasoning_utils.write_reasoning(experiment_results=experiment_results)
