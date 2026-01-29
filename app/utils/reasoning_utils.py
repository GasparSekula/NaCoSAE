"""Utility functions for displaying LLM reasoning in the dashboard.

This module provides helper functions to render the language model's
reasoning process and explanations from experiment results.
"""

import streamlit as st

from utils import results


def write_reasoning(experiment_results: results.ExperimentResults) -> None:
    """Display the LLM's reasoning process for each iteration.

    Writes the reasoning text and concept names for each iteration,
    with the final summary iteration highlighted separately.

    Args:
        experiment_results: The experiment results containing reasoning data.
    """
    reasoning_list = experiment_results.reasoning

    for i in range(len(reasoning_list) - 1):
        st.markdown(f"### Iteration {i+1} - {reasoning_list[i][0]}")
        st.markdown(f"{reasoning_list[i][1]}")

    st.markdown(
        f"### **Summary concept:** {reasoning_list[len(reasoning_list)-1][0]}"
    )
    st.markdown(f"{reasoning_list[len(reasoning_list)-1][1]}")
