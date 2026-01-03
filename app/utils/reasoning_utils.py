import streamlit as st

from utils import results


def write_reasoning(experiment_results: results.ExperimentResults) -> None:
    reasoning_list = experiment_results.reasoning

    for i in range(len(reasoning_list) - 1):
        st.markdown(f"### Iteration {i+1} - {reasoning_list[i][0]}")
        st.markdown(f"{reasoning_list[i][1]}")

    st.markdown(
        f"### **Summary concept:** {reasoning_list[len(reasoning_list)-1][0]}"
    )
    st.markdown(f"{reasoning_list[len(reasoning_list)-1][1]}")
