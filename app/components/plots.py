from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import streamlit as st


def plot_score_vs_iteration(
    generation_history_scores: Sequence[float],
) -> None:
    fig, ax = plt.subplots()
    ax.plot(generation_history_scores)
    st.pyplot(fig)
