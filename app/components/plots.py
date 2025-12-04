from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

COLOR = "red"


def plot_design_setup() -> None:
    plt.rcParams["figure.figsize"] = (16, 9)
    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["savefig.facecolor"] = "none"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["axes.titlecolor"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"
    plt.rcParams["font.size"] = 20


def plot_score_vs_iteration(
    generation_history_scores: Sequence[float],
) -> None:
    scores_array = np.array(generation_history_scores, dtype=np.float32)
    x_ticks = range(1, len(scores_array) + 1)

    fig, ax = plt.subplots()
    ax.set_title("Score vs iteration")
    ax.plot(x_ticks, scores_array, color=COLOR)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_ylim(bottom=0.9 * min(scores_array), top=1.1 * max(scores_array))
    ax.set_xticks(x_ticks)
    ax.grid(True)
    st.pyplot(fig)


def plot_relative_score_over_iteration(
    generation_history_scores: Sequence[float],
) -> None:
    scores_array = np.array(generation_history_scores, dtype=np.float32)
    best_score = max(scores_array)
    relative_scores_array = scores_array / best_score
    x_ticks = range(1, len(relative_scores_array) + 1)

    fig, ax = plt.subplots()
    ax.set_title("Relative score vs iteration")
    ax.plot(x_ticks, relative_scores_array, color=COLOR)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative score")
    ax.set_ylim(bottom=0.9 * min(relative_scores_array), top=1.1)
    ax.set_xticks(x_ticks)
    ax.grid(True)
    st.pyplot(fig)


def plot_best_score_over_iteration(
    generation_history_scores: Sequence[float],
) -> None:
    scores_array = np.array(generation_history_scores, dtype=np.float32)
    cumulative_best_score = np.maximum.accumulate(scores_array)
    x_ticks = range(1, len(scores_array) + 1)

    fig, ax = plt.subplots()
    ax.set_title("Cumulative best score vs iteration")
    ax.plot(x_ticks, cumulative_best_score, color=COLOR)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative score")
    ax.set_ylim(
        bottom=0.9 * min(cumulative_best_score), top=1.1 * max(scores_array)
    )
    ax.set_xticks(x_ticks)
    ax.grid(True)
    st.pyplot(fig)
