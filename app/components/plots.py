from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

COLOR = "#1a5e9a"


def plot_design_setup() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 14

    navy_color = "#1a5e9a"
    grid_color = "#e6e6e6"

    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["savefig.facecolor"] = "none"

    plt.rcParams["text.color"] = "#212529"
    plt.rcParams["axes.labelcolor"] = navy_color
    plt.rcParams["axes.titlecolor"] = navy_color
    plt.rcParams["xtick.color"] = "#495057"
    plt.rcParams["ytick.color"] = "#495057"

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True
    plt.rcParams["axes.edgecolor"] = "#ced4da"

    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = grid_color
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5


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
    ax.set_ylim(
        bottom=np.min(scores_array) - 0.1 * abs(np.min(scores_array)),
        top=np.max(scores_array) + 0.1 * abs(np.max(scores_array)),
    )
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
    ax.set_ylim(
        bottom=np.min(relative_scores_array)
        - 0.1 * abs(np.min(relative_scores_array)),
        top=np.max(relative_scores_array)
        + 0.1 * abs(np.max(relative_scores_array)),
    )
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
        bottom=np.min(cumulative_best_score)
        - 0.1 * abs(np.min(cumulative_best_score)),
        top=np.max(scores_array) + 0.1 * abs(np.max(scores_array)),
    )
    ax.set_xticks(x_ticks)
    ax.grid(True)
    st.pyplot(fig)
