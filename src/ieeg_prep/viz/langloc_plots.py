"""Language localizer response plots."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_sent_nw(
    trial_tensor: np.ndarray,
    conditions: list[str],
    frequency: float,
    lang_mask: np.ndarray | None = None,
    word_onsets: np.ndarray | None = None,
    show_channels: bool = False,
    title: str | None = None,
    output_path: str | None = None,
) -> plt.Figure:
    """Plot mean trial responses for sentence and non-word conditions.

    Averages the trial tensor across both the trial and channel dimensions.
    When ``lang_mask`` is omitted a 1×2 grid is produced (one panel per
    condition).  When provided a 2×2 grid is produced, splitting each
    condition panel by language-responsive vs. non-language electrodes.

    Parameters
    ----------
    trial_tensor : np.ndarray, shape (2, n_trials, n_channels, n_time)
        Output of :func:`~ieeg_prep.task_analysis.langloc.analysis.build_trial_tensor`.
        Axis 0 must be ``[sentence, non_word]``.
    conditions : list of str
        Condition labels returned alongside the tensor (always
        ``["sentence", "non_word"]``).
    frequency : float
        Sampling frequency in Hz, used to convert sample indices to seconds.
    show_channels : bool, default False
        If True, plot each channel's trial-averaged trace as a faint line
        behind the mean ± SE.
    lang_mask : np.ndarray of bool, shape (n_channels,), optional
        Language-responsive electrode mask.  If provided, each condition is
        split into language (top row) and non-language (bottom row) panels.
    word_onsets : np.ndarray of int, shape (n_words,), optional
        Word onset times in samples, as returned by
        :func:`~ieeg_prep.task_analysis.langloc.analysis.build_trial_tensor`.
        Plotted as vertical dashed lines.
    title : str, optional
        Figure suptitle.
    output_path : str or Path, optional
        If provided, save the figure here instead of calling ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    def _fmt(cond: str) -> str:
        return cond.replace("_", "-").capitalize()

    cond_labels = [_fmt(c) for c in conditions]  # e.g. ["Sentence", "Non-word"]

    # Average across trials -> (2, n_channels, n_time)
    avg = trial_tensor.mean(axis=1)
    sent_avg = avg[0]   # (n_channels, n_time)
    nw_avg   = avg[1]

    n_time = sent_avg.shape[1]
    x = np.arange(n_time) / frequency

    def _mean_se(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and SE across channels at each timepoint."""
        mean = arr.mean(axis=0)
        se   = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        return mean, se

    def _fill_panel(ax: plt.Axes, arr: np.ndarray, color: str, label: str) -> None:
        if show_channels:
            for ch_trace in arr:
                ax.plot(x, ch_trace, color=color, linewidth=0.5, alpha=0.15, zorder=1)
        mean, se = _mean_se(arr)
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2, zorder=2)
        ax.plot(x, mean, color=color, linewidth=2, zorder=3)
        ax.set_title(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _add_word_onsets(axes_flat: list[plt.Axes]) -> None:
        if word_onsets is None:
            return
        onsets_s = np.asarray(word_onsets) / frequency
        for ax in axes_flat:
            for i, onset in enumerate(onsets_s):
                ax.axvline(
                    onset, color="black", linestyle="--",
                    alpha=0.5, linewidth=1.0, zorder=0,
                    label="Word onset" if ax is axes_flat[0] and i == 0 else None,
                )

    masked = lang_mask is not None

    if masked:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        axes_flat = list(axes.ravel())
        _add_word_onsets(axes_flat)

        _fill_panel(axes[0, 0], sent_avg[lang_mask],  "blue",   f"Language — {cond_labels[0]}")
        _fill_panel(axes[0, 1], nw_avg[lang_mask],    "purple", f"Language — {cond_labels[1]}")
        _fill_panel(axes[1, 0], sent_avg[~lang_mask], "red",    f"Non-language — {cond_labels[0]}")
        _fill_panel(axes[1, 1], nw_avg[~lang_mask],   "gray",   f"Non-language — {cond_labels[1]}")

        axes[0, 0].set_ylabel("Average response")
        axes[1, 0].set_ylabel("Average response")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 1].set_xlabel("Time (s)")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        axes_flat = list(axes)
        _add_word_onsets(axes_flat)

        _fill_panel(axes[0], sent_avg, "blue",   cond_labels[0])
        _fill_panel(axes[1], nw_avg,   "purple", cond_labels[1])

        axes[0].set_ylabel("Average response")
        axes[0].set_xlabel("Time (s)")
        axes[1].set_xlabel("Time (s)")

    if word_onsets is not None:
        axes_flat[0].legend(frameon=True, facecolor="white", edgecolor="black")

    if title is not None:
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig
