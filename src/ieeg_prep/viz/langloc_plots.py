"""Language localizer response plots."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from ..task_analysis.langloc.stat import amplitude_permutation_test


def plot_sent_nw_timeseries(
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
        ax.set_title(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if arr.shape[0] == 0:
            ax.text(0.5, 0.5, "No electrodes", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=11)
            return
        if show_channels:
            for ch_trace in arr:
                ax.plot(x, ch_trace, color=color, linewidth=0.5, alpha=0.15, zorder=1)
        mean, se = _mean_se(arr)
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2, zorder=2)
        ax.plot(x, mean, color=color, linewidth=2, zorder=3)

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


def plot_sent_nw_mean_amplitude(
    trial_tensor: np.ndarray,
    mask: np.ndarray,
    title: str = "Sentence vs Non-word by Mask",
    group_labels: tuple[str, str] = ("Language", "Non-language"),
    plot_points: bool = True,
    jitter: float = 0.0,
    point_alpha: float = 0.6,
    point_size: int = 18,
    run_permutation_test: bool = False,
    n_permutations: int = 5000,
    perm_seed: int | None = 42,
) -> plt.Figure:
    """Bar plot of sentence vs. non-word magnitude split by an electrode mask.

    Collapses the trial tensor across trials and time to produce one scalar per
    channel per condition, then plots grouped bars for each mask group.

    Parameters
    ----------
    trial_tensor : np.ndarray, shape (2, n_trials, n_channels, n_time)
        Output of :func:`~ieeg_prep.task_analysis.langloc.analysis.build_trial_tensor`.
        Axis 0 must be ``[sentence, non_word]``.
    mask : np.ndarray of bool, shape (n_channels,)
        ``True`` for electrodes in group 1 (e.g. language-responsive).
    title : str
        Axes title.
    group_labels : tuple of str
        Labels for the two mask groups (mask-True group first).
    plot_points : bool
        Whether to overlay individual electrode values with connecting lines.
    jitter : float
        Horizontal jitter applied to individual points (0 = no jitter).
    point_alpha : float
        Opacity of individual data points.
    point_size : float
        Marker size for individual data points.
    run_permutation_test : bool
        If True, run :func:`~ieeg_prep.task_analysis.langloc.stat.amplitude_permutation_test`
        for each group.  SE error bars are replaced by a horizontal dashed
        reference line at ``nw_group_mean + null_95th`` (the one-tailed
        significance threshold), and a symbol (``*`` / ``**`` / ``***`` /
        ``ns``) is drawn above each group.
    n_permutations : int
        Number of permutations (passed through when ``run_permutation_test=True``).
    perm_seed : int or None
        RNG seed for the permutation test.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    mask = np.asarray(mask, dtype=bool)

    # Collapse trials then time → (n_channels,) per condition
    avg = trial_tensor.mean(axis=1).mean(axis=-1)  # (2, n_channels)
    sent_vals = avg[0]
    nw_vals   = avg[1]

    group1, group2 = mask, ~mask
    sent_g1, nw_g1 = sent_vals[group1], nw_vals[group1]
    sent_g2, nw_g2 = sent_vals[group2], nw_vals[group2]

    def _mean_se(x: np.ndarray) -> tuple[float, float]:
        if len(x) == 0:
            return np.nan, np.nan
        if len(x) == 1:
            return float(np.mean(x)), 0.0
        return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))

    means = np.array([
        [_mean_se(sent_g1)[0], _mean_se(nw_g1)[0]],
        [_mean_se(sent_g2)[0], _mean_se(nw_g2)[0]],
    ])
    ses = np.array([
        [_mean_se(sent_g1)[1], _mean_se(nw_g1)[1]],
        [_mean_se(sent_g2)[1], _mean_se(nw_g2)[1]],
    ])

    x = np.arange(2)
    width = 0.34
    sent_pos = x - width / 2
    nw_pos   = x + width / 2

    sent_color = "#1f3a8a"
    nw_color   = "#b91c1c"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(sent_pos, means[:, 0], width=width, color=sent_color, label="Sentence", zorder=1)
    ax.bar(nw_pos,   means[:, 1], width=width, color=nw_color,   label="Non-word", zorder=1)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    for idx, grp in enumerate([group1, group2]):
        if grp.sum() == 0:
            ax.text(x[idx], 0.5, "No electrodes", transform=ax.get_xaxis_transform(),
                    ha="center", va="center", color="gray", fontsize=10)

    if plot_points:
        rng = np.random.default_rng(0)

        def _draw_group(sg: np.ndarray, ng: np.ndarray, idx: int) -> None:
            for sv, nv in zip(sg, ng):
                j1 = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
                j2 = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
                x1, x2 = sent_pos[idx] + j1, nw_pos[idx] + j2
                ax.plot([x1, x2], [sv, nv], linestyle="--", color="black",
                        linewidth=0.8, alpha=0.5, zorder=2)
                ax.scatter([x1, x2], [sv, nv], color="gray", alpha=point_alpha,
                           s=point_size, zorder=3)

        _draw_group(sent_g1, nw_g1, 0)
        _draw_group(sent_g2, nw_g2, 1)

    def _p_to_star(p: float) -> str:
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    if run_permutation_test:
        groups = [(mask, 0), (~mask, 1)]
        for grp_mask, idx in groups:
            if grp_mask.sum() == 0:
                continue
            result = amplitude_permutation_test(
                trial_tensor, grp_mask,
                n_permutations=n_permutations,
                seed=perm_seed,
            )
            null_threshold = float(np.percentile(result["null"], 95))
            ref_y = means[idx, 1] + null_threshold  # nw_mean + null 95th

            # Dashed reference line spanning both bars in this group
            ax.hlines(ref_y, x[idx] - width, x[idx] + width,
                      linestyles="--", colors="black", linewidth=1.0, zorder=6)

            # Significance symbol just above the top of the axes (data x, axes y)
            ax.text(x[idx], 1.02, _p_to_star(result["p_value"]),
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=13, zorder=7)
    else:
        ax.errorbar(sent_pos, means[:, 0], yerr=ses[:, 0], fmt="none",
                    ecolor="black", elinewidth=1.5, capsize=5, zorder=5)
        ax.errorbar(nw_pos,   means[:, 1], yerr=ses[:, 1], fmt="none",
                    ecolor="black", elinewidth=1.5, capsize=5, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Average magnitude")
    ax.set_xlabel("Mask condition")
    ax.set_title(title, pad=25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def plot_sent_nw_diff_amplitude(
    trial_tensor: np.ndarray,
    mask: np.ndarray,
    title: str = "Sentence − Non-word Amplitude by Mask",
    group_labels: tuple[str, str] = ("Language", "Non-language"),
    plot_points: bool = True,
    jitter: float = 0.0,
    point_alpha: float = 0.6,
    point_size: int = 18,
    run_permutation_test: bool = False,
    n_permutations: int = 5000,
    perm_seed: int | None = 42,
) -> plt.Figure:
    """Bar plot of the sentence − non-word amplitude difference split by an electrode mask.

    One bar per group (mask-True and mask-False), where each bar is the group
    mean of the per-electrode difference (sentence − non-word), averaged across
    trials and time.  A horizontal line at y = 0 marks the null.

    Parameters
    ----------
    trial_tensor : np.ndarray, shape (2, n_trials, n_channels, n_time)
        Output of :func:`~ieeg_prep.task_analysis.langloc.analysis.build_trial_tensor`.
        Axis 0 must be ``[sentence, non_word]``.
    mask : np.ndarray of bool, shape (n_channels,)
        ``True`` for electrodes in group 1 (e.g. language-responsive).
    title : str
        Axes title.
    group_labels : tuple of str
        Labels for the two mask groups (mask-True group first).
    plot_points : bool
        Whether to overlay individual electrode differences as scatter points.
    jitter : float
        Horizontal jitter applied to individual points (0 = no jitter).
    point_alpha : float
        Opacity of individual data points.
    point_size : float
        Marker size for individual data points.
    run_permutation_test : bool
        If True, run :func:`~ieeg_prep.task_analysis.langloc.stat.amplitude_permutation_test`
        for each group.  SE error bars are replaced by a dashed reference line
        at the null 95th percentile (one-tailed significance threshold) and a
        symbol (``*`` / ``**`` / ``***`` / ``ns``) above each bar.
    n_permutations : int
        Number of permutations (passed through when ``run_permutation_test=True``).
    perm_seed : int or None
        RNG seed for the permutation test.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    mask = np.asarray(mask, dtype=bool)

    # Per-electrode difference, averaged over trials then time → (n_channels,)
    avg = trial_tensor.mean(axis=1).mean(axis=-1)  # (2, n_channels)
    diff_vals = avg[0] - avg[1]                    # sent - nw per electrode

    diff_g1 = diff_vals[mask]
    diff_g2 = diff_vals[~mask]

    def _mean_se(x: np.ndarray) -> tuple[float, float]:
        if len(x) == 0:
            return np.nan, np.nan
        if len(x) == 1:
            return float(np.mean(x)), 0.0
        return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))

    bar_means = np.array([_mean_se(diff_g1)[0], _mean_se(diff_g2)[0]])
    bar_ses   = np.array([_mean_se(diff_g1)[1], _mean_se(diff_g2)[1]])

    x = np.arange(2)
    width = 0.5
    bar_color = "#1f3a8a"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(x, bar_means, width=width, color=bar_color, zorder=1)
    ax.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    for idx, grp in enumerate([mask, ~mask]):
        if grp.sum() == 0:
            ax.text(x[idx], 0.5, "No electrodes", transform=ax.get_xaxis_transform(),
                    ha="center", va="center", color="gray", fontsize=10)

    if plot_points:
        rng = np.random.default_rng(0)
        for idx, diffs in enumerate([diff_g1, diff_g2]):
            for d in diffs:
                j = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
                ax.scatter(x[idx] + j, d, color="gray", alpha=point_alpha,
                           s=point_size, zorder=3)

    def _p_to_star(p: float) -> str:
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    if run_permutation_test:
        for grp_mask, idx in [(mask, 0), (~mask, 1)]:
            if grp_mask.sum() == 0:
                continue
            result = amplitude_permutation_test(
                trial_tensor, grp_mask,
                n_permutations=n_permutations,
                seed=perm_seed,
            )
            null_threshold = float(np.percentile(result["null"], 95))

            # Dashed reference line at the null 95th percentile
            ax.hlines(null_threshold, x[idx] - width / 2, x[idx] + width / 2,
                      linestyles="--", colors="gray", linewidth=1.0, zorder=6)

            # Symbol just above the top of the axes (data x, axes y)
            ax.text(x[idx], 1.02, _p_to_star(result["p_value"]),
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=13, zorder=7)
    else:
        ax.errorbar(x, bar_means, yerr=bar_ses, fmt="none",
                    ecolor="black", elinewidth=1.5, capsize=5, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Sentence − Non-word amplitude")
    ax.set_xlabel("Mask condition")
    ax.set_title(title, pad=25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig
