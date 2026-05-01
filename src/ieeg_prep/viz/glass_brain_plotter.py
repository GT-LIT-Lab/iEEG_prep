"""Glass brain visualization for iEEG electrode coordinates."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from nilearn import plotting


def plot_glass_brain(
    coords,
    data,
    masked=False,
    colors=None,
    labels=None,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    center=None,
    symmetric=False,
    s=20,
    title="Glass Brain",
    colorbar=True,
    cmap_label="Value",
    output_path=None,
):
    """
    Plot electrode coordinates on a glass brain.

    Supports two modes controlled by the ``masked`` flag:

    - **Continuous mode** (``masked=False``, default): ``data`` is a 1D array of
      scalar values; electrodes are colored by value using ``cmap``.
    - **Mask mode** (``masked=True``): ``data`` is a list of boolean arrays, one
      per group; each group is plotted in a distinct color.

    In both modes, point transparency encodes approximate depth to give a
    pseudo-3-D appearance.

    Parameters
    ----------
    coords : array-like, shape (n_channels, 3)
        MNI coordinates for each electrode.
    data : array-like or list of array-like
        * ``masked=False``: 1-D array of scalar values, shape ``(n_channels,)``.
          ``NaN`` / ``Inf`` values are skipped.
        * ``masked=True``: list of boolean arrays of length ``n_channels``,
          one per group to highlight.
    masked : bool, default False
        Switch between continuous (``False``) and mask (``True``) mode.
    colors : list of str, optional
        Required when ``masked=True``. One color per mask group (e.g.
        ``["red", "blue"]``).
    labels : list of str, optional
        Required when ``masked=True``. One legend label per mask group.
    cmap : str or Colormap, default "coolwarm"
        Colormap used in continuous mode.
    vmin, vmax : float or None
        Color-scale limits for continuous mode.  Defaults to the
        min / max of the finite values in ``data``.
    center : float or None
        If provided, a diverging ``TwoSlopeNorm`` is used centered at this
        value (continuous mode only).
    symmetric : bool, default False
        Expand ``vmin`` / ``vmax`` symmetrically around ``center`` (or 0 if
        ``center`` is ``None``) so that the colormap is balanced (continuous
        mode only).
    s : int or float, default 20
        Scatter marker size.
    title : str, default "Glass Brain"
        Figure title.
    colorbar : bool, default True
        Show a colorbar (continuous mode only).
    output_path : str or Path or None, default None
        If provided, save the figure to this path instead of calling
        ``plt.show()``.  The format is inferred from the file extension
        (e.g. ``.png``, ``.svg``).

    Returns
    -------
    display : nilearn GlassBrainDisplay
    norm : matplotlib Normalize  (continuous mode) or ``None`` (mask mode)
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_channels, 3)")

    # --- depth-based alpha (shared) ---
    z_all = coords[:, 2]
    z_norm = (z_all - z_all.min()) / (z_all.max() - z_all.min() + 1e-8)

    x_all = coords[:, 0]
    ax_alpha = 0.2 + 0.8 * (
        1 - (x_all - x_all.min()) / (x_all.max() - x_all.min() + 1e-8)
    )
    cor_alpha = 0.2 + 0.8 * z_norm
    axi_alpha = 0.2 + 0.8 * np.abs(z_norm - 0.5) * 2

    display = plotting.plot_glass_brain(None, display_mode="ortho", title=title)

    ax_x = display.axes["x"].ax  # sagittal (y vs z)
    ax_x.invert_xaxis()
    ax_y = display.axes["y"].ax  # coronal  (x vs z)
    ax_z = display.axes["z"].ax  # axial    (x vs y)

    # ------------------------------------------------------------------
    if masked:
        # --- mask mode ---
        if colors is None or labels is None:
            raise ValueError("colors and labels are required when masked=True")
        if len(data) != len(colors) or len(data) != len(labels):
            raise ValueError("data (masks), colors, and labels must have the same length")

        legend_handles = []

        for mask, color, label in zip(data, colors, labels):
            mask = np.asarray(mask, dtype=bool)
            x = coords[mask, 0]
            y = coords[mask, 1]
            z = coords[mask, 2]

            for yi, zi, a in zip(y, z, ax_alpha[mask]):
                ax_x.scatter(yi, zi, s=s, c=color, alpha=a)
            for xi, zi, a in zip(x, z, cor_alpha[mask]):
                ax_y.scatter(xi, zi, s=s, c=color, alpha=a)
            for xi, yi, a in zip(x, y, axi_alpha[mask]):
                ax_z.scatter(xi, yi, s=s, c=color, alpha=a)

            legend_handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o", color="w",
                    markerfacecolor=color, markersize=8, label=label,
                )
            )

        ax_z.legend(handles=legend_handles, loc="lower right", framealpha=0.7)
        if output_path is not None:
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
        return display, None

    # --- continuous mode ---
    values = np.asarray(data, dtype=float)
    if values.ndim != 1:
        raise ValueError("data must be a 1-D array when masked=False")
    if len(values) != len(coords):
        raise ValueError("data must have the same length as coords")

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        raise ValueError("data contains no finite entries")

    vals_finite = values[finite_mask]

    if vmin is None:
        vmin = vals_finite.min()
    if vmax is None:
        vmax = vals_finite.max()

    if symmetric:
        c = 0.0 if center is None else center
        radius = max(abs(vals_finite.min() - c), abs(vals_finite.max() - c))
        vmin = c - radius
        vmax = c + radius

    if center is not None:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cmap_obj = plt.get_cmap(cmap)

    for i in range(len(coords)):
        if not np.isfinite(values[i]):
            continue
        x, y, z = coords[i]
        color = cmap_obj(norm(values[i]))
        ax_x.scatter(y, z, s=s, c=[color], alpha=ax_alpha[i])
        ax_y.scatter(x, z, s=s, c=[color], alpha=cor_alpha[i])
        ax_z.scatter(x, y, s=s, c=[color], alpha=axi_alpha[i])

    if colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cax = plt.gcf().add_axes([1.05, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(cmap_label)

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return display, norm
