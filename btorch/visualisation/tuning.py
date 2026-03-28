"""Neuron response curve visualization.

Plotting utilities for f-I (frequency-current) and V-I (voltage-current)
curves, commonly used to characterize neuron excitability and response
properties.
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_fi_vi_curve(
    results: dict | None = None,
    plot_fi: bool = True,
    plot_vi: bool = True,
    get_data_func: Callable | None = None,
    data_func_kwargs: dict | None = None,
    name: str = "fi_vi_curve",
    file_path: str | None = None,
) -> Figure:
    """Plot f-I and V-I response curves for neuron characterization.

    Generates subplots showing the relationship between input current and
    firing rate (f-I curve), and input current and membrane voltage traces
    (V-I curve, displayed as waterfall plot).

    Args:
        results: Dictionary containing simulation results with keys:
            - "currents": Input current values, shape (n_steps,)
            - "frequencies": Firing rates in Hz, shape (n_steps,)
            - "voltages": Voltage traces, shape (time, n_steps)
            - "time": Optional time array, shape (time,)
        plot_fi: Whether to plot the f-I curve.
        plot_vi: Whether to plot the V-I waterfall traces.
        get_data_func: Function to generate results if not provided.
            Called as `get_data_func(**data_func_kwargs)`.
        data_func_kwargs: Arguments passed to `get_data_func`.
        name: Base filename for saving (if file_path is provided).
        file_path: Directory path for saving figure. If None, figure is
            returned but not saved.

    Returns:
        Figure containing the requested subplots.

    Raises:
        ValueError: If neither `results` nor `get_data_func` is provided.

    Example:
        >>> results = {
        ...     "currents": torch.linspace(0, 10, 20),
        ...     "frequencies": firing_rates,
        ...     "voltages": voltage_traces,
        ... }
        >>> fig = plot_fi_vi_curve(results, file_path="./output/")
    """
    if results is None:
        if get_data_func is None:
            raise ValueError("Either 'results' or 'get_data_func' must be provided.")
        results = get_data_func(**(data_func_kwargs or {}))

    currents = results["currents"].detach().cpu().numpy()
    frequencies = results["frequencies"].detach().cpu().numpy()
    voltages = results["voltages"].detach().cpu().numpy()

    # Determine number of subplots
    num_plots = sum([plot_fi, plot_vi])
    if num_plots == 0:
        raise ValueError("At least one of plot_fi or plot_vi must be True.")

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), squeeze=False)
    axes = axes.flatten()
    plot_idx = 0

    if plot_fi:
        ax = axes[plot_idx]
        ax.plot(currents, frequencies, "o-")
        ax.set_xlabel("Input Current")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_title("f-I Curve")
        ax.grid(True)
        plot_idx += 1

    if plot_vi:
        ax = axes[plot_idx]

        time = results.get("time")
        if time is not None:
            time = time.detach().cpu().numpy()
        else:
            time = np.arange(voltages.shape[0])

        # Use "waterfall" plot: offset traces vertically
        # Calculate offset based on voltage range
        v_min = voltages.min()
        v_max = voltages.max()
        v_range = v_max - v_min
        if v_range == 0:
            v_range = 1.0  # Avoid division by zero

        # Offset amount: fraction of range per trace
        offset_step = v_range * 0.2  # Space out well

        num_steps = voltages.shape[1]
        cm = plt.get_cmap("viridis")

        yticks = []
        yticklabels = []

        for i in range(num_steps):
            color = cm(i / num_steps)
            offset = i * offset_step
            # Plot trace with offset
            ax.plot(
                time, voltages[:, i] + offset, color=color, alpha=0.8, linewidth=1.0
            )

            # Record tick for this trace (centered on its baseline approx)
            # Or just label current values
            if i % (max(1, num_steps // 5)) == 0:  # Sparse labels
                yticks.append(voltages[0, i] + offset)  # Assuming start at rest
                yticklabels.append(f"{currents[i]:.1f}")

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Membrane Potential (Offset)")
        ax.set_title("Voltage Traces (Waterfall)")

        # Add a secondary axis or just colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cm, norm=plt.Normalize(vmin=currents.min(), vmax=currents.max())
        )
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Input Current")

        plot_idx += 1

    plt.tight_layout()
    if file_path is not None:
        fig.savefig(f"{file_path}/{name}.png")
    return fig
