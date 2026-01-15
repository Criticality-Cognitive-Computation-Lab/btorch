import matplotlib.pyplot as plt
import numpy as np


def plot_fi_vi_curve(
    results=None,
    plot_fi=True,
    plot_vi=True,
    get_data_func=None,
    data_func_kwargs=None,
    name="fi_vi_curve",
    file_path=None,
):
    """Plot f-I and V-I curves.

    Args:
        results: Dictionary containing 'currents', 'frequencies', 'voltages'.
        plot_fi: Whether to plot the f-I curve.
        plot_vi: Whether to plot the V-I curve (voltage traces).
        get_data_func: Function to generate results if not provided.
        data_func_kwargs: Arguments to pass to get_data_func.
        name: Name of the figure to save.
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
        return

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
    plt.tight_layout()
    if file_path is not None:
        fig.savefig(file_path / f"{name}.png")
    return fig
