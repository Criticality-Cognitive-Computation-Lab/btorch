import pandas as pd


def simple_id_to_root_id(neurons: pd.DataFrame, reverse: bool = False) -> dict:
    """Build an id lookup from a neuron table.

    Args:
        neurons: DataFrame with ``root_id`` and ``simple_id`` columns.
        reverse: If False (default), return a ``simple_id -> root_id`` mapping,
            as the function name says. If True, return the reversed mapping,
            ``root_id -> simple_id``.

    Returns:
        Mapping dictionary in the requested direction.
    """
    return dict(
        neurons[
            ["root_id", "simple_id"] if reverse else ["simple_id", "root_id"]
        ].to_numpy()
    )
