"""Pandas DataFrame utilities.

Helpers for common DataFrame operations used in connectome analysis and
data aggregation workflows.
"""

from typing import Any, Optional, Sequence

import pandas as pd


def groupby_to_dict(
    df: pd.DataFrame, column_select: Optional[Sequence[str]] = None, **groupby_args
) -> dict[Any, pd.DataFrame]:
    """Group DataFrame and return as dictionary mapping keys to subframes.

    Args:
        df: Input DataFrame to group.
        column_select: Optional column subset to include in output values.
        **groupby_args: Arguments passed to ``df.groupby()``.

    Returns:
        Dictionary mapping group keys to DataFrame slices.

    Example:
        >>> df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
        >>> groupby_to_dict(df, column_select=["b"], by="a")
        {1:    b
         0  3
         1  4, 2:    b
         2  5}
    """
    return {
        key: df.loc[ind, column_select] if column_select is not None else df.loc[ind]
        for key, ind in df.groupby(**groupby_args).groups.items()
    }
