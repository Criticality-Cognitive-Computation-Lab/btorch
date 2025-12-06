from typing import Any, Optional, Sequence

import pandas as pd


def groupby_to_dict(
    df: pd.DataFrame, column_select: Optional[Sequence[str]] = None, **groupby_args
) -> dict[Any, pd.DataFrame]:
    return {
        key: df.loc[ind, column_select] if column_select is not None else df.loc[ind]
        for key, ind in df.groupby(**groupby_args).groups.items()
    }
