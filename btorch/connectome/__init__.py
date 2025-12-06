import pandas as pd


def simple_id_to_root_id(neurons: pd.DataFrame, reverse: bool = False):
    return dict(
        neurons[
            ["root_id", "simple_id"] if reverse else ["simple_id", "root_id"]
        ].to_numpy()
    )
