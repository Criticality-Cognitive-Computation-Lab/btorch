import pandas as pd

from btorch.visualisation.hexmap import hex_heatmap


def test_hex_heatmap_multi_column_creates_frames():
    dataset = pd.DataFrame({"p": [0, 1], "q": [0, 0]})
    values = pd.DataFrame(
        {
            "p": [0, 1],
            "q": [0, 0],
            "metric_a": [0.1, 0.2],
            "metric_b": [0.3, 0.4],
        }
    )

    fig = hex_heatmap(values.copy(), dataset)

    assert len(fig.frames) == 2
    assert all(len(frame.data) == 2 for frame in fig.frames)
    assert len(fig.data) == 2  # initial background + first metric
