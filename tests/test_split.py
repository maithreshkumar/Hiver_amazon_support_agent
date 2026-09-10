import pandas as pd

from hiver_support.data.prepare import assert_disjoint, temporal_split


def test_temporal_thread_split_is_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "thread_id": [str(index) for index in range(10)],
            "split_time": pd.date_range("2017-01-01", periods=10, tz="UTC"),
        }
    )
    splits = temporal_split(frame, {"train": 0.7, "dev": 0.1, "test": 0.2})
    assert_disjoint(splits)
    assert splits["train"] == {str(index) for index in range(7)}
    assert len(splits["test"]) == 2
