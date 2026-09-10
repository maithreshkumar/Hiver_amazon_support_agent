from hiver_support.data.threads import component_map


def test_components_are_undirected_and_isolated() -> None:
    mapping = component_map(["1", "2", "3", "9"], [("2", "1"), ("2", "3")])
    assert mapping["1"] == mapping["2"] == mapping["3"] == "1"
    assert mapping["9"] == "9"

