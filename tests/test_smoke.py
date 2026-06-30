from pathlib import Path

from handover_compare.data import load_feature_base


def test_processed_datasets_load():
    root = Path(__file__).resolve().parents[1]
    for filename in [
        "anatel_concatbases.csv",
        "anatel_concatbases_gru.csv",
        "anatel_concatbases_tcn.csv",
    ]:
        X, y, _ = load_feature_base(root / "data" / "processed" / filename)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 50
        assert set(y).issubset({0, 1})
