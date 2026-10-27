"""Проверки ваших агрегатов. Пока функция не написана, тест помечается как skipped."""
import numpy as np
import pytest

from src.config import ID_COL
from tests.fake_data import fake_bureau


def _bureau_features():
    from src.features.bureau import PREFIX, build_bureau_features
    bureau = fake_bureau(np.arange(100000, 101000), seed=3)
    try:
        feats = build_bureau_features(bureau)
    except NotImplementedError:
        pytest.skip("build_bureau_features ещё не написана (шаг 2)")
    return bureau, feats, PREFIX


def test_one_row_per_client():
    bureau, feats, _ = _bureau_features()
    assert feats[ID_COL].is_unique
    assert set(feats[ID_COL]) == set(bureau[ID_COL])


def test_columns_prefixed_and_no_target():
    _, feats, prefix = _bureau_features()
    other = [c for c in feats.columns if c != ID_COL]
    assert other, "нет ни одного признака"
    assert all(c.startswith(prefix) for c in other)
    assert "TARGET" not in feats.columns


def test_no_infinities():
    _, feats, _ = _bureau_features()
    num = feats.drop(columns=[ID_COL]).select_dtypes(include="number")
    assert not np.isinf(num.to_numpy(dtype=float)).any()
