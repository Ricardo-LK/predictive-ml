import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from predictive_ml import data
from predictive_ml.artifact import FEATURE_COLUMNS
from predictive_ml.train import build_preprocessor, eval_thresholds

from .conftest import make_frame


def test_preprocessor_splits_numeric_and_categorical_columns():
    X, _ = make_frame()
    pre = build_preprocessor(X).fit(X)

    cols = {name: c for name, _, c in pre.transformers_ if name != "remainder"}
    assert cols["categ"] == ["Type"]
    assert cols["num"] == FEATURE_COLUMNS[1:]
    # 5 scaled numeric columns + 3 one-hot product types
    assert pre.transform(X).shape == (len(X), 8)


def test_preprocessor_imputes_missing_values():
    X, _ = make_frame()
    pre = build_preprocessor(X).fit(X)

    row = X.iloc[[0]].copy()
    row.loc[:, "Torque"] = np.nan
    row.loc[:, "Type"] = None

    out = np.asarray(pre.transform(row))
    assert not np.isnan(out).any()


def test_eval_thresholds_uses_out_of_fold_scores(fitted_pipeline):
    X, y = make_frame()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    res = eval_thresholds(clone(fitted_pipeline), X, y, cv)

    assert list(res.columns) == ["threshold", "precision", "recall", "f1"]
    assert res["f1"].is_monotonic_decreasing  # sorted best-first
    assert res["threshold"].between(0.1, 0.9).all()


class _FakeDataset:
    def __init__(self, features, targets):
        self.data = type("D", (), {"features": features, "targets": targets})()


def test_load_data_selects_contract_columns(monkeypatch):
    X, y = make_frame(10)
    features = X[FEATURE_COLUMNS[::-1]].assign(UDI=range(10))  # reordered + extra
    targets = pd.DataFrame({"Machine failure": y, "TWF": 0})
    monkeypatch.setattr(data, "fetch_ucirepo", lambda id: _FakeDataset(features, targets))

    X_out, y_out = data.load_data()

    assert list(X_out.columns) == FEATURE_COLUMNS
    assert y_out.tolist() == y.tolist()


def test_load_data_reports_schema_change(monkeypatch):
    X, y = make_frame(10)
    renamed = X.rename(columns={"Torque": "Torque [Nm]"})
    monkeypatch.setattr(
        data,
        "fetch_ucirepo",
        lambda id: _FakeDataset(renamed, pd.DataFrame({"Machine failure": y})),
    )

    with pytest.raises(RuntimeError, match="missing feature columns"):
        data.load_data()


def test_load_data_network_error_is_actionable(monkeypatch):
    def boom(id):
        raise ConnectionError("offline")

    monkeypatch.setattr(data, "fetch_ucirepo", boom)

    with pytest.raises(RuntimeError, match="internet access"):
        data.load_data()
