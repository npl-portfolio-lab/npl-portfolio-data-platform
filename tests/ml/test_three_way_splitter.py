import numpy as np
import pandas as pd
import pytest

from npl_portfolio.ml.three_way_splitter import StratifiedThreeWaySplitter


@pytest.fixture
def clients():
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(10000),
        "TARGET": np.array([0] * 9000 + [1] * 1000),
    })


def test_split_sizes(clients):
    split = StratifiedThreeWaySplitter().split(clients)

    assert len(split.train_ids) == 6400
    assert len(split.validation_ids) == 1600
    assert len(split.test_ids) == 2000


def test_no_overlapping_clients(clients):
    split = StratifiedThreeWaySplitter().split(clients)

    train = set(split.train_ids)
    validation = set(split.validation_ids)
    test = set(split.test_ids)

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)

    assert len(train | validation | test) == len(clients)


def test_reproducibility(clients):
    splitter = StratifiedThreeWaySplitter(random_state=42)

    first = splitter.split(clients)
    second = splitter.split(clients)

    np.testing.assert_array_equal(first.train_ids, second.train_ids)
    np.testing.assert_array_equal(
        first.validation_ids, second.validation_ids
    )
    np.testing.assert_array_equal(first.test_ids, second.test_ids)


def test_stratification(clients):
    split = StratifiedThreeWaySplitter().split(clients)

    labels = clients.set_index("SK_ID_CURR")["TARGET"]

    for ids in (
        split.train_ids,
        split.validation_ids,
        split.test_ids,
    ):
        prevalence = labels.loc[ids].mean()
        assert abs(prevalence - 0.10) < 0.01


def test_reject_duplicate_ids(clients):
    invalid = clients.copy()
    invalid.loc[1, "SK_ID_CURR"] = invalid.loc[0, "SK_ID_CURR"]

    with pytest.raises(ValueError, match="duplicados"):
        StratifiedThreeWaySplitter().split(invalid)


def test_reject_invalid_target(clients):
    invalid = clients.copy()
    invalid.loc[0, "TARGET"] = 2

    with pytest.raises(ValueError, match="TARGET"):
        StratifiedThreeWaySplitter().split(invalid)


def test_reject_invalid_proportions():
    with pytest.raises(ValueError, match="sumar 1"):
        StratifiedThreeWaySplitter(
            train_size=0.60,
            validation_size=0.20,
            test_size=0.30,
        )
