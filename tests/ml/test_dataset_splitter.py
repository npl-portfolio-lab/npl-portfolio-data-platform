import pandas as pd

from npl_portfolio.ml.dataset_splitter import (
    StratifiedDatasetSplitter,
)


def test_stratified_dataset_splitter() -> None:
    clients = pd.DataFrame(
        {
            "SK_ID_CURR": range(1000, 2000),
            "TARGET": (
                [0] * 900
                + [1] * 100
            ),
        }
    )

    splitter = StratifiedDatasetSplitter(
        validation_size=0.20,
        random_state=42,
    )

    result = splitter.split(clients)

    assert len(result.train_ids) == 800
    assert len(result.validation_ids) == 200

    train_ids = set(result.train_ids)
    validation_ids = set(result.validation_ids)

    # Ningún cliente puede aparecer en ambos conjuntos.
    assert train_ids.isdisjoint(validation_ids)

    # No podemos perder clientes durante el split.
    assert len(
        train_ids | validation_ids
    ) == 1000

    target_by_id = clients.set_index(
        "SK_ID_CURR"
    )["TARGET"]

    train_target = target_by_id.loc[
        list(train_ids)
    ]

    validation_target = target_by_id.loc[
        list(validation_ids)
    ]

    assert train_target.mean() == 0.10
    assert validation_target.mean() == 0.10