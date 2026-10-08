from pathlib import Path
from npl_portfolio.core.paths import PROJECT_ROOT

from npl_portfolio.features.application_features import (
    ApplicationFeatureBuilder,
)


ROOT = PROJECT_ROOT

SOURCE = ROOT / "data" / "processed" / "home_credit"
OUTPUT = ROOT / "data" / "interim" / "features"

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


def build_application_checkpoint(
    source_name: str,
    output_name: str,
) -> None:
    source_path = SOURCE / source_name
    output_path = OUTPUT / output_name

    print()
    print("=" * 80)
    print(f"GENERANDO {output_name}")
    print("=" * 80)

    builder = ApplicationFeatureBuilder(
        parquet_path=source_path,
    )

    builder.build_to_parquet(
        output_path=output_path,
        overwrite=True,
    )


def main() -> None:
    build_application_checkpoint(
        source_name="application_train.parquet",
        output_name="application_features_train.parquet",
    )

    build_application_checkpoint(
        source_name="application_test.parquet",
        output_name="application_features_test.parquet",
    )

    print()
    print("=" * 80)
    print("APPLICATION FEATURES FINALIZADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
