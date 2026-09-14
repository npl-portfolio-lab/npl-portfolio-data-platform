from pathlib import Path

from npl_portfolio.ingestion.batch_csv_reader import BatchCSVReader
from npl_portfolio.profiling.csv_profiler import CSVDataProfiler
from npl_portfolio.profiling.dataset_profiler import DatasetProfiler


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    source_directory = project_root / "data" / "raw" / "home_credit"

    output_directory = project_root / "data" / "interim" / "profiling"

    batch_reader = BatchCSVReader(
        chunk_size=100_000,
    )

    csv_profiler = CSVDataProfiler(
        reader=batch_reader,
    )

    profiler = DatasetProfiler(
        source_directory=source_directory,
        output_directory=output_directory,
        csv_profiler=csv_profiler,
    )

    profiles = profiler.run()

    print()
    print("=" * 60)
    print("PERFILADO FINALIZADO")
    print(f"Tablas analizadas: {len(profiles)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
