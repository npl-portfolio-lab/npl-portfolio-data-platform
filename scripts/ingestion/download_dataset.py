from pathlib import Path
from npl_portfolio.core.paths import PROJECT_ROOT

from npl_portfolio.ingestion.home_credit_downloader import HomeCreditDownloader


def main() -> None:
    project_root = PROJECT_ROOT

    destination = project_root / "data" / "raw" / "home_credit"

    downloader = HomeCreditDownloader(
        destination=destination,
    )

    downloader.run()


if __name__ == "__main__":
    main()
