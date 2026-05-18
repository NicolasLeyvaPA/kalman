"""
Backfill: run a one-shot ingest pass without starting the full scheduler.
Useful for populating data into a fresh database for offline analysis.
"""
import asyncio
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from services.trade_ingester import ingest_once  # noqa: E402
from services.scoring_engine import score_recent  # noqa: E402
from services.cluster_detector import run_once as cluster_once  # noqa: E402
from services.resolution_tracker import run_once as resolution_once  # noqa: E402


async def main() -> None:
    print("ingesting...")
    print(await ingest_once())
    print("attributing resolutions...")
    print(await resolution_once())
    print("scoring...")
    print({"scored": await score_recent()})
    print("clustering...")
    print(await cluster_once())


if __name__ == "__main__":
    asyncio.run(main())
