"""
Populate the known_addresses table from the curated list.
"""
import asyncio
import pathlib
import sys

from sqlalchemy.dialects.postgresql import insert

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from data.database import db_session  # noqa: E402
from data.known_exchanges import KNOWN_ADDRESSES  # noqa: E402
from data.models import KnownAddress  # noqa: E402


async def run() -> None:
    async with db_session() as session:
        for addr, meta in KNOWN_ADDRESSES.items():
            stmt = insert(KnownAddress).values(
                address=addr.lower(),
                label=meta["label"],
                category=meta["category"],
                chain="polygon",
            ).on_conflict_do_update(
                index_elements=[KnownAddress.address],
                set_={"label": meta["label"], "category": meta["category"]},
            )
            await session.execute(stmt)
    print(f"seeded {len(KNOWN_ADDRESSES)} known addresses")


if __name__ == "__main__":
    asyncio.run(run())
