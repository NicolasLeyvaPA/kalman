"""
Apply the SQL schema. Useful when not running the Postgres container with
the entrypoint init script.
"""
import asyncio
import pathlib
import sys

from sqlalchemy import text

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from data.database import engine  # noqa: E402


async def run() -> None:
    sql_path = ROOT / "scripts" / "init_db.sql"
    sql = sql_path.read_text()
    async with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            await conn.execute(text(stmt))
    print("schema applied")


if __name__ == "__main__":
    asyncio.run(run())
