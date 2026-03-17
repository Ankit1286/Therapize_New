"""
Database migration runner.
Runs all SQL migration files in order.

Usage:
    python scripts/migrate.py
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from src.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_migrations() -> None:
    settings = get_settings()
    from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
    raw = str(settings.database_url).replace("postgresql+asyncpg://", "postgresql://")
    parsed = urlparse(raw)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items() if k != "channel_binding"}
    dsn = urlunparse(parsed._replace(query=urlencode(params)))

    conn = await asyncpg.connect(dsn)
    try:
        # Create migrations tracking table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                filename VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Find all migration files
        migrations_dir = Path(__file__).parent.parent / "src" / "storage" / "migrations"
        migration_files = sorted(migrations_dir.glob("*.sql"))

        for migration_file in migration_files:
            # Check if already applied
            already_applied = await conn.fetchval(
                "SELECT 1 FROM _migrations WHERE filename = $1",
                migration_file.name
            )
            if already_applied:
                logger.info("Skipping %s (already applied)", migration_file.name)
                continue

            logger.info("Applying %s...", migration_file.name)
            sql = migration_file.read_text()

            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO _migrations (filename) VALUES ($1)",
                    migration_file.name
                )

            logger.info("Applied %s successfully", migration_file.name)

        logger.info("All migrations applied.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migrations())
