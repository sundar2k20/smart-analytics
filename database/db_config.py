"""
Centralised PostgreSQL connection configuration.

Values are read from environment variables so credentials never live in
source. All variables have safe defaults except POSTGRES_PASSWORD, which
must be set explicitly (no default) to avoid accidental empty passwords.

A .env file at the repository root is auto-loaded if python-dotenv is
installed; existing OS environment variables always take precedence.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Walk up from this file to find the project root (where .env lives)
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.is_file():
        load_dotenv(_ENV_PATH, override=False)
except ImportError:
    # python-dotenv is optional; OS env vars still work without it
    pass


def get_db_config() -> dict:
    """
    Return a kwargs dict suitable for psycopg2.connect(**...) and
    SimpleConnectionPool(**...).
    """
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "postgres"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.environ["POSTGRES_PASSWORD"],
    }


def get_pool_config() -> dict:
    """
    Same as get_db_config but also includes pool sizing knobs.
    """
    return {
        "minconn": int(os.getenv("POSTGRES_POOL_MIN", "1")),
        "maxconn": int(os.getenv("POSTGRES_POOL_MAX", "10")),
        **get_db_config(),
    }
