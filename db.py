"""Database abstraction — SQLite (default) or PostgreSQL via DATABASE_URL."""

import os
import sqlite3

DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    import psycopg2
    import psycopg2.extras


class PgRowWrapper:
    """Make psycopg2 rows behave like sqlite3.Row (dict-like access)."""
    def __init__(self, row, description):
        self._data = {}
        if row and description:
            for i, col in enumerate(description):
                self._data[col.name] = row[i]

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self._data.values())[key]
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return repr(self._data)


class PgConnection:
    """Wrapper around psycopg2 connection to match sqlite3 interface."""
    def __init__(self, dsn):
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False

    def execute(self, sql, params=None):
        # Convert SQLite ? placeholders to Postgres %s
        sql = sql.replace("?", "%s")
        # Convert SQLite-specific syntax
        sql = sql.replace("INSERT OR IGNORE", "INSERT ON CONFLICT DO NOTHING --")
        # Handle PRAGMA (skip for Postgres)
        if sql.strip().upper().startswith("PRAGMA"):
            return PgCursor([], [])
        cursor = self._conn.cursor()
        cursor.execute(sql, params or [])
        return PgCursor(cursor, cursor.description)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class PgCursor:
    """Wrapper to make psycopg2 cursor results behave like sqlite3."""
    def __init__(self, cursor, description):
        self._cursor = cursor
        self._description = description

    def fetchall(self):
        if not self._cursor or not self._description:
            return []
        rows = self._cursor.fetchall() if hasattr(self._cursor, 'fetchall') else self._cursor
        return [PgRowWrapper(r, self._description) for r in rows]

    def fetchone(self):
        if not self._cursor or not self._description:
            return None
        row = self._cursor.fetchone() if hasattr(self._cursor, 'fetchone') else None
        return PgRowWrapper(row, self._description) if row else None


def get_db():
    """Get a database connection — Postgres if DATABASE_URL is set, else SQLite."""
    if DATABASE_URL:
        return PgConnection(DATABASE_URL)
    else:
        base_dir = os.path.dirname(__file__)
        db_path = os.path.join(base_dir, "library.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn


def init_db_postgres(conn):
    """Create tables for PostgreSQL."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id TEXT PRIMARY KEY,
            original_name TEXT,
            mp3_path TEXT,
            duration REAL,
            transcription TEXT,
            transcription_en TEXT,
            graph_json TEXT,
            created_at TEXT,
            insights_json TEXT,
            updated_at TEXT,
            tags TEXT,
            notebook TEXT,
            starred INTEGER DEFAULT 0,
            trashed_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notebooks (
            name TEXT PRIMARY KEY,
            created_at TEXT,
            stack TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_searches (
            id TEXT PRIMARY KEY,
            name TEXT,
            query TEXT,
            filters TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attachments (
            id TEXT PRIMARY KEY,
            file_id TEXT,
            filename TEXT,
            mime_type TEXT,
            size INTEGER,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT,
            file_hash TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS note_versions (
            id TEXT PRIMARY KEY,
            file_id TEXT,
            title TEXT,
            transcription TEXT,
            transcription_en TEXT,
            created_at TEXT,
            source TEXT DEFAULT 'autosave'
        )
    """)
    conn.commit()


def is_postgres():
    """Check if using PostgreSQL."""
    return bool(DATABASE_URL)
