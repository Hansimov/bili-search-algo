from __future__ import annotations

import sqlite3
import time

from pathlib import Path


class GroupCursor:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS processed ("
            "doc_key TEXT PRIMARY KEY, "
            "content_hash TEXT NOT NULL, "
            "updated_at INTEGER NOT NULL)"
        )
        self.connection.commit()

    def is_unchanged(self, doc_key: str, content_hash: str) -> bool:
        row = self.connection.execute(
            "SELECT content_hash FROM processed WHERE doc_key = ?",
            (doc_key,),
        ).fetchone()
        return bool(row and row[0] == content_hash)

    def mark_many(self, items: list[tuple[str, str]]) -> None:
        if not items:
            return
        updated_at = int(time.time())
        self.connection.executemany(
            "INSERT INTO processed(doc_key, content_hash, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(doc_key) DO UPDATE SET "
            "content_hash = excluded.content_hash, updated_at = excluded.updated_at",
            [(doc_key, content_hash, updated_at) for doc_key, content_hash in items],
        )
        self.connection.commit()

    def count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM processed").fetchone()
        return int(row[0] if row else 0)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "GroupCursor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
