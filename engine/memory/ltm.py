import sqlite3, json, contextlib, pathlib

DB = pathlib.Path("configs/ltm.sqlite")
SCHEMA = """CREATE TABLE IF NOT EXISTS memo (
    id        TEXT PRIMARY KEY,
    data      TEXT,
    weight    REAL,
    valence   REAL,
    updated   REAL
);"""


def _cx():
    cx = sqlite3.connect(DB)
    cx.execute(SCHEMA)
    return cx


@contextlib.contextmanager
def cx():
    conn = _cx()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert(id: str, blob: dict):
    with cx() as c:
        c.execute(
            "REPLACE INTO memo VALUES(?,?,?,?,strftime('%s','now'))",
            (id, json.dumps(blob), blob["weight"], blob["valence"]),
        )


def topk(limit: int = 100):
    with cx() as c:
        return [
            json.loads(r[0])
            for r in c.execute(
                "SELECT data FROM memo ORDER BY weight DESC LIMIT ?", (limit,)
            )
        ]
