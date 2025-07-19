import yaml, pathlib, time, functools
from engine.config import IDENTITY_PATH as _ID_PATH

_last_mtime = 0


@functools.lru_cache(1)
def _load() -> dict:
    return yaml.safe_load(_ID_PATH.read_text()) if _ID_PATH.exists() else {}


def get(key, default=None):
    return _load().get(key, default)


def refresh_if_changed():
    global _last_mtime
    try:
        m = _ID_PATH.stat().st_mtime
        if m != _last_mtime:
            _load.cache_clear()
            _last_mtime = m
    except FileNotFoundError:
        pass
