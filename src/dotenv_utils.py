from __future__ import annotations

from pathlib import Path
import os


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_dotenv(path: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


_DOTENV = _parse_dotenv(_repo_root() / ".env")


def _require_env(name: str) -> str:
    value = os.getenv(name, _DOTENV.get(name))
    if value is None:
        raise RuntimeError(
            f"Missing required environment key '{name}' in .env or OS environment."
        )
    return value


def get_env_str(name: str) -> str:
    return _require_env(name)


def get_env_int(name: str) -> int:
    return int(_require_env(name))


def get_env_float(name: str) -> float:
    return float(_require_env(name))


def get_env_optional_float(name: str) -> float | None:
    raw = _require_env(name).strip()
    if raw == "" or raw.lower() == "none":
        return None
    return float(raw)

