"""H2-lite: disk cache for LLM prompt/response pairs + audit log."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]


def prompt_key(prompt: str, *, model: str = "") -> str:
    raw = (str(model) + "\n" + str(prompt)).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def cache_path(run_dir: PathLike, key: str) -> Path:
    return Path(run_dir) / "llm_cache" / f"{key}.json"


def get_cached_response(
    run_dir: PathLike,
    prompt: str,
    *,
    model: str = "",
) -> Optional[str]:
    path = cache_path(run_dir, prompt_key(prompt, model=model))
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("response")


def put_cached_response(
    run_dir: PathLike,
    prompt: str,
    response: str,
    *,
    model: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    key = prompt_key(prompt, model=model)
    path = cache_path(run_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "model": model,
        "prompt": prompt,
        "response": response,
        "extra": extra or {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Append compact audit line
    audit = Path(run_dir) / "llm_prompt_response.jsonl"
    with audit.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": payload["ts"],
                    "key": key,
                    "model": model,
                    "prompt_chars": len(prompt),
                    "response_chars": len(response),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return path
