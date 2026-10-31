"""xAI (Grok) API keys for CARD.

Loads keys from ``xai_keys.json`` (or ``XAI_KEYS_PATH``), formatted as
``{"keys": ["xai-...", ...]}``. Falls back to ``XAI_API_KEY``.
Never commit ``xai_keys.json``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

COOLDOWN_SECONDS = 60
XAI_BASE_URL = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
DEFAULT_XAI_MIN_REQUEST_INTERVAL = float(os.environ.get("XAI_MIN_REQUEST_INTERVAL", "1.0"))
DEFAULT_XAI_REQUEST_TIMEOUT = float(os.environ.get("XAI_TIMEOUT_SECONDS", "600"))
DEFAULT_XAI_MAX_ATTEMPTS = int(os.environ.get("XAI_MAX_ATTEMPTS", "8"))
DEFAULT_XAI_TRANSIENT_BACKOFF = float(os.environ.get("XAI_TRANSIENT_BACKOFF", "2.0"))

_PLACEHOLDER_PREFIXES = ("xai_REPLACE", "xai-REPLACE", "xai_your_", "YOUR_KEY")
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XAI_KEYS_PATH = os.path.join(_REPO_ROOT, "xai_keys.json")


def is_xai_model(model_name: str) -> bool:
    name = (model_name or "").strip().lower()
    return name.startswith("grok-") or name.startswith("xai/")


def normalize_xai_model(model_name: str) -> str:
    name = (model_name or "").strip()
    if name.lower().startswith("xai/"):
        return name[4:]
    return name


def _xai_http_client():
    proxy = (
        os.environ.get("XAI_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or ""
    ).strip()
    if not proxy:
        return None
    import httpx
    return httpx.Client(proxy=proxy, timeout=DEFAULT_XAI_REQUEST_TIMEOUT)


def _is_placeholder_key(key: str) -> bool:
    k = (key or "").strip()
    if not k:
        return True
    return any(k.startswith(p) for p in _PLACEHOLDER_PREFIXES)


def _is_request_too_large(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status == 413:
        return True
    return "request too large" in text or "please reduce your message size" in text


def _is_rate_limit_error(exc: Exception) -> bool:
    if _is_request_too_large(exc):
        return False
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text


def _is_transient_error(exc: Exception) -> bool:
    if _is_rate_limit_error(exc):
        return True
    status = getattr(exc, "status_code", None)
    if status in (408, 409, 425, 500, 502, 503, 504):
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    needles = (
        "timeout", "timed out", "connecterror", "connection reset",
        "connection aborted", "connection refused", "temporarily unavailable",
        "handshake", "network", "empty completion", "apiconnectionerror",
    )
    if any(n in name for n in ("timeout", "connect", "network", "unavailable")):
        return True
    return any(n in text for n in needles)


def _is_non_retryable_client_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if _is_request_too_large(exc):
        return True
    if status in (401, 403):
        return True
    if status == 404 or "model_not_found" in text or "does not exist" in text:
        return True
    if status == 400 and "model" in text:
        return True
    return False


def _message_text(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
        return str(content or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _response_headers(response: Any) -> dict:
    raw = getattr(response, "_response", None) or getattr(response, "response", None)
    if raw is None:
        return {}
    headers = getattr(raw, "headers", None)
    if headers is None:
        return {}
    try:
        return {str(k).lower(): v for k, v in dict(headers).items()}
    except Exception:  # noqa: BLE001
        return {}


class XAIRateLimitPacer:
    def __init__(self, *, min_interval_seconds: float = DEFAULT_XAI_MIN_REQUEST_INTERVAL) -> None:
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._last_request_at = 0.0
        self._hold_until = 0.0

    def wait_turn(self) -> None:
        now = time.time()
        target = max(self._last_request_at + self.min_interval_seconds, self._hold_until)
        if now < target:
            time.sleep(target - now)
        self._last_request_at = time.time()

    def observe_response(self, response: Any) -> None:
        headers = _response_headers(response)
        retry_after = headers.get("retry-after") or headers.get("retry_after")
        if retry_after is not None:
            try:
                self._hold_until = max(self._hold_until, time.time() + float(retry_after))
            except (TypeError, ValueError):
                pass


_xai_pacer_singleton: Optional[XAIRateLimitPacer] = None


def get_xai_pacer() -> XAIRateLimitPacer:
    global _xai_pacer_singleton
    if _xai_pacer_singleton is None:
        _xai_pacer_singleton = XAIRateLimitPacer()
    return _xai_pacer_singleton


def resolve_xai_keys_path() -> str:
    return os.environ.get("XAI_KEYS_PATH", DEFAULT_XAI_KEYS_PATH)


def load_xai_keys(keys_path: Optional[str] = None) -> List[str]:
    path = keys_path or resolve_xai_keys_path()
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("keys", [])
    if not isinstance(raw, list):
        raise ValueError(f"{path}: 'keys' must be a JSON array")
    return [str(k).strip() for k in raw if k and not _is_placeholder_key(str(k))]


def resolve_single_xai_api_key() -> Optional[str]:
    key = (os.environ.get("XAI_API_KEY") or "").strip()
    return key or None


class XAIKeyManager:
    def __init__(self, keys_path: Optional[str] = None, *, keys: Optional[List[str]] = None):
        self.keys_path = keys_path or resolve_xai_keys_path()
        if keys is not None:
            self.keys = [k for k in keys if k and not _is_placeholder_key(k)]
        else:
            self.keys = load_xai_keys(self.keys_path)
        if not self.keys:
            single_key = resolve_single_xai_api_key()
            self.keys = [single_key] if single_key else []
        if not self.keys:
            raise RuntimeError(
                f"No xAI API keys available. Set XAI_API_KEY or create "
                f'{self.keys_path} as {{"keys": ["xai-...", ...]}}'
            )
        self.index = 0
        self.cooldown_until = {key: 0.0 for key in self.keys}

    def _current_key(self) -> str:
        return self.keys[self.index]

    def _advance(self) -> None:
        self.index = (self.index + 1) % len(self.keys)

    def _mark_rate_limited(self, key: str) -> None:
        self.cooldown_until[key] = time.time() + COOLDOWN_SECONDS
        logger.warning("xAI key ...%s rate-limited; cooling down %ds", key[-6:], COOLDOWN_SECONDS)

    def _select_available_key(self) -> str:
        now = time.time()
        for _ in range(len(self.keys)):
            key = self._current_key()
            if self.cooldown_until[key] <= now:
                return key
            self._advance()
        return self._current_key()

    def get_client(self) -> Tuple[Any, str]:
        key = self._select_available_key()
        kwargs = dict(
            api_key=key,
            base_url=XAI_BASE_URL,
            max_retries=0,
            timeout=DEFAULT_XAI_REQUEST_TIMEOUT,
        )
        http_client = _xai_http_client()
        if http_client is not None:
            kwargs["http_client"] = http_client
        return OpenAI(**kwargs), key

    def chat_completion(self, **kwargs: Any) -> Any:
        if "model" in kwargs:
            kwargs = dict(kwargs)
            kwargs["model"] = normalize_xai_model(str(kwargs["model"]))
        pacer = get_xai_pacer()
        last_exception: Optional[Exception] = None
        max_attempts = max(DEFAULT_XAI_MAX_ATTEMPTS, len(self.keys) * 2)
        for attempt in range(1, max_attempts + 1):
            pacer.wait_turn()
            client, key = self.get_client()
            try:
                response = client.chat.completions.create(**kwargs)
                pacer.observe_response(response)
                if not _message_text(response):
                    last_exception = RuntimeError("xAI returned an empty completion.")
                    delay = min(30.0, DEFAULT_XAI_TRANSIENT_BACKOFF * (2.0 ** (attempt - 1)))
                    time.sleep(delay)
                    self._advance()
                    continue
                return response
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                if _is_non_retryable_client_error(exc):
                    raise
                if _is_rate_limit_error(exc):
                    self._mark_rate_limited(key)
                    self._advance()
                    continue
                if _is_transient_error(exc):
                    delay = min(60.0, DEFAULT_XAI_TRANSIENT_BACKOFF * (2.0 ** (attempt - 1)))
                    logger.warning(
                        "xAI transient error on key ...%s (attempt %d/%d): %s; retrying in %.1fs",
                        key[-6:], attempt, max_attempts, type(exc).__name__, delay,
                    )
                    time.sleep(delay)
                    self._advance()
                    continue
                raise
        raise RuntimeError(
            f"All {len(self.keys)} xAI API keys are rate-limited or failing "
            f"after {max_attempts} attempts. Last error: {last_exception}"
        )


_xai_manager_singleton: Optional[XAIKeyManager] = None


def get_xai_key_manager(*, keys_path: Optional[str] = None, force_reload: bool = False) -> XAIKeyManager:
    global _xai_manager_singleton
    if force_reload or _xai_manager_singleton is None:
        _xai_manager_singleton = XAIKeyManager(keys_path=keys_path)
    return _xai_manager_singleton
