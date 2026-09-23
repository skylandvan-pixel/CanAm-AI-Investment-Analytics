"""Optional server-side TypeSafe transport; no Streamlit/committee imports."""
from __future__ import annotations
import json
import math
import os
import time
import ssl
import socket
import http.client
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from pydantic import ValidationError
from core.jev_models import JevResponse, JevResult
from core.jev_questions import market_questions

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MAX_RESPONSE_BYTES = 65536

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

@dataclass
class HttpReply:
    status: int
    body: bytes
    retry_after: str | None = None

class ResponseReadFailure(Exception):
    """Carry status without embedding exception text or response contents."""
    def __init__(self, cause, status):
        super().__init__("response_read_failed")
        self.cause = cause
        self.status = status


def safe_diagnostic(exc=None, status=None):
    """Whitelist classifications only: never str/repr(exc), headers or body."""
    outer = exc
    if isinstance(exc, ResponseReadFailure):
        status = exc.status
        exc = exc.cause
    if isinstance(exc, URLError) and not isinstance(exc, HTTPError):
        exc = exc.reason
    cases = (
        (ssl.SSLCertVerificationError, "ssl", "certificate_verification_failed"),
        (ssl.SSLError, "ssl", "tls_error"),
        (socket.gaierror, "dns", "name_resolution_failed"),
        (TimeoutError, "timeout", "operation_timed_out"),
        (ConnectionRefusedError, "socket", "connection_refused"),
        (ConnectionResetError, "socket", "connection_reset"),
        (http.client.HTTPException, "http_protocol", "http_protocol_error"),
        (OSError, "socket", "socket_or_os_error"),
        (ValueError, "client", "invalid_transport_value"),
    )
    name, category, reason = "Exception", "unknown", "unclassified_transport_error"
    for kind, category_value, reason_value in cases:
        if isinstance(exc, kind):
            name, category, reason = kind.__name__, category_value, reason_value
            break
    if isinstance(outer, HTTPError):
        status = outer.code
    status = status if type(status) is int and 100 <= status <= 599 else None
    if exc is None and status is not None:
        name, category, reason = None, "http", "http_response_received"
    return {"exception_class": "URLError" if isinstance(outer, URLError) and not isinstance(outer, HTTPError) else name,
            "cause_class": name, "category": category, "sanitized_reason": reason,
            "http_status": status, "response_status": status}


def _post(payload: bytes, key: str, timeout: float) -> HttpReply:
    request = Request(ENDPOINT, data=payload, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            try:
                return HttpReply(response.status, response.read(MAX_RESPONSE_BYTES + 1))
            except Exception as exc:
                raise ResponseReadFailure(exc, response.status) from None
    except HTTPError as exc:
        try:
            return HttpReply(exc.code, b"", exc.headers.get("Retry-After"))
        finally:
            exc.close()

def evaluate(state_or_factory, *, transport=None, clock=time.monotonic,
             sleep=time.sleep, max_retries: int | None = None, diagnostics: bool = False) -> JevResult:
    """Explicit call only. Lazy state guarantees disabled means no market I/O.

    Only 429/529 retry, at most once. Never log raw errors/headers. CLI uses
    zero retries for ONE real request. Not connected to Streamlit reruns.
    """
    if os.getenv("JEV_ENABLED", "false").strip().lower() != "true":
        return JevResult(status="disabled")
    key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not key:
        return JevResult(status="unavailable", reason="missing_api_key")
    if not key.isascii() or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in key):
        return JevResult(status="unavailable", reason="invalid_api_key")
    try:
        timeout = float(os.getenv("JEV_TIMEOUT_SECONDS", "5"))
        total = float(os.getenv("JEV_TOTAL_TIMEOUT_SECONDS", "12"))
        retries = int(os.getenv("JEV_MAX_RETRIES", "1")) if max_retries is None else max_retries
        model = os.getenv("JEV_MODEL", "jev-latest").strip()
        if not model or not all(math.isfinite(x) and 0 < x <= 60 for x in (timeout, total)) or retries not in (0, 1):
            raise ValueError()
    except (ValueError, TypeError):
        return JevResult(status="unavailable", reason="invalid_configuration")
    try:
        state = state_or_factory() if callable(state_or_factory) else state_or_factory
        if not isinstance(state, dict) or not state.get("spy") or not state.get("data_timestamp"):
            raise ValueError()
        payload = json.dumps({"model": model, "state": state, "questions": market_questions()}, allow_nan=False).encode()
    except Exception:
        return JevResult(status="insufficient_data", reason="market_state_unavailable")
    start = clock()
    attempts = 0
    diagnostic = None
    def result(status, reason=None, response=None):
        return JevResult(status=status, reason=reason, response=response,
                         latency_ms=max(0., (clock() - start) * 1000), attempts=attempts,
                         diagnostic=diagnostic if diagnostics else None)
    post = transport or _post
    for attempt in range(retries + 1):
        remaining = total - (clock() - start)
        if remaining <= 0:
            return result("unavailable", "timeout")
        attempts += 1
        try:
            reply = post(payload, key, min(timeout, remaining))
        except Exception as exc:
            diagnostic = safe_diagnostic(exc) if diagnostics else None
            cause = exc.cause if isinstance(exc, ResponseReadFailure) else exc
            cause = cause.reason if isinstance(cause, URLError) else cause
            return result("unavailable", "timeout" if isinstance(cause, TimeoutError) else "network_error")
        if clock() - start >= total:
            return result("unavailable", "timeout")
        if not isinstance(reply, HttpReply):
            return result("invalid_response", "unexpected_transport_response")
        diagnostic = safe_diagnostic(status=reply.status) if diagnostics else None
        if reply.status in (429, 529) and attempt < retries:
            delay = 1.0
            if reply.retry_after:
                try:
                    requested_delay = float(reply.retry_after)
                    if not math.isfinite(requested_delay) or requested_delay < 0:
                        return result("unavailable", "retry_deferred")
                    delay = max(delay, requested_delay)
                except (ValueError, TypeError):
                    return result("unavailable", "retry_deferred")
            if not math.isfinite(delay) or delay >= total - (clock() - start):
                return result("unavailable", "retry_deferred")
            sleep(delay)
            continue
        if reply.status != 200:
            reason = {401: "authentication_failed", 422: "invalid_request", 429: "rate_limited", 529: "overloaded"}.get(reply.status, "http_error")
            return result("unavailable", reason)
        try:
            if len(reply.body) > MAX_RESPONSE_BYTES:
                raise ValueError()
            parsed = JevResponse.model_validate_json(reply.body)
            if key in parsed.model:
                raise ValueError()
        except (ValidationError, ValueError, TypeError):
            return result("invalid_response", "response_validation_failed")
        return result("ok", response=parsed)
    return result("unavailable", "retry_exhausted")
