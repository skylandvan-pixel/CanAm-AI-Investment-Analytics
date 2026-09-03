from __future__ import annotations

import hashlib
import hmac
import os


def configured_beta_codes() -> tuple[str, ...]:
    return tuple(code.strip() for code in os.getenv("CANAM_BETA_CODES", "").split(",") if code.strip())


def validate_beta_code(candidate: str, codes: tuple[str, ...] | None = None) -> bool:
    allowed = codes if codes is not None else configured_beta_codes()
    candidate_digest = hashlib.sha256(candidate.strip().encode()).digest()
    return any(hmac.compare_digest(candidate_digest, hashlib.sha256(code.encode()).digest()) for code in allowed)
