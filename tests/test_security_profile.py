from __future__ import annotations

import ast
from pathlib import Path

from core.security_profile import resolve_security_profile

MODULE_PATH = Path(__file__).resolve().parents[1] / "core" / "security_profile.py"


def test_stock_profile_resolves_with_core_fields():
    profile = resolve_security_profile("NVDA")
    assert profile is not None
    assert profile.kind_label == "股票（Stock）"
    assert "英伟达" in profile.title and "NVIDIA" in profile.title
    assert profile.category_label == "板块"
    assert "信息技术" in profile.category_value
    assert profile.subcategory_label == "行业"
    assert profile.subcategory_value == "半导体"
    assert profile.description


def test_etf_profile_resolves_with_core_fields():
    profile = resolve_security_profile("sgov")  # case-insensitive, mirrors ticker normalization elsewhere
    assert profile is not None
    assert profile.ticker == "SGOV"
    assert profile.kind_label == "ETF"
    assert profile.category_label == "资产类别"
    assert profile.category_value == "固定收益"
    assert profile.subcategory_label == "基金类别"
    assert "国债" in profile.description


def test_unknown_ticker_fails_closed_without_guessing():
    assert resolve_security_profile("ZZZZ") is None
    assert resolve_security_profile("") is None
    assert resolve_security_profile(None) is None


def test_finn_profile_matches_the_resolved_canadian_identity():
    """Regression for the P0 ticker-collision bug: the Security Profile
    for FINN must describe the same Canadian ETF that core.ticker_resolution
    resolves to for valuation (FINN.NE), never the unrelated bare-FINN
    U.S. OTC stock (First National of Nebraska) that caused it."""
    from core.ticker_resolution import CANADIAN_REFERENCE

    identity = CANADIAN_REFERENCE["FINN"]
    assert identity.resolved_ticker == "FINN.NE"
    profile = resolve_security_profile("FINN")
    assert profile is not None
    assert "Fidelity Global Innovators" in profile.title
    assert "First National" not in profile.title


def test_every_canadian_reference_ticker_has_one_canonical_security_profile():
    """One ticker -> one canonical security identity: every ticker the
    market-data resolver treats as a known Canadian security must also
    have a Security Profile entry, not a second, independently-drifting
    identity system."""
    from core.ticker_resolution import CANADIAN_REFERENCE

    for ticker in CANADIAN_REFERENCE:
        assert resolve_security_profile(ticker) is not None, f"{ticker} missing from Security Profile"


def test_security_profile_module_has_no_network_or_llm_dependency():
    """Layer 1 must stay a pure local lookup: no import of core.ai, any LLM
    SDK, or any network/HTTP library -- clicking a ticker can never trigger
    a Gemini/Claude/OpenAI call."""
    tree = ast.parse(MODULE_PATH.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert modules <= {"__future__", "dataclasses"}
