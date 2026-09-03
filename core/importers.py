"""Lightweight Portfolio CSV/XLSX import: read the file, auto-match common
column headers, and fall back to a minimal manual mapping only when a
required column cannot be identified. No PDF/OCR/broker integration.

The imported result has the exact same shape (Ticker, Quantity, Average
Cost) as the manual holdings table, so it can be written straight into the
canonical holdings state with no separate preview data model."""

from __future__ import annotations

import io

import pandas as pd

REQUIRED_FIELDS = ("ticker", "quantity")

COLUMN_ALIASES: dict[str, set[str]] = {
    "ticker": {"ticker", "symbol", "stock", "代码", "股票代码", "证券代码"},
    "quantity": {"quantity", "shares", "qty", "units", "股数", "数量", "持仓数量"},
    "average_cost": {"average cost", "avg cost", "average_cost", "cost basis", "book cost", "acb", "成本", "平均成本", "持仓成本"},
}

TEMPLATE_COLUMNS = ["Ticker", "Quantity", "Average Cost"]


def read_portfolio_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read raw CSV or XLSX bytes into a DataFrame. Caller must catch parse errors."""
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    return pd.read_csv(io.BytesIO(file_bytes))


def auto_map_columns(raw: pd.DataFrame) -> dict[str, str | None]:
    """Best-effort case-insensitive header match; None means unresolved."""
    normalized = {str(col).strip().lower(): col for col in raw.columns}
    mapping: dict[str, str | None] = {}
    for field, aliases in COLUMN_ALIASES.items():
        match = next((normalized[key] for key in normalized if key in aliases), None)
        mapping[field] = match
    return mapping


def missing_required_fields(mapping: dict[str, str | None]) -> list[str]:
    return [field for field in REQUIRED_FIELDS if not mapping.get(field)]


def build_preview(raw: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Apply a resolved column mapping into the standard Ticker/Quantity/
    Average Cost columns. Rows without a ticker or a numeric quantity are
    dropped; Average Cost falls back to blank rather than blocking import."""
    if missing_required_fields(mapping):
        raise ValueError("Ticker and Quantity columns are required")
    out = pd.DataFrame()
    out["Ticker"] = raw[mapping["ticker"]].astype(str).str.strip().str.upper()
    out["Quantity"] = pd.to_numeric(raw[mapping["quantity"]], errors="coerce")
    out["Average Cost"] = pd.to_numeric(raw[mapping["average_cost"]], errors="coerce") if mapping.get("average_cost") else None
    out = out[out["Ticker"].str.len() > 0]
    out = out.dropna(subset=["Quantity"])
    return out.reset_index(drop=True)


def template_csv_bytes() -> bytes:
    frame = pd.DataFrame([
        {"Ticker": "VOO", "Quantity": 10, "Average Cost": 480.0},
        {"Ticker": "AAPL", "Quantity": 15, "Average Cost": 210.0},
    ], columns=TEMPLATE_COLUMNS)
    return frame.to_csv(index=False).encode("utf-8-sig")
