from __future__ import annotations

import pandas as pd
import pytest

from core.importers import (
    auto_map_columns, build_preview, missing_required_fields, read_portfolio_file, template_csv_bytes,
)
from core.models import HoldingInput


def test_auto_map_recognizes_common_headers():
    raw = pd.DataFrame({"Symbol": ["VOO"], "Shares": [10], "Book Cost": [480.0]})
    mapping = auto_map_columns(raw)
    assert mapping["ticker"] == "Symbol"
    assert mapping["quantity"] == "Shares"
    assert mapping["average_cost"] == "Book Cost"
    assert not missing_required_fields(mapping)


def test_auto_map_recognizes_chinese_headers():
    raw = pd.DataFrame({"代码": ["AAPL"], "数量": [5], "成本": [200.0]})
    mapping = auto_map_columns(raw)
    assert not missing_required_fields(mapping)


def test_auto_map_reports_missing_required_columns():
    raw = pd.DataFrame({"Name": ["VOO"], "Value": [10]})
    mapping = auto_map_columns(raw)
    assert set(missing_required_fields(mapping)) == {"ticker", "quantity"}


def test_build_preview_applies_manual_mapping_for_unmatched_columns():
    raw = pd.DataFrame({"Name": ["voo", "aapl"], "Value": [10, 5]})
    mapping = auto_map_columns(raw)
    mapping["ticker"], mapping["quantity"] = "Name", "Value"
    preview = build_preview(raw, mapping)
    assert list(preview.columns) == ["Ticker", "Quantity", "Average Cost"]
    assert list(preview["Ticker"]) == ["VOO", "AAPL"]
    assert list(preview["Quantity"]) == [10, 5]


def test_build_preview_reads_average_cost_when_present():
    raw = pd.DataFrame({"Ticker": ["VOO", "AAPL"], "Quantity": [1, 2], "Average Cost": [480.0, 210.0]})
    mapping = auto_map_columns(raw)
    preview = build_preview(raw, mapping)
    assert list(preview["Average Cost"]) == [480.0, 210.0]


def test_build_preview_defaults_average_cost_to_blank_when_column_missing():
    raw = pd.DataFrame({"Ticker": ["VOO"], "Quantity": [1]})
    mapping = auto_map_columns(raw)
    preview = build_preview(raw, mapping)
    assert preview.iloc[0]["Average Cost"] is None or pd.isna(preview.iloc[0]["Average Cost"])


def test_build_preview_drops_rows_without_ticker_or_quantity():
    raw = pd.DataFrame({"Ticker": ["VOO", ""], "Quantity": [1, None]})
    mapping = auto_map_columns(raw)
    preview = build_preview(raw, mapping)
    assert len(preview) == 1


def test_build_preview_raises_when_required_fields_unresolved():
    raw = pd.DataFrame({"Name": ["VOO"]})
    mapping = auto_map_columns(raw)
    with pytest.raises(ValueError):
        build_preview(raw, mapping)


def test_read_portfolio_file_csv_roundtrip():
    csv_bytes = b"Ticker,Quantity,Average Cost\nVOO,10,480\n"
    raw = read_portfolio_file(csv_bytes, "holdings.csv")
    mapping = auto_map_columns(raw)
    preview = build_preview(raw, mapping)
    assert preview.iloc[0]["Ticker"] == "VOO"
    assert preview.iloc[0]["Quantity"] == 10


def test_read_portfolio_file_xlsx_roundtrip(tmp_path):
    path = tmp_path / "holdings.xlsx"
    pd.DataFrame([{"Ticker": "AAPL", "Quantity": 3, "Average Cost": 210.0}]).to_excel(path, index=False)
    raw = read_portfolio_file(path.read_bytes(), "holdings.xlsx")
    mapping = auto_map_columns(raw)
    preview = build_preview(raw, mapping)
    assert preview.iloc[0]["Ticker"] == "AAPL"
    assert preview.iloc[0]["Quantity"] == 3


def test_template_csv_has_expected_columns_and_parses_back():
    raw = read_portfolio_file(template_csv_bytes(), "template.csv")
    assert list(raw.columns) == ["Ticker", "Quantity", "Average Cost"]
    mapping = auto_map_columns(raw)
    assert not missing_required_fields(mapping)


def test_imported_preview_has_same_shape_as_manual_holdings_table():
    """The whole point of the simplified importer: its output must be a
    drop-in replacement for the manual holdings table, with no separate
    preview schema (Ticker/Quantity/Average Cost only, nothing else)."""
    raw = pd.DataFrame({"Symbol": ["VOO"], "Shares": [10]})
    mapping = auto_map_columns(raw)
    preview = build_preview(raw, mapping)
    assert list(preview.columns) == ["Ticker", "Quantity", "Average Cost"]


def test_imported_preview_feeds_holding_input_without_error():
    raw = pd.DataFrame({"Ticker": ["VOO"], "Quantity": [10]})
    mapping = auto_map_columns(raw)
    preview = build_preview(raw, mapping)
    row = preview.iloc[0]
    holding = HoldingInput(row["Ticker"], float(row["Quantity"])).normalized()
    assert holding.ticker == "VOO"
    assert holding.quantity == 10
