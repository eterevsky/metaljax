"""Ledger-cell parsing in scripts/release/model_gate_report.py.

Annotated cells ("**237.3** ᴳ", "**16.81** ᶠ (13 GB)") used to parse as
None, mis-labeling ledgered rows "NEW (was blocked)" and silently skipping
the >10% regression check on them (2026-08-18 re-gate, rows 1/4/7/11/14/18/19).
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent
           / "scripts" / "release" / "model_gate_report.py")
_spec = importlib.util.spec_from_file_location("model_gate_report", _SCRIPT)
model_gate_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_gate_report)


@pytest.mark.parametrize("cell,expected", [
    # the four annotated shapes from the 2026-08-18 re-gate
    ("**237.3** ᴳ", 237.3),
    ("**21.9** ᶜ (spread 21.9–24.2)", 21.9),
    ("**16.81** ᶠ (13 GB)", 16.81),
    ("**459.2** ᶠ (25 GB)", 459.2),
    # plain and previously-working shapes stay parsed
    ("94.3", 94.3),
    ("**94.3**", 94.3),
    ("92.9 ᴾ²⁷", 92.9),
    ("1865", 1865.0),
    # absent cells stay absent
    ("✗", None),
    ("✗ download ᴳ (93.4 GB streams)", None),
    ("✗ declined ᴳ", None),
    ("—", None),
    ("-", None),
    ("n/a", None),
    ("", None),
    # annotation-only cell (row 15): no number to extract
    ("✅ **coherent** ᵛ (10/10 deterministic; ms/tok not re-measured)", None),
])
def test_clean_cell(cell, expected):
    assert model_gate_report.clean_cell(cell) == expected
