"""Static web page tests (FR2.6, FR5.3): structure, data-testid coverage and safe rendering."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "web"
INTERACTIVE = {"a", "button", "input", "select", "textarea", "form"}
DASHBOARD_FIELDS = {
    "tokens_sent",
    "tokens_from_cache",
    "full_price_tokens",
    "spend_usd",
    "saved_usd",
    "storage_cost_usd",
    "hit_rate",
    "p50_latency_ms",
    "p99_latency_ms",
}


class ElementCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


def _elements(page: str) -> list[tuple[str, dict[str, str | None]]]:
    collector = ElementCollector()
    collector.feed((WEB / page).read_text(encoding="utf-8"))
    return collector.elements


def _test_ids(page: str) -> set[str]:
    return {attrs["data-testid"] for _, attrs in _elements(page) if attrs.get("data-testid")}


@pytest.mark.parametrize("page", ["index.html", "admin.html", "dashboard.html"])
def test_every_interactive_element_has_a_kebab_case_data_testid(page: str) -> None:
    missing = [
        (tag, attrs)
        for tag, attrs in _elements(page)
        if tag in INTERACTIVE and not attrs.get("data-testid")
    ]
    ids = _test_ids(page)

    assert missing == []
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", test_id) for test_id in ids)


def test_dashboard_shows_every_required_metric_and_the_hit_rate_chart() -> None:
    fields = {
        attrs["data-field"] for _, attrs in _elements("dashboard.html") if attrs.get("data-field")
    }

    assert fields == DASHBOARD_FIELDS
    assert "dashboard-hit-rate-chart" in _test_ids("dashboard.html")
    assert {"usage-scope-select", "usage-team-input", "usage-group-by-select"} <= _test_ids(
        "dashboard.html"
    )


def test_admin_page_exposes_upload_update_retire_history_and_approve() -> None:
    ids = _test_ids("admin.html")
    script = (WEB / "app.js").read_text(encoding="utf-8")

    assert {
        "upload-submit-button",
        "update-submit-button",
        "history-submit-button",
        "upload-classification-select",
    } <= ids
    for dynamic in (
        "documents-row-retire-button",
        "documents-row-history-button",
        "documents-row-approve-button",
    ):
        assert dynamic in script


def test_app_never_renders_api_data_as_html_and_uses_pkce() -> None:
    script = (WEB / "app.js").read_text(encoding="utf-8")

    for sink in (".innerHTML", ".outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in script
    assert 'code_challenge_method: "S256"' in script
    assert "console.log" not in script
    assert "window.KEEPSAKE_CONFIG" in (WEB / "config.example.js").read_text(encoding="utf-8")
