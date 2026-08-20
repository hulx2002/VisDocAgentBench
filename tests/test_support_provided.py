from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
SUPPORT_DIR = ROOT / "baselines" / "support_provided"
sys.path.insert(0, str(SUPPORT_DIR))

from support_context import (  # noqa: E402
    INITIAL_EVENT_NAME,
    inject_initial_page_observations,
    support_page_ids,
)


class HandleSpace:
    def __init__(self, pages: dict[str, SimpleNamespace]):
        self.pages = pages

    def page_id_to_handle(self, page_id: str) -> str:
        return self.pages[page_id].page_handle

    def require_page(self, page_handle: str) -> SimpleNamespace:
        return next(page for page in self.pages.values() if page.page_handle == page_handle)


class Runtime:
    def __init__(self, pages: dict[str, SimpleNamespace]):
        self.handle_space = HandleSpace(pages)
        self.trace: list[dict] = []
        self.discovered_page_handles: set[str] = set()
        self.inspected_page_handles: set[str] = set()
        self.seen_page_handles: set[str] = set()
        self.seen_image_handles: set[str] = set()
        self.usage = {"search_calls": 0, "inspect_calls": 0}

    def public_usage(self) -> dict[str, int]:
        return dict(self.usage)

    def _record(self, tool_name, arguments, observation, artifacts) -> None:
        self.trace.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation,
                "artifacts": list(artifacts),
            }
        )


def test_complete_support_is_role_blind_and_free(tmp_path: Path) -> None:
    row = {
        "query_id": "q",
        "query": "find the page",
        "gold_page_id": "answer",
        "level": 3,
        "support_unit_ids": ["s1", "s2"],
    }
    pages: dict[str, SimpleNamespace] = {}
    for index, page_id in enumerate(support_page_ids(row), start=1):
        image = tmp_path / f"{page_id}.png"
        image.write_bytes(b"image")
        pages[page_id] = SimpleNamespace(
            page_handle=f"page_{index:06d}",
            page_width=1000,
            page_height=1400,
            image_path=image,
        )
    runtime = Runtime(pages)
    metadata = inject_initial_page_observations(
        runtime, row, route="visual", shuffle_seed=20260815
    )
    assert runtime.public_usage() == {"search_calls": 0, "inspect_calls": 0}
    assert len(runtime.trace) == 1
    assert runtime.trace[0]["tool_name"] == INITIAL_EVENT_NAME
    assert set(metadata["injected_page_ids"]) == {"s1", "s2"}
    event = runtime.trace[0]
    visible = json.dumps(
        {
            "tool_name": event["tool_name"],
            "arguments": event["arguments"],
            "observation": event["observation"],
            "artifacts": event["artifacts"],
        },
        ensure_ascii=False,
    ).lower()
    for forbidden in ("gold", "support", "first hop", "second hop", "path order"):
        assert forbidden not in visible


def test_support_shape_is_checked() -> None:
    bad = {"query_id": "q", "gold_page_id": "p", "level": 3, "support_unit_ids": ["s1"]}
    try:
        support_page_ids(bad)
    except ValueError as error:
        assert "invalid support path" in str(error)
    else:
        raise AssertionError("invalid support shape was accepted")
