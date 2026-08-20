from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines" / "agent_common"))

from benchmark_io import load_benchmark_rows  # noqa: E402


DATA_ROOT = ROOT / "data"
REQUIRED_DATA_FILES = (
    DATA_ROOT / "benchmark" / "queries.jsonl",
    DATA_ROOT / "benchmark" / "evaluator_annotations.jsonl",
    DATA_ROOT / "corpus" / "documents.jsonl",
    DATA_ROOT / "corpus" / "pages.jsonl",
)


def require_public_dataset() -> None:
    missing = [path for path in REQUIRED_DATA_FILES if not path.is_file()]
    if missing:
        pytest.skip(
            "public dataset is not downloaded; run `python scripts/download_data.py` "
            "before the dataset integration tests"
        )


def test_missing_public_dataset_is_reported_as_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys.modules[__name__],
        "REQUIRED_DATA_FILES",
        (tmp_path / "missing.jsonl",),
    )
    with pytest.raises(pytest.skip.Exception, match="public dataset is not downloaded"):
        require_public_dataset()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_public_benchmark_contract() -> None:
    require_public_dataset()
    data = DATA_ROOT
    queries = read_jsonl(data / "benchmark" / "queries.jsonl")
    annotations = read_jsonl(data / "benchmark" / "evaluator_annotations.jsonl")
    documents = read_jsonl(data / "corpus" / "documents.jsonl")
    pages = read_jsonl(data / "corpus" / "pages.jsonl")

    assert len(queries) == 120
    assert len(annotations) == 120
    assert len(documents) == 100
    assert len(pages) == 2375
    assert Counter(int(row["level"]) for row in annotations) == Counter({1: 40, 2: 40, 3: 40})
    assert len({row["answer_page_id"] for row in annotations}) == 120
    assert all(len(row["support_page_ids"]) == int(row["level"]) - 1 for row in annotations)


def test_standard_agent_rows_do_not_contain_support_annotations() -> None:
    require_public_dataset()
    data = DATA_ROOT
    rows = load_benchmark_rows(
        data / "benchmark" / "queries.jsonl",
        data / "benchmark" / "evaluator_annotations.jsonl",
    )
    assert len(rows) == 120
    assert all("support_page_ids" not in row and "support_unit_ids" not in row for row in rows)
    assert all(row["gold_page_id"] for row in rows)
