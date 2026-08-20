from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_isolated_module(
    name: str,
    path: Path,
    *,
    hidden_modules: tuple[str, ...],
) -> ModuleType:
    original_path = list(sys.path)
    hidden = {key: sys.modules.pop(key, None) for key in hidden_modules}
    previous_module = sys.modules.get(name)
    try:
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load test module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path
        sys.modules.pop(name, None)
        if previous_module is not None:
            sys.modules[name] = previous_module
        for key in hidden_modules:
            sys.modules.pop(key, None)
            if hidden[key] is not None:
                sys.modules[key] = hidden[key]


build_ocr_cache = load_isolated_module(
    "test_build_ocr_cache_module",
    ROOT / "dataset_tools" / "build_ocr_cache.py",
    hidden_modules=("common", "corpus", "ocr_backends"),
)
prepare_inputs = load_isolated_module(
    "test_prepare_inputs_module",
    ROOT / "baselines" / "non_agentic" / "prepare_inputs.py",
    hidden_modules=("common",),
)
page_ocr_cache = load_isolated_module(
    "test_page_ocr_cache_module",
    ROOT / "baselines" / "agent_ocr_text" / "ocr_cache.py",
    hidden_modules=("text_common",),
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def ocr_args(
    tmp_path: Path, manifest_path: Path, out_path: Path, *, limit: int
) -> argparse.Namespace:
    return argparse.Namespace(
        page_manifest=manifest_path,
        out=out_path,
        limit=limit,
        resume=True,
        fail_fast=False,
        ocr_backend="test_ocr",
        paddleocr_model_dir=str(tmp_path / "ocr_model"),
        paddleocr_layout_model_dir=str(tmp_path / "layout_model"),
        paddleocr_pipeline_version="test-v1",
        paddleocr_vl_rec_server_url="",
    )


def make_manifest(
    tmp_path: Path, count: int = 3
) -> tuple[Path, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for index in range(1, count + 1):
        page_id = f"page_{index}"
        image_path = tmp_path / f"{page_id}.png"
        image_path.write_bytes(f"image-{index}".encode())
        rows.append(
            {
                "page_id": page_id,
                "document_id": "doc_1",
                "page_index": index,
                "image_path": str(image_path),
            }
        )
    manifest_path = tmp_path / "pages.jsonl"
    write_jsonl(manifest_path, rows)
    return manifest_path, rows


def cached_row(
    page: dict[str, object],
    *,
    manifest_path: Path,
    config_hash: str,
    ok: bool = True,
) -> dict[str, object]:
    image_path = build_ocr_cache.resolve_page_image_path(page, manifest_path, ROOT)
    return {
        "page_id": page["page_id"],
        "document_id": page["document_id"],
        "page_index": page["page_index"],
        "image_sha256": build_ocr_cache.sha256_file(image_path),
        "ocr_config_sha256": config_hash,
        "ocr_ok": ok,
        "ocr_engine": "test_ocr",
        "ocr_text": "cached" if ok else "",
        "blocks": [],
        "text_chars": 6 if ok else 0,
        "text_words": 1 if ok else 0,
    }


def test_ocr_limit_preserves_complete_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest = make_manifest(tmp_path)
    out_path = tmp_path / "ocr.jsonl"
    args = ocr_args(tmp_path, manifest_path, out_path, limit=1)
    config_hash = build_ocr_cache.ocr_config_hash(args)
    write_jsonl(
        out_path,
        [
            cached_row(page, manifest_path=manifest_path, config_hash=config_hash)
            for page in manifest
        ],
    )
    monkeypatch.setattr(build_ocr_cache, "parse_args", lambda: args)

    assert build_ocr_cache.main() == 0
    rows = build_ocr_cache.read_jsonl(out_path)
    assert [row["page_id"] for row in rows] == [page["page_id"] for page in manifest]


def test_ocr_failed_row_is_retried_with_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest = make_manifest(tmp_path, count=2)
    out_path = tmp_path / "ocr.jsonl"
    args = ocr_args(tmp_path, manifest_path, out_path, limit=1)
    config_hash = build_ocr_cache.ocr_config_hash(args)
    write_jsonl(
        out_path,
        [
            cached_row(
                manifest[0],
                manifest_path=manifest_path,
                config_hash=config_hash,
                ok=False,
            ),
            cached_row(
                manifest[1], manifest_path=manifest_path, config_hash=config_hash
            ),
        ],
    )

    class FakeBackend:
        def __init__(self) -> None:
            self.calls: list[Path] = []

        def recognize(self, image_path: Path) -> dict[str, object]:
            self.calls.append(image_path)
            return {"ocr_text": "recovered", "blocks": [], "engine": "test_ocr"}

    backend = FakeBackend()
    monkeypatch.setattr(build_ocr_cache, "parse_args", lambda: args)
    monkeypatch.setattr(build_ocr_cache, "build_ocr_backend", lambda _: backend)

    assert build_ocr_cache.main() == 0
    rebuilt = build_ocr_cache.read_jsonl(out_path)
    assert len(rebuilt) == 2
    assert len(backend.calls) == 1
    assert rebuilt[0]["ocr_ok"] is True
    assert rebuilt[0]["ocr_text"] == "recovered"


def test_ocr_limit_removes_deferred_stale_success_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest = make_manifest(tmp_path, count=3)
    out_path = tmp_path / "ocr.jsonl"
    args = ocr_args(tmp_path, manifest_path, out_path, limit=1)
    config_hash = build_ocr_cache.ocr_config_hash(args)
    write_jsonl(
        out_path,
        [
            cached_row(page, manifest_path=manifest_path, config_hash=config_hash)
            for page in manifest
        ],
    )
    Path(str(manifest[0]["image_path"])).write_bytes(b"changed-image-1")
    Path(str(manifest[1]["image_path"])).write_bytes(b"changed-image-2")

    class FakeBackend:
        def recognize(self, _image_path: Path) -> dict[str, object]:
            return {"ocr_text": "refreshed", "blocks": [], "engine": "test_ocr"}

    monkeypatch.setattr(build_ocr_cache, "parse_args", lambda: args)
    monkeypatch.setattr(build_ocr_cache, "build_ocr_backend", lambda _: FakeBackend())

    assert build_ocr_cache.main() == 0
    rebuilt = build_ocr_cache.read_jsonl(out_path)
    assert [row["page_id"] for row in rebuilt] == ["page_1", "page_3"]
    assert {row["page_id"]: row["ocr_text"] for row in rebuilt} == {
        "page_1": "refreshed",
        "page_3": "cached",
    }

    report = json.loads(
        out_path.with_suffix(".report.json").read_text(encoding="utf-8")
    )
    assert report["num_current_pages"] == 2
    assert report["remaining_page_ids"] == ["page_2"]
    assert report["cache_complete"] is False

    page_by_id = {str(row["page_id"]): row for row in manifest}
    text_by_page = {str(row["page_id"]): row for row in rebuilt}
    with pytest.raises(ValueError, match="missing 1 corpus pages"):
        prepare_inputs.validate_page_text_cache(
            page_by_id=page_by_id,
            text_by_page=text_by_page,
            text_field="ocr_text",
        )
    with pytest.raises(ValueError, match="missing 1 corpus pages"):
        page_ocr_cache.load_page_ocr_cache(
            out_path,
            expected_page_ids=set(page_by_id),
        )


def test_ocr_cache_invalidates_changed_configuration_or_page(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = make_manifest(tmp_path, count=1)
    args = ocr_args(tmp_path, manifest_path, tmp_path / "ocr.jsonl", limit=0)
    config_hash = build_ocr_cache.ocr_config_hash(args)
    row = cached_row(
        manifest[0], manifest_path=manifest_path, config_hash=config_hash
    )
    manifest_by_id = {"page_1": manifest[0]}

    changed_config, invalidated = build_ocr_cache.reusable_existing_rows(
        existing_by_id={"page_1": row},
        manifest_by_id=manifest_by_id,
        page_manifest=manifest_path,
        config_hash="different-config",
    )
    assert changed_config == {}
    assert invalidated == 1

    Path(str(manifest[0]["image_path"])).write_bytes(b"changed-image")
    changed_page, invalidated = build_ocr_cache.reusable_existing_rows(
        existing_by_id={"page_1": row},
        manifest_by_id=manifest_by_id,
        page_manifest=manifest_path,
        config_hash=config_hash,
    )
    assert changed_page == {}
    assert invalidated == 1


def test_page_text_cache_requires_exact_page_coverage() -> None:
    pages = {"p1": {"page_id": "p1"}, "p2": {"page_id": "p2"}}
    with pytest.raises(ValueError, match="missing 1 corpus pages"):
        prepare_inputs.validate_page_text_cache(
            page_by_id=pages,
            text_by_page={"p1": {"page_id": "p1", "ocr_text": "one"}},
            text_field="ocr_text",
        )
    with pytest.raises(ValueError, match="non-corpus pages"):
        prepare_inputs.validate_page_text_cache(
            page_by_id=pages,
            text_by_page={
                "p1": {"page_id": "p1", "ocr_text": "one"},
                "p2": {"page_id": "p2", "ocr_text": "two"},
                "p3": {"page_id": "p3", "ocr_text": "three"},
            },
            text_field="ocr_text",
        )


def test_page_text_cache_rejects_failed_ocr_but_allows_empty_success() -> None:
    pages = {"p1": {"page_id": "p1"}}
    with pytest.raises(ValueError, match="failed OCR rows"):
        prepare_inputs.validate_page_text_cache(
            page_by_id=pages,
            text_by_page={
                "p1": {"page_id": "p1", "ocr_text": "", "ocr_ok": False}
            },
            text_field="ocr_text",
        )
    prepare_inputs.validate_page_text_cache(
        page_by_id=pages,
        text_by_page={"p1": {"page_id": "p1", "ocr_text": "", "ocr_ok": True}},
        text_field="ocr_text",
    )


def test_build_page_text_corpus_never_substitutes_missing_text() -> None:
    manifest = [
        {
            "page_id": "p1",
            "document_id": "d1",
            "page_index": 1,
            "image_path": "p1.png",
        }
    ]
    with pytest.raises(KeyError, match="p1"):
        prepare_inputs.build_page_text_corpus(manifest, {}, "ocr_text")
