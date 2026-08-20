from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
NON_AGENTIC = ROOT / "baselines" / "non_agentic"
sys.path.insert(0, str(NON_AGENTIC))

import cache_text_embeddings  # noqa: E402
import cache_visual_embeddings  # noqa: E402
from common import (  # noqa: E402
    directory_identity,
    file_collection_identity,
    read_jsonl,
    write_jsonl,
)


def test_visual_embedding_identity_tracks_image_bytes_and_preprocessing(
    tmp_path: Path,
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"first-image")
    item = {"page_id": "p1", "image_path": "page.png"}
    args = SimpleNamespace(
        base_url="http://localhost:8002/v1/",
        model="visual-model",
        service_revision="revision-a",
        normalize=True,
        id_field="page_id",
        text_field="",
        image_field="image_path",
        project_root=tmp_path,
        max_pixels=1000,
        max_file_size_mb=10.0,
    )

    first = cache_visual_embeddings.current_fingerprint(item, args)
    image.write_bytes(b"other-image")
    second = cache_visual_embeddings.current_fingerprint(item, args)
    assert first != second

    args.max_pixels = 2000
    third = cache_visual_embeddings.current_fingerprint(item, args)
    assert second != third


def test_text_embedding_identity_tracks_normalization_endpoint_and_revision() -> None:
    item = {"query_id": "q1", "text": "same input"}
    args = SimpleNamespace(
        base_url="http://localhost:8001/v1",
        model="text-model",
        service_revision="revision-a",
        normalize=True,
        id_field="query_id",
        text_field="text",
        include_text=False,
    )

    first = cache_text_embeddings.current_fingerprint(item, args)
    args.normalize = False
    assert cache_text_embeddings.current_fingerprint(item, args) != first
    args.normalize = True
    args.base_url = "http://localhost:9001/v1"
    assert cache_text_embeddings.current_fingerprint(item, args) != first
    args.base_url = "http://localhost:8001/v1"
    args.service_revision = "revision-b"
    assert cache_text_embeddings.current_fingerprint(item, args) != first


def test_visual_embedding_limit_preserves_current_rows_and_defers_stale_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items_path = tmp_path / "visual_items.jsonl"
    output_path = tmp_path / "visual_embeddings.jsonl"
    report_path = tmp_path / "visual_report.json"
    items = []
    for index in range(1, 5):
        image_path = tmp_path / f"page_{index}.png"
        image_path.write_bytes(f"image-{index}".encode())
        items.append({"page_id": f"p{index}", "image_path": image_path.name})
    write_jsonl(items_path, items)

    args = SimpleNamespace(
        items=items_path,
        out=output_path,
        report=report_path,
        id_field="page_id",
        text_field="",
        image_field="image_path",
        project_root=tmp_path,
        base_url="http://localhost:8002/v1",
        model="visual-model",
        service_revision="revision-a",
        api_key="EMPTY",
        workers=1,
        timeout=1.0,
        max_retries=0,
        retry_backoff=0.0,
        limit=1,
        max_pixels=1000,
        max_file_size_mb=10.0,
        normalize=True,
        resume=True,
    )

    def make_row(item: dict[str, object]) -> dict[str, object]:
        return cache_visual_embeddings.make_embedding_row(
            item=item,
            id_field=args.id_field,
            model=args.model,
            embedding=[1.0, 0.0],
            normalize=args.normalize,
            input_sha256=cache_visual_embeddings.current_fingerprint(item, args),
            base_url=args.base_url,
            service_revision=args.service_revision,
        )

    write_jsonl(output_path, [make_row(item) for item in items])
    encoded: list[str] = []

    def fake_encode(
        item: dict[str, object], _args: SimpleNamespace
    ) -> tuple[str, dict[str, object]]:
        encoded.append(str(item[args.id_field]))
        return str(item[args.id_field]), make_row(item)

    monkeypatch.setattr(cache_visual_embeddings, "parse_args", lambda: args)
    monkeypatch.setattr(cache_visual_embeddings, "encode_one", fake_encode)

    assert cache_visual_embeddings.main() == 0
    assert encoded == []
    assert [row["page_id"] for row in read_jsonl(output_path)] == [
        "p1",
        "p2",
        "p3",
        "p4",
    ]

    (tmp_path / "page_3.png").write_bytes(b"changed-3")
    (tmp_path / "page_4.png").write_bytes(b"changed-4")
    assert cache_visual_embeddings.main() == 0
    assert encoded == ["p3"]
    assert [row["page_id"] for row in read_jsonl(output_path)] == [
        "p1",
        "p2",
        "p3",
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["num_pending_total"] == 2
    assert report["num_pending_deferred"] == 1
    assert report["num_output_embeddings"] == 3


def test_text_embedding_limit_preserves_current_rows_and_defers_stale_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items_path = tmp_path / "text_items.jsonl"
    output_path = tmp_path / "text_embeddings.jsonl"
    report_path = tmp_path / "text_report.json"
    items = [
        {"query_id": f"q{index}", "text": f"text {index}"}
        for index in range(1, 5)
    ]
    write_jsonl(items_path, items)

    args = SimpleNamespace(
        items=items_path,
        out=output_path,
        report=report_path,
        id_field="query_id",
        text_field="text",
        base_url="http://localhost:8001/v1",
        model="text-model",
        service_revision="revision-a",
        api_key="EMPTY",
        batch_size=1,
        timeout=1.0,
        max_retries=0,
        retry_backoff=0.0,
        limit=1,
        normalize=True,
        resume=True,
        include_text=False,
    )

    def make_row(item: dict[str, object]) -> dict[str, object]:
        return cache_text_embeddings.make_embedding_row(
            item=item,
            id_field=args.id_field,
            text_field=args.text_field,
            model=args.model,
            embedding=[1.0, 0.0],
            normalize=args.normalize,
            include_text=args.include_text,
            input_sha256=cache_text_embeddings.current_fingerprint(item, args),
            base_url=args.base_url,
            service_revision=args.service_revision,
        )

    write_jsonl(output_path, [make_row(item) for item in items])
    embedded_texts: list[str] = []

    def fake_call_embeddings(
        _base_url: str,
        _model: str,
        _api_key: str,
        texts: list[str],
        _timeout: float,
        _max_retries: int,
        _retry_backoff: float,
    ) -> list[list[float]]:
        embedded_texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(cache_text_embeddings, "parse_args", lambda: args)
    monkeypatch.setattr(
        cache_text_embeddings, "call_embeddings", fake_call_embeddings
    )

    assert cache_text_embeddings.main() == 0
    assert embedded_texts == []
    assert [row["query_id"] for row in read_jsonl(output_path)] == [
        "q1",
        "q2",
        "q3",
        "q4",
    ]

    items[2]["text"] = "changed 3"
    items[3]["text"] = "changed 4"
    write_jsonl(items_path, items)
    assert cache_text_embeddings.main() == 0
    assert embedded_texts == ["changed 3"]
    assert [row["query_id"] for row in read_jsonl(output_path)] == [
        "q1",
        "q2",
        "q3",
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["num_pending_total"] == 2
    assert report["num_pending_deferred"] == 1
    assert report["num_output_embeddings"] == 3


def test_file_and_directory_identities_track_content(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "nested" / "second.bin"
    second.parent.mkdir()
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    collection_before = file_collection_identity([("first", first), ("second", second)])
    directory_before = directory_identity(tmp_path)
    second.write_bytes(b"changed")
    collection_after = file_collection_identity([("first", first), ("second", second)])
    directory_after = directory_identity(tmp_path)

    assert collection_before["sha256"] != collection_after["sha256"]
    assert directory_before["sha256"] != directory_after["sha256"]


@pytest.mark.parametrize(
    ("directory", "module_name", "class_name"),
    [
        ("agent_visual", "vl_retriever", "VisualRetriever"),
        ("agent_ocr_text", "text_retriever", "TextRetriever"),
    ],
)
def test_page_matrix_cache_rejects_changed_embedding_source(
    tmp_path: Path,
    directory: str,
    module_name: str,
    class_name: str,
) -> None:
    script = textwrap.dedent(
        f"""
        import json
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        import numpy as np

        root = Path({str(ROOT)!r})
        sys.path.insert(0, str(root / 'baselines' / {directory!r}))
        module = __import__({module_name!r})
        retriever_type = getattr(module, {class_name!r})
        work = Path({str(tmp_path)!r})
        source = work / 'embeddings.jsonl'
        cache = work / 'matrix.npz'
        pages = [
            SimpleNamespace(page_id='p1', page_handle='h1'),
            SimpleNamespace(page_id='p2', page_handle='h2'),
        ]
        handle_space = SimpleNamespace(pages=pages)

        def write(v1, v2):
            source.write_text(
                json.dumps({{'page_id': 'p1', 'embedding': v1}}) + '\\n' +
                json.dumps({{'page_id': 'p2', 'embedding': v2}}) + '\\n',
                encoding='utf-8',
            )

        retriever = retriever_type.__new__(retriever_type)
        retriever.model = 'model-a'
        write([1.0, 0.0], [0.0, 1.0])
        np.savez_compressed(
            cache,
            page_ids=np.asarray(['p1', 'p2']),
            page_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        )
        _, first = retriever._load_page_matrix(
            page_embeddings_path=source,
            handle_space=handle_space,
            matrix_cache_path=cache,
        )
        assert np.allclose(first[0], [1.0, 0.0])
        write([0.0, 1.0], [1.0, 0.0])
        _, second = retriever._load_page_matrix(
            page_embeddings_path=source,
            handle_space=handle_space,
            matrix_cache_path=cache,
        )
        assert not np.array_equal(first, second)
        assert np.allclose(second[0], [0.0, 1.0])

        retriever.model = 'model-b'
        retriever._load_page_matrix(
            page_embeddings_path=source,
            handle_space=handle_space,
            matrix_cache_path=cache,
        )
        with np.load(cache, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json'].item()))
        assert metadata['identity']['embedding_model'] == 'model-b'
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
