#!/usr/bin/env python3
"""OCR backends for the VisDocAgentBench agent harness."""

from __future__ import annotations

import json
import os
import tempfile
import ast
import threading
from pathlib import Path
from typing import Any


class OCRBackend:
    engine_name = "abstract"

    def recognize(self, image_path: Path) -> dict[str, Any]:
        raise NotImplementedError


def _collect_text_blocks(obj: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    def parse_parsing_res_string(value: str) -> dict[str, Any] | None:
        if "content:" not in value or "bbox:" not in value:
            return None
        label = ""
        bbox: Any = None
        content_lines: list[str] = []
        in_content = False
        for line in value.splitlines():
            if line.startswith("#################"):
                if in_content:
                    break
                continue
            if line.startswith("label:"):
                label = line.split(":", 1)[1].strip()
                in_content = False
            elif line.startswith("bbox:"):
                raw_bbox = line.split(":", 1)[1].strip()
                try:
                    bbox = ast.literal_eval(raw_bbox)
                except Exception:
                    bbox = raw_bbox
                in_content = False
            elif line.startswith("content:"):
                content_lines.append(line.split(":", 1)[1].strip())
                in_content = True
            elif in_content:
                content_lines.append(line.rstrip())
        text = "\n".join(content_lines).strip()
        if not text:
            return None
        return {"text": text, "bbox": _json_safe(bbox), "label": label}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            text = value.get("text") or value.get("transcription") or value.get("content")
            if isinstance(text, str) and text.strip():
                bbox = (
                    value.get("bbox")
                    or value.get("box")
                    or value.get("coordinate")
                    or value.get("poly")
                    or value.get("points")
                )
                blocks.append(
                    {
                        "text": text.strip(),
                        "bbox": _json_safe(bbox),
                        "confidence": _json_safe(value.get("confidence") or value.get("score")),
                    }
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            parsed = parse_parsing_res_string(value)
            if parsed is not None:
                blocks.append(parsed)

    visit(obj)
    return blocks


def _json_safe(value: Any) -> Any:
    """Convert OCR outputs such as ndarray/tensor/scalars into JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "numpy"):
        try:
            return _json_safe(value.numpy())
        except Exception:
            pass
    return str(value)


def _compact_ocr_result(parsed: Any) -> Any:
    """Keep human-useful OCR fields while dropping large image arrays."""
    parsed = _json_safe(parsed)
    if not isinstance(parsed, dict):
        return parsed

    compact: dict[str, Any] = {}
    for key in ("input_path", "page_index", "page_count", "width", "height"):
        if key in parsed:
            compact[key] = parsed[key]
    if "parsing_res_list" in parsed:
        compact["parsing_res_list"] = parsed["parsing_res_list"]

    layout = parsed.get("layout_det_res")
    if isinstance(layout, dict):
        boxes = layout.get("boxes") or layout.get("dt_polys") or layout.get("layout_det_res")
        if boxes is not None:
            compact["layout_det_res"] = {"boxes": boxes}

    blocks = _collect_text_blocks(parsed)
    compact["text_blocks"] = blocks
    compact["ocr_text"] = "\n\n".join(block["text"] for block in blocks if block.get("text"))
    return compact


class PaddleOCRVLBackend(OCRBackend):
    engine_name = "paddleocr_vl_1_6"

    def __init__(
        self,
        *,
        pipeline_version: str = "v1.6",
        model_dir: str = "",
        layout_model_dir: str = "",
        vl_rec_server_url: str = "",
    ):
        self.pipeline_version = pipeline_version
        self.model_dir = model_dir
        self.layout_model_dir = layout_model_dir
        self.vl_rec_server_url = vl_rec_server_url
        self._pipeline: Any = None
        self._lock = threading.Lock()

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "outputs/.paddlex_cache")
        try:
            from paddleocr import PaddleOCRVL  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR-VL backend requires the official PaddleOCR package. "
                "Install PaddlePaddle and `paddleocr[doc-parser]>=3.6.0`, then rerun. "
                f"Configured local model path: {self.model_dir or '(not set)'}"
            ) from exc

        kwargs: dict[str, Any] = {"pipeline_version": self.pipeline_version}
        if self.vl_rec_server_url:
            kwargs["vl_rec_backend"] = "vllm-server"
            kwargs["vl_rec_server_url"] = self.vl_rec_server_url
        if self.layout_model_dir:
            kwargs["layout_detection_model_dir"] = self.layout_model_dir

        if self.model_dir:
            try:
                self._pipeline = PaddleOCRVL(**kwargs, vl_rec_model_dir=self.model_dir)
                return self._pipeline
            except TypeError:
                try:
                    self._pipeline = PaddleOCRVL(**kwargs, model_dir=self.model_dir)
                    return self._pipeline
                except TypeError:
                    pass
        self._pipeline = PaddleOCRVL(**kwargs)
        return self._pipeline

    def recognize(self, image_path: Path) -> dict[str, Any]:
        with self._lock:
            pipeline = self._load_pipeline()
            output = list(pipeline.predict(str(image_path)))
        raw_results: list[Any] = []
        blocks: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory(prefix="visdocagentbench_ocr_") as tmp:
            tmpdir = Path(tmp)
            for index, result in enumerate(output):
                parsed: Any = None
                if isinstance(result, dict):
                    parsed = result
                elif hasattr(result, "to_dict"):
                    parsed = result.to_dict()
                elif hasattr(result, "json"):
                    value = result.json()
                    parsed = json.loads(value) if isinstance(value, str) else value
                elif hasattr(result, "save_to_json"):
                    before = set(tmpdir.glob("*.json"))
                    result.save_to_json(save_path=str(tmpdir))
                    after = set(tmpdir.glob("*.json"))
                    new_files = sorted(after - before)
                    if new_files:
                        parsed = json.loads(new_files[-1].read_text(encoding="utf-8"))
                if parsed is None:
                    parsed = {"text": str(result)}
                parsed = _compact_ocr_result(parsed)
                raw_results.append(parsed)
                for block in _collect_text_blocks(parsed):
                    if block not in blocks:
                        blocks.append(block)

        ocr_text = "\n".join(block["text"] for block in blocks if block.get("text"))
        if not ocr_text and raw_results:
            ocr_text = "\n".join(str(result) for result in raw_results)
        return {
            "ocr_text": ocr_text,
            "blocks": blocks,
            "engine": self.engine_name,
            "raw_result_count": len(raw_results),
        }


class DisabledOCRBackend(OCRBackend):
    engine_name = "disabled"

    def recognize(self, image_path: Path) -> dict[str, Any]:
        raise RuntimeError("OCR backend is disabled for this run")


def build_ocr_backend(args: Any) -> OCRBackend:
    backend = str(getattr(args, "ocr_backend", "paddleocr_vl")).strip().lower()
    if backend in {"none", "disabled"}:
        return DisabledOCRBackend()
    if backend in {"paddleocr_vl", "paddleocr-vl", "paddleocr_vl_1_6"}:
        return PaddleOCRVLBackend(
            pipeline_version=str(getattr(args, "paddleocr_pipeline_version", "v1.6")),
            model_dir=str(getattr(args, "paddleocr_model_dir", "")),
            layout_model_dir=str(getattr(args, "paddleocr_layout_model_dir", "")),
            vl_rec_server_url=str(getattr(args, "paddleocr_vl_rec_server_url", "")),
        )
    raise ValueError(f"Unknown OCR backend: {backend}")
