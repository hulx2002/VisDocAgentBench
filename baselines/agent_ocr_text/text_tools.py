#!/usr/bin/env python3
"""Tool runtime for the VisDocAgentBench OCR-text retrieval agent."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from text_common import ensure_dir
from text_corpus import PageHandleSpace
from text_retriever import TextRetriever
from ocr_cache import CachedPageOCR


INFORMATION_TOOLS = frozenset(
    {"text_search", "get_page_ocr", "inspect_pages", "crop_pages"}
)
MAX_SEARCH_RESULTS = 50
MAX_VIEW_IMAGES_PER_CALL = 10
MIN_CROP_SIDE_FRACTION = 0.01


@dataclass
class ToolUsage:
    text_search_calls: int = 0
    get_page_ocr_calls: int = 0
    inspect_pages_calls: int = 0
    inspected_page_images: int = 0
    crop_pages_calls: int = 0
    cropped_regions: int = 0


@dataclass
class ImageArtifact:
    handle: str
    path: Path
    width: int
    height: int
    source_page_handle: str = ""
    region: list[float] = field(default_factory=list)


class AgentTextToolRuntime:
    def __init__(
        self,
        *,
        handle_space: PageHandleSpace,
        retriever: TextRetriever,
        page_ocr_cache: Mapping[str, CachedPageOCR],
        artifact_dir: Path,
        enabled_tools: frozenset[str] | None = None,
    ):
        self.handle_space = handle_space
        self.retriever = retriever
        self.page_ocr_cache = page_ocr_cache
        self.artifact_dir = ensure_dir(artifact_dir)
        self.crop_dir = ensure_dir(self.artifact_dir / "crops")
        self.enabled_tools = (
            INFORMATION_TOOLS if enabled_tools is None else frozenset(enabled_tools)
        )
        unknown_tools = self.enabled_tools - INFORMATION_TOOLS
        if unknown_tools:
            raise ValueError(
                f"Unknown enabled information tools: {sorted(unknown_tools)}"
            )

        self.usage = ToolUsage()
        self.image_registry: dict[str, ImageArtifact] = {}
        self.trace: list[dict[str, Any]] = []
        self.discovered_page_handles: set[str] = set()
        self.inspected_page_handles: set[str] = set()
        self.seen_page_handles: set[str] = set()
        self.seen_image_handles: set[str] = set()
        self._artifact_recency: list[str] = []
        self._visible_artifact_batch: list[str] = []
        self.crop_counter = 0

        for page in self.handle_space.pages:
            self.image_registry[page.page_handle] = ImageArtifact(
                handle=page.page_handle,
                path=page.image_path,
                width=page.page_width,
                height=page.page_height,
            )

    def public_usage(self) -> dict[str, int]:
        return {
            "text_search_calls": self.usage.text_search_calls,
            "get_page_ocr_calls": self.usage.get_page_ocr_calls,
            "inspect_pages_calls": self.usage.inspect_pages_calls,
            "inspected_page_images": self.usage.inspected_page_images,
            "crop_pages_calls": self.usage.crop_pages_calls,
            "cropped_regions": self.usage.cropped_regions,
        }

    def public_per_call_limits(self) -> dict[str, int | float]:
        return {
            "max_text_search_top_k": MAX_SEARCH_RESULTS,
            "max_get_page_ocr_pages_per_call": 1,
            "max_inspect_pages_per_call": MAX_VIEW_IMAGES_PER_CALL,
            "max_crop_regions_per_call": MAX_VIEW_IMAGES_PER_CALL,
        }

    def checkpoint_state(self) -> dict[str, Any]:
        crop_artifacts = []
        for artifact in self.image_registry.values():
            if not artifact.source_page_handle:
                continue
            crop_artifacts.append(
                {
                    "handle": artifact.handle,
                    "path": str(artifact.path.resolve()),
                    "width": artifact.width,
                    "height": artifact.height,
                    "source_page_handle": artifact.source_page_handle,
                    "region": list(artifact.region),
                }
            )
        return {
            "runtime": "text",
            "version": 1,
            "usage": self.public_usage(),
            "trace": copy.deepcopy(self.trace),
            "discovered_page_handles": sorted(self.discovered_page_handles),
            "inspected_page_handles": sorted(self.inspected_page_handles),
            "seen_page_handles": sorted(self.seen_page_handles),
            "seen_image_handles": sorted(self.seen_image_handles),
            "artifact_recency": list(self._artifact_recency),
            "visible_artifact_batch": list(self._visible_artifact_batch),
            "crop_counter": self.crop_counter,
            "crop_artifacts": crop_artifacts,
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("runtime") != "text" or int(state.get("version", 0)) != 1:
            raise ValueError("incompatible text runtime checkpoint")
        usage = state.get("usage", {})
        if not isinstance(usage, dict):
            raise ValueError("text runtime checkpoint usage must be an object")
        self.usage = ToolUsage(**{key: int(value) for key, value in usage.items()})
        self.trace = copy.deepcopy(state.get("trace", []))
        self.discovered_page_handles = set(state.get("discovered_page_handles", []))
        self.inspected_page_handles = set(state.get("inspected_page_handles", []))
        self.seen_page_handles = set(state.get("seen_page_handles", []))
        self.seen_image_handles = set(state.get("seen_image_handles", []))
        self.crop_counter = int(state.get("crop_counter", 0))

        crop_artifacts = state.get("crop_artifacts", [])
        if not isinstance(crop_artifacts, list):
            raise ValueError("text runtime checkpoint crop_artifacts must be a list")
        for row in crop_artifacts:
            if not isinstance(row, dict):
                raise ValueError("text runtime checkpoint crop artifact is invalid")
            path = Path(str(row.get("path", "")))
            if not path.is_file():
                raise ValueError(f"checkpoint crop artifact does not exist: {path}")
            artifact = ImageArtifact(
                handle=str(row.get("handle", "")),
                path=path,
                width=int(row.get("width", 0)),
                height=int(row.get("height", 0)),
                source_page_handle=str(row.get("source_page_handle", "")),
                region=[float(value) for value in row.get("region", [])],
            )
            if not artifact.handle or not artifact.source_page_handle:
                raise ValueError("checkpoint crop artifact is missing its handle or source")
            self.image_registry[artifact.handle] = artifact

        self._artifact_recency = [
            str(handle) for handle in state.get("artifact_recency", [])
        ]
        self._visible_artifact_batch = [
            str(handle) for handle in state.get("visible_artifact_batch", [])
        ]
        known_handles = set(self.image_registry)
        for label, handles in (
            ("discovered", self.discovered_page_handles),
            ("inspected", self.inspected_page_handles),
            ("seen page", self.seen_page_handles),
            ("seen image", self.seen_image_handles),
            ("artifact recency", set(self._artifact_recency)),
            ("visible artifact", set(self._visible_artifact_batch)),
        ):
            unknown = handles - known_handles
            if unknown:
                raise ValueError(
                    f"text runtime checkpoint has unknown {label} handles: {sorted(unknown)}"
                )

    def restore_legacy_tool_trace(
        self, trace: list[dict[str, Any]], usage: dict[str, Any]
    ) -> None:
        """Restore old failure rows that contain no crop artifacts."""
        self.usage = ToolUsage(**{key: int(value) for key, value in usage.items()})
        self.trace = copy.deepcopy(trace)
        for item in trace:
            tool_name = str(item.get("tool_name", ""))
            observation = item.get("observation", {})
            if not isinstance(observation, dict):
                raise ValueError("legacy text tool trace observation is invalid")
            artifacts = [str(value) for value in item.get("artifacts", [])]
            if any(handle.startswith("crop_") for handle in artifacts):
                raise ValueError(
                    "legacy text trace has crop artifacts without checkpoint paths"
                )
            if tool_name == "text_search" and observation.get("ok"):
                for result in observation.get("results", []):
                    if isinstance(result, dict):
                        self.discovered_page_handles.add(
                            str(result.get("page_handle", ""))
                        )
            elif tool_name == "get_page_ocr" and observation.get("ok"):
                self.seen_page_handles.add(str(observation.get("page_handle", "")))
            elif tool_name == "inspect_pages" and observation.get("ok"):
                for page in observation.get("loaded_pages", []):
                    if not isinstance(page, dict):
                        continue
                    handle = str(page.get("page_handle", ""))
                    self.inspected_page_handles.add(handle)
                    self.seen_page_handles.add(handle)
                    self.seen_image_handles.add(handle)
            if artifacts:
                self._refresh_artifact_recency(artifacts)
            visible = item.get("visible_artifacts_after")
            if isinstance(visible, list):
                self._visible_artifact_batch = [str(value) for value in visible]

        known_handles = set(self.image_registry)
        restored_handles = (
            self.discovered_page_handles
            | self.inspected_page_handles
            | self.seen_page_handles
            | self.seen_image_handles
            | set(self._artifact_recency)
            | set(self._visible_artifact_batch)
        )
        unknown = restored_handles - known_handles
        if unknown:
            raise ValueError(
                f"legacy text trace has unknown handles: {sorted(unknown)}"
            )

    def _refresh_artifact_recency(self, artifacts: list[str]) -> None:
        for handle in artifacts:
            if handle not in self.image_registry:
                continue
            if handle in self._artifact_recency:
                self._artifact_recency.remove(handle)
            self._artifact_recency.append(handle)

    def _record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        observation: dict[str, Any],
        artifacts: list[str] | None = None,
    ) -> None:
        artifact_handles = artifacts or []
        if artifact_handles:
            self._refresh_artifact_recency(artifact_handles)
            self._visible_artifact_batch = list(artifact_handles)
        self.trace.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation,
                "artifacts": artifact_handles,
                "visible_artifacts_after": list(self._visible_artifact_batch),
                "usage_after": self.public_usage(),
            }
        )

    def _error(
        self, tool_name: str, arguments: dict[str, Any], message: str
    ) -> tuple[dict[str, Any], list[str]]:
        observation = {"ok": False, "error": message, "usage": self.public_usage()}
        self._record(tool_name, arguments, observation, [])
        return observation, []

    def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        if tool_name not in INFORMATION_TOOLS:
            return self._error(tool_name, arguments, f"Unknown tool: {tool_name}")
        if tool_name not in self.enabled_tools:
            return self._error(
                tool_name, arguments, f"Tool is disabled for this run: {tool_name}"
            )
        if tool_name == "text_search":
            return self.text_search(arguments)
        if tool_name == "get_page_ocr":
            return self.get_page_ocr(arguments)
        if tool_name == "inspect_pages":
            return self.inspect_pages(arguments)
        if tool_name == "crop_pages":
            return self.crop_pages(arguments)
        raise AssertionError(f"Unhandled information tool: {tool_name}")

    def text_search(
        self, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return self._error(
                "text_search", arguments, "text_search requires non-empty query"
            )
        try:
            top_k = int(arguments.get("top_k", 10))
        except (TypeError, ValueError):
            return self._error(
                "text_search", arguments, "text_search top_k must be an integer"
            )
        if top_k < 1 or top_k > MAX_SEARCH_RESULTS:
            return self._error(
                "text_search",
                arguments,
                f"text_search top_k must be between 1 and {MAX_SEARCH_RESULTS}",
            )

        self.usage.text_search_calls += 1
        results = self.retriever.search(query, top_k)
        for result in results:
            self.discovered_page_handles.add(str(result["page_handle"]))
        observation = {"ok": True, "results": results, "usage": self.public_usage()}
        self._record("text_search", arguments, observation, [])
        return observation, []

    def get_page_ocr(
        self, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        unexpected_arguments = sorted(set(arguments) - {"page_handle"})
        if unexpected_arguments:
            return self._error(
                "get_page_ocr",
                arguments,
                "get_page_ocr accepts only the page_handle argument; unexpected arguments: "
                + ", ".join(unexpected_arguments),
            )
        raw_page_handle = arguments.get("page_handle", "")
        if not isinstance(raw_page_handle, str):
            return self._error(
                "get_page_ocr",
                arguments,
                "get_page_ocr accepts exactly one full-page page_handle string",
            )
        page_handle = raw_page_handle.strip()
        if not page_handle:
            return self._error(
                "get_page_ocr", arguments, "get_page_ocr requires one page_handle"
            )
        if page_handle.startswith("crop_"):
            return self._error(
                "get_page_ocr",
                arguments,
                "get_page_ocr accepts full page handles only, not crop handles",
            )
        if page_handle not in self.discovered_page_handles:
            return self._error(
                "get_page_ocr",
                arguments,
                f"Page {page_handle!r} was not returned by a prior text_search call",
            )
        try:
            page = self.handle_space.require_page(page_handle)
        except KeyError as exc:
            return self._error("get_page_ocr", arguments, str(exc))

        cached = self.page_ocr_cache.get(page.page_id)
        if cached is None:
            return self._error(
                "get_page_ocr",
                arguments,
                f"No cached OCR row exists for page {page_handle!r}",
            )

        self.usage.get_page_ocr_calls += 1
        if not cached.ocr_ok:
            observation = {
                "ok": False,
                "page_handle": page_handle,
                "error": cached.error or "Cached OCR failed for this page",
                "usage": self.public_usage(),
            }
            self._record("get_page_ocr", arguments, observation, [])
            return observation, []

        self.seen_page_handles.add(page_handle)
        observation = {
            "ok": True,
            "page_handle": page_handle,
            "ocr_text": cached.ocr_text,
            "usage": self.public_usage(),
        }
        self._record("get_page_ocr", arguments, observation, [])
        return observation, []

    def inspect_pages(
        self, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        raw_handles = arguments.get("page_handles", [])
        if isinstance(raw_handles, str):
            raw_handles = [raw_handles]
        if not isinstance(raw_handles, list) or not raw_handles:
            return self._error(
                "inspect_pages",
                arguments,
                "inspect_pages requires a non-empty page_handles list",
            )
        page_handles = [str(handle).strip() for handle in raw_handles]
        if len(page_handles) > MAX_VIEW_IMAGES_PER_CALL:
            return self._error(
                "inspect_pages",
                arguments,
                f"inspect_pages accepts at most {MAX_VIEW_IMAGES_PER_CALL} page handles per call; got {len(page_handles)}",
            )
        if len(set(page_handles)) != len(page_handles):
            return self._error(
                "inspect_pages",
                arguments,
                "inspect_pages page_handles must be distinct",
            )

        pages: list[Any] = []
        errors: list[str] = []
        for page_handle in page_handles:
            if (
                page_handle not in self.discovered_page_handles
                and page_handle not in self.seen_page_handles
            ):
                errors.append(
                    f"{page_handle}: not returned by a prior text_search call"
                )
                continue
            try:
                page = self.handle_space.require_page(page_handle)
            except KeyError as exc:
                errors.append(f"{page_handle}: {exc}")
                continue
            if not page.image_path.is_file():
                errors.append(
                    f"{page_handle}: image file does not exist: {page.image_path}"
                )
                continue
            pages.append(page)

        if errors:
            return self._error(
                "inspect_pages",
                arguments,
                "inspect_pages is atomic; no pages were loaded because "
                + "; ".join(errors),
            )

        loaded = [
            {
                "page_handle": page.page_handle,
                "page_width": page.page_width,
                "page_height": page.page_height,
            }
            for page in pages
        ]
        artifacts = [page.page_handle for page in pages]
        for page in pages:
            self.inspected_page_handles.add(page.page_handle)
            self.seen_page_handles.add(page.page_handle)
            self.seen_image_handles.add(page.page_handle)

        if artifacts != page_handles:
            raise AssertionError(
                "inspect_pages requested/loaded artifact invariant failed"
            )

        self.usage.inspect_pages_calls += 1
        self.usage.inspected_page_images += len(loaded)
        observation = {
            "ok": True,
            "requested_page_handles": page_handles,
            "loaded_pages": loaded,
            "failed_pages": [],
            "usage": self.public_usage(),
        }
        self._record("inspect_pages", arguments, observation, artifacts)
        return observation, artifacts

    def _validated_crop_request(
        self, item: Any, index: int
    ) -> tuple[Any, list[float], str, bool]:
        if not isinstance(item, dict):
            raise ValueError(f"crops[{index}] must be an object")
        raw_purpose = item.get("purpose", "")
        purpose = raw_purpose.strip() if isinstance(raw_purpose, str) else ""
        purpose_missing = not purpose
        page_handle = str(item.get("page_handle", "")).strip()
        if page_handle not in self.inspected_page_handles:
            raise ValueError(
                f"crops[{index}] source page {page_handle!r} has not been viewed with inspect_pages"
            )
        page = self.handle_space.require_page(page_handle)
        if not page.image_path.is_file():
            raise ValueError(
                f"crops[{index}] source image does not exist: {page.image_path}"
            )
        region = item.get("region", [])
        if not isinstance(region, list) or len(region) != 4:
            raise ValueError(f"crops[{index}] requires region [x1, y1, x2, y2]")
        try:
            x1, y1, x2, y2 = [float(value) for value in region]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"crops[{index}] region values must be numeric") from exc
        if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
            raise ValueError(f"crops[{index}] region coordinates must be within [0, 1]")
        if x2 - x1 < MIN_CROP_SIDE_FRACTION or y2 - y1 < MIN_CROP_SIDE_FRACTION:
            raise ValueError(f"crops[{index}] region is empty, reversed, or too small")
        return page, [x1, y1, x2, y2], purpose, purpose_missing

    def crop_pages(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        raw_crops = arguments.get("crops", [])
        if isinstance(raw_crops, dict):
            raw_crops = [raw_crops]
        if not isinstance(raw_crops, list) or not raw_crops:
            return self._error(
                "crop_pages", arguments, "crop_pages requires a non-empty crops list"
            )
        if len(raw_crops) > MAX_VIEW_IMAGES_PER_CALL:
            return self._error(
                "crop_pages",
                arguments,
                f"crop_pages accepts at most {MAX_VIEW_IMAGES_PER_CALL} regions per call; got {len(raw_crops)}",
            )

        crop_rows: list[dict[str, Any]] = []
        failed_crops: list[dict[str, Any]] = []
        artifacts: list[str] = []
        self.usage.crop_pages_calls += 1
        for index, item in enumerate(raw_crops):
            try:
                page, region, purpose, purpose_missing = (
                    self._validated_crop_request(item, index)
                )
            except (KeyError, ValueError) as exc:
                failed_crops.append(
                    {
                        "index": index,
                        "status": "invalid_request",
                        "error": str(exc),
                    }
                )
                continue

            try:
                x1, y1, x2, y2 = region
                with Image.open(page.image_path) as image:
                    width, height = image.size
                    box = (
                        int(round(x1 * width)),
                        int(round(y1 * height)),
                        int(round(x2 * width)),
                        int(round(y2 * height)),
                    )
                    crop = image.crop(box)
                    self.crop_counter += 1
                    crop_handle = (
                        f"crop_{page.page_handle.removeprefix('page_')}_"
                        f"{self.crop_counter:06d}"
                    )
                    crop_path = self.crop_dir / f"{crop_handle}.png"
                    crop.save(crop_path)
                    crop_width, crop_height = crop.size

                self.image_registry[crop_handle] = ImageArtifact(
                    handle=crop_handle,
                    path=crop_path,
                    width=crop_width,
                    height=crop_height,
                    source_page_handle=page.page_handle,
                    region=region,
                )
                self.seen_page_handles.add(page.page_handle)
                self.seen_image_handles.add(crop_handle)
                artifacts.append(crop_handle)
                crop_rows.append(
                    {
                        "index": index,
                        "status": "success",
                        "crop_handle": crop_handle,
                        "source_page_handle": page.page_handle,
                        "region": region,
                        "purpose": purpose,
                        "purpose_missing": purpose_missing,
                        "area_fraction": (x2 - x1) * (y2 - y1),
                        "crop_width": crop_width,
                        "crop_height": crop_height,
                    }
                )
            except Exception as exc:
                failed_crops.append(
                    {
                        "index": index,
                        "status": "execution_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        self.usage.cropped_regions += len(crop_rows)
        observation = {
            "ok": bool(crop_rows),
            "crops": crop_rows,
            "failed_crops": failed_crops,
            "usage": self.public_usage(),
        }
        self._record("crop_pages", arguments, observation, artifacts)
        return observation, artifacts

    def artifact_paths(self, handles: list[str]) -> list[Path]:
        return [
            self.image_registry[handle].path
            for handle in handles
            if handle in self.image_registry
        ]

    def context_artifact_handles(
        self, max_images: int = MAX_VIEW_IMAGES_PER_CALL
    ) -> list[str]:
        if max_images <= 0:
            return []
        return self._visible_artifact_batch[-max_images:]

    def validate_evidence_handles(self, handles: list[str]) -> list[str]:
        invalid: list[str] = []
        for handle in handles:
            if handle not in self.handle_space.by_handle:
                invalid.append(handle)
            elif handle not in self.seen_page_handles:
                invalid.append(handle)
        return invalid
