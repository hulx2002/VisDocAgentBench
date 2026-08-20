from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from dataset_tools.validate_dataset import validate_page_images


def page_row(path: Path, *, width: int = 12, height: int = 8) -> dict[str, object]:
    return {
        "page_id": "page_1",
        "image_path": path.name,
        "width": width,
        "height": height,
    }


def test_dataset_validator_decodes_png_and_checks_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "page_1.png"
    Image.new("RGB", (12, 8), color="white").save(image_path)
    validate_page_images([page_row(image_path)], tmp_path)

    with pytest.raises(ValueError, match="dimensions differ"):
        validate_page_images([page_row(image_path, width=11)], tmp_path)


def test_dataset_validator_rejects_corrupt_png(tmp_path: Path) -> None:
    image_path = tmp_path / "page_1.png"
    image_path.write_bytes(b"not a png")
    with pytest.raises(ValueError, match="Could not decode"):
        validate_page_images([page_row(image_path)], tmp_path)
