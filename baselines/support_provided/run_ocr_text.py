#!/usr/bin/env python3
"""Run the OCR-text agent with complete support context shown at step zero."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
TEXT_DIR = PROJECT_ROOT / "baselines/agent_ocr_text"
for module_dir in (THIS_DIR, TEXT_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import run as runner  # noqa: E402
from support_context import install_support_provided_adapter  # noqa: E402


def main() -> int:
    install_support_provided_adapter(runner, route="ocr_text")
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
