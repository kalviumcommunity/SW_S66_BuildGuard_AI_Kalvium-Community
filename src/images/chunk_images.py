"""Extract and group image assets from selected NBC source PDFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = ROOT / "data" / "nationalbuilding01.pdf"
OUTPUT_DIR = ROOT / "outputs" / "images"
ASSET_DIR = OUTPUT_DIR / "assets"
LOG_PATH = OUTPUT_DIR / "image_extraction.log"


def _safe_extension(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower()) or "bin"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _configure_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    for handler in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root_logger.addHandler(handler)


def extract(pdf_paths: list[Path]) -> dict[str, Any]:
    _configure_logging()
    missing = [str(path) for path in pdf_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing PDF(s): " + ", ".join(missing))
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict[str, Any]] = {}
    source_pages = 0
    for fallback_volume, pdf_path in enumerate(pdf_paths, 1):
        volume_match = re.search(r"nationalbuilding0?(\d+)", pdf_path.stem, re.I)
        volume = int(volume_match.group(1)) if volume_match else fallback_volume
        with pymupdf.open(pdf_path) as document:
            for page in document:
                source_pages += 1
                seen_xrefs: set[int] = set()
                for image_info in page.get_images(full=True):
                    xref = int(image_info[0])
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        extracted = document.extract_image(xref)
                        data = extracted["image"]
                        digest = hashlib.sha256(data).hexdigest()
                        relative_path = f"assets/{digest}.{_safe_extension(extracted.get('ext', 'png'))}"
                        target = OUTPUT_DIR / relative_path
                        if not target.exists():
                            target.write_bytes(data)
                        width = int(extracted.get("width", 0))
                        height = int(extracted.get("height", 0))
                        rects = page.get_image_rects(xref)
                        occurrence = {
                            "id": f"img-v{volume}-p{page.number + 1}-xref{xref}-{digest[:12]}",
                            "source": pdf_path.name,
                            "volume": volume,
                            "page": page.number + 1,
                            "image_index": str(xref),
                            "path": relative_path,
                            "width": width,
                            "height": height,
                            "sha256": digest,
                            "placements": [
                                {"bbox_pdf_points": [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)]}
                                for r in rects
                            ],
                        }
                        group = groups.setdefault(digest, {
                            "id": f"asset-{digest[:16]}",
                            "asset_group_id": f"asset-{digest[:16]}",
                            "modality": "image",
                            "sha256": digest,
                            "path": relative_path,
                            "width": width,
                            "height": height,
                            "file_status": "pending_validation",
                            "status": "pending_validation",
                            "occurrences": [],
                        })
                        group["occurrences"].append(occurrence)
                    except Exception as exc:
                        logging.warning("Could not extract %s page %d xref %d: %s", pdf_path.name, page.number + 1, xref, exc)
                if page.number % 100 == 99:
                    logging.info("Scanned %s: %d/%d pages", pdf_path.name, page.number + 1, document.page_count)
    records = list(groups.values())
    _write_jsonl(OUTPUT_DIR / "image_chunks.jsonl", records)
    summary = {
        "status": "extracted",
        "source_pdfs": [str(p) for p in pdf_paths],
        "source_pages_scanned": source_pages,
        "unique_image_groups": len(records),
        "image_occurrences": sum(len(g["occurrences"]) for g in records),
    }
    logging.info("Extracted %d image occurrences into %d unique groups", summary["image_occurrences"], len(records))
    logging.info("Run image validation next: uv run python -m src.images.clean_validate_images")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, nargs="+", default=[DEFAULT_PDF], help="PDF(s) to scan")
    args = parser.parse_args()
    try:
        summary = extract([path.resolve() for path in args.pdf])
        print(f"Extracted {summary['image_occurrences']} placements into {summary['unique_image_groups']} unique image groups.")
        print(f"Raw grouped image records: {OUTPUT_DIR / 'image_chunks.jsonl'}")
        print("Next: uv run python -m src.images.clean_validate_images")
    except Exception as exc:
        logging.exception("Image extraction failed: %s", exc)
        print(f"Image extraction failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
