"""Clean, validate, classify, and render previews for extracted image chunks."""
from __future__ import annotations

import argparse
import hashlib
import html
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
DEFAULT_INPUT = OUTPUT_DIR / "image_chunks.jsonl"
LOG_PATH = OUTPUT_DIR / "image_pipeline.log"
TINY_THRESHOLD = 16
MIN_EFFECTIVE_DPI = 150
REUSED_OCCURRENCE_THRESHOLD = 8
MIN_LOW_RESOLUTION_SIDE = 64
RENDER_DPI = 300
RENDER_PADDING_POINTS = 4
MAX_ANNOTATED_OBJECTS_PER_PAGE = 12


def _resolve_from_root(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _read_image_dimensions(path: Path) -> tuple[int, int]:
    pixmap = pymupdf.Pixmap(str(path))
    return int(pixmap.width), int(pixmap.height)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _classify(group: dict[str, Any]) -> tuple[str, str]:
    if group.get("file_status") != "valid":
        return "invalid", str(group.get("file_status", "invalid_file"))
    width, height = int(group.get("width", 0)), int(group.get("height", 0))
    if width <= 0 or height <= 0:
        return "invalid", "invalid_image_dimensions"
    placements = [
        placement
        for occurrence in group.get("occurrences", [])
        for placement in occurrence.get("placements", [])
    ]
    if not placements or not any(p.get("visible_on_page") for p in placements):
        return "exclude_off_page_component", "no_visible_pdf_page_placement"
    if len(group.get("occurrences", [])) >= REUSED_OCCURRENCE_THRESHOLD:
        return "exclude_reused_component", f"same_asset_occurs_at_least_{REUSED_OCCURRENCE_THRESHOLD}_times"
    if min(width, height) < TINY_THRESHOLD:
        return "exclude_tiny", f"shorter_dimension_below_{TINY_THRESHOLD}px"
    if min(width, height) < MIN_LOW_RESOLUTION_SIDE:
        return "low_resolution", f"shorter_dimension_below_{MIN_LOW_RESOLUTION_SIDE}px"
    dpis = [
        placement["effective_dpi"]
        for occurrence in group.get("occurrences", [])
        for placement in occurrence.get("placements", [])
        if placement.get("effective_dpi") is not None
    ]
    group["min_effective_dpi"] = min(dpis) if dpis else None
    if group["min_effective_dpi"] is not None and group["min_effective_dpi"] <= MIN_EFFECTIVE_DPI:
        return "low_resolution", f"effective_resolution_at_or_below_{MIN_EFFECTIVE_DPI}_dpi"
    return "candidate", "file_valid_and_resolution_checks_passed"


def _mark_visible_placements(groups: list[dict[str, Any]], pdf_paths: list[Path]) -> None:
    """Mark image boxes outside the physical PDF page as non-visible components."""
    by_source: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for group in groups:
        for occurrence in group.get("occurrences", []):
            by_source[occurrence.get("source", "")].append((group, occurrence))
    pdf_by_name = {path.name: path for path in pdf_paths}
    for source_name, items in by_source.items():
        pdf_path = pdf_by_name.get(source_name)
        if pdf_path is None:
            for _group, occurrence in items:
                for placement in occurrence.get("placements", []):
                    placement["visible_on_page"] = False
            continue
        with pymupdf.open(pdf_path) as document:
            for _group, occurrence in items:
                page_number = int(occurrence["page"])
                if page_number < 1 or page_number > document.page_count:
                    for placement in occurrence.get("placements", []):
                        placement["visible_on_page"] = False
                    continue
                page_rect = document[page_number - 1].rect
                for placement in occurrence.get("placements", []):
                    rect = pymupdf.Rect(placement["bbox_pdf_points"])
                    visible = rect & page_rect
                    placement["visible_on_page"] = visible.width > 0 and visible.height > 0


def _validate_group(group: dict[str, Any]) -> None:
    file_path = OUTPUT_DIR / group["path"]
    try:
        data = file_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != group.get("sha256"):
            group["file_status"] = "hash_mismatch"
            return
        width, height = _read_image_dimensions(file_path)
        group["width"], group["height"] = width, height
        group["file_size_bytes"] = len(data)
        group["file_status"] = "valid"
    except Exception as exc:
        group["file_status"] = f"image_read_or_decode_error: {type(exc).__name__}: {exc}"
        group["file_size_bytes"] = file_path.stat().st_size if file_path.exists() else None
        return

    for occurrence in group.get("occurrences", []):
        for placement in occurrence.get("placements", []):
            x0, y0, x1, y1 = placement["bbox_pdf_points"]
            width_points, height_points = x1 - x0, y1 - y0
            dpi_x = width / (width_points / 72) if width_points > 0 else 0
            dpi_y = height / (height_points / 72) if height_points > 0 else 0
            placement["effective_dpi"] = round(min(dpi_x, dpi_y), 1) if dpi_x and dpi_y else None


def _configure_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)


def _write_gallery(path: Path, groups: list[dict[str, Any]]) -> None:
    low_pages: dict[tuple[str, int], dict[str, Any]] = {}
    for group in groups:
        if group.get("status") != "low_resolution":
            continue
        for occurrence in group.get("occurrences", []):
            key = (occurrence.get("source", ""), int(occurrence["page"]))
            page_record = low_pages.setdefault(key, {
                "source": occurrence.get("source", ""),
                "page": int(occurrence["page"]),
                "preview": occurrence.get("high_res_page_preview", group["path"]),
                "objects": [],
            })
            page_record["objects"].append({
                "id": group["asset_group_id"],
                "xref": occurrence.get("image_index"),
                "width": group.get("width"),
                "height": group.get("height"),
                "reason": group.get("reason"),
                "placements": [p for p in occurrence.get("placements", []) if p.get("visible_on_page")],
                "crops": occurrence.get("high_res_regions", []),
            })

    lines = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>NBC image review</title><style>",
        "body{font:15px system-ui,sans-serif;margin:24px;background:#f5f6f8;color:#202124}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}",
        ".card{background:white;border:1px solid #ddd;border-radius:8px;padding:10px;overflow:hidden}",
        ".card img{width:100%;height:210px;object-fit:contain;background:#eee}",
        ".meta{font-size:12px;overflow-wrap:anywhere}.meta p{margin:4px 0}",
        "details{margin-top:8px}li{margin:4px 0}",
        "</style></head><body><h1>NBC image review</h1>",
        "<p>Exact duplicate files share one group. Low-resolution assets are shown once per PDF page, with their XREFs and placements listed below.</p>",
    ]

    def append_group_cards(status: str, title: str) -> None:
        selected = [g for g in groups if g.get("status") == status]
        lines.append(f"<h2>{html.escape(title)} ({len(selected)})</h2><div class='grid'>")
        for group in selected:
            src = html.escape(group["path"], quote=True)
            refs = "; ".join(
                f"p{o['page']} / xref{o['image_index']}"
                for o in group.get("occurrences", [])[:30]
            )
            if len(group.get("occurrences", [])) > 30:
                refs += f" … (+{len(group['occurrences']) - 30})"
            lines.extend([
                "<article class='card'>",
                f"<a href='{src}' target='_blank' rel='noopener'><img loading='lazy' src='{src}' alt='Asset {html.escape(group['asset_group_id'])}'></a>",
                "<div class='meta'>",
                f"<p><b>Group:</b> {html.escape(group['asset_group_id'])}</p>",
                f"<p><b>Status:</b> {html.escape(str(group.get('status')))} — {html.escape(str(group.get('reason')))}</p>",
                f"<p><b>Size:</b> {group.get('width')} × {group.get('height')} px</p>",
                f"<p><b>Occurrences:</b> {html.escape(refs)}</p>",
                "</div></article>",
            ])
        lines.append("</div>")

    append_group_cards("candidate", "Candidates")
    lines.append(f"<h2>Low-resolution pages ({len(low_pages)})</h2><div class='grid'>")
    for (_source, _page), record in sorted(low_pages.items()):
        src = html.escape(record["preview"], quote=True)
        details = []
        for obj in record["objects"]:
            box_text = ", ".join(str(p.get("bbox_pdf_points")) for p in obj["placements"])
            details.append(
                "<li><b>" + html.escape(str(obj["id"])) + "</b>: xref "
                + html.escape(str(obj["xref"])) + ", "
                + html.escape(str(obj["width"])) + "×" + html.escape(str(obj["height"]))
                + " px; " + html.escape(str(obj["reason"]))
                + ("; boxes " + html.escape(box_text) if box_text else "; no visible placement box")
                + "</li>"
            )
        lines.extend([
            "<article class='card'>",
            f"<a href='{src}' target='_blank' rel='noopener'><img loading='lazy' src='{src}' alt='PDF page {record['page']}'></a>",
            "<div class='meta'>",
            f"<p><b>PDF page:</b> {record['page']} ({html.escape(record['source'])})</p>",
            f"<p><b>Low-resolution image objects on this page:</b> {len(record['objects'])}</p>",
            "<details><summary>Show XREFs and placements</summary><ul>" + "".join(details) + "</ul></details>",
            "</div></article>",
        ])
    lines.append("</div>")
    append_group_cards("invalid", "Files needing repair")
    lines.append("</body></html>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_low_resolution(groups: list[dict[str, Any]], pdf_paths: list[Path]) -> None:
    pdf_by_name = {path.name: path for path in pdf_paths}
    source_items: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for group in groups:
        if group["status"] != "low_resolution":
            continue
        for occurrence in group.get("occurrences", []):
            source_items[occurrence["source"]].append((group, occurrence))
    for source_name, items in source_items.items():
        source_pdf = pdf_by_name.get(source_name)
        if source_pdf is None or not source_pdf.is_file():
            logging.warning("Source PDF %s is unavailable; skipping high-resolution previews", source_name)
            continue
        with pymupdf.open(source_pdf) as document:
            page_items: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
            for group, occurrence in items:
                page_items[occurrence["page"]].append((group, occurrence))
                occurrence["high_res_regions"] = []
                page = document[occurrence["page"] - 1]
                for placement_index, placement in enumerate(occurrence.get("placements", []), 1):
                    if not placement.get("visible_on_page"):
                        continue
                    clip = pymupdf.Rect(placement["bbox_pdf_points"])
                    clip.x0 -= RENDER_PADDING_POINTS
                    clip.y0 -= RENDER_PADDING_POINTS
                    clip.x1 += RENDER_PADDING_POINTS
                    clip.y1 += RENDER_PADDING_POINTS
                    clip &= page.rect
                    if clip.is_empty or clip.is_infinite:
                        continue
                    region_dir = OUTPUT_DIR / "high_res_regions"
                    region_dir.mkdir(parents=True, exist_ok=True)
                    name = f"v{occurrence['volume']}_p{occurrence['page']}_xref{occurrence['image_index']}_{group['sha256'][:12]}_placement{placement_index}_{RENDER_DPI}dpi.png"
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72), clip=clip, alpha=False)
                    pix.save(region_dir / name)
                    occurrence["high_res_regions"].append({"path": f"high_res_regions/{name}", "dpi": RENDER_DPI, "width": pix.width, "height": pix.height, "placement_index": placement_index})
            page_dir = OUTPUT_DIR / "high_res_pages"
            page_dir.mkdir(parents=True, exist_ok=True)
            for page_number, page_occurrences in page_items.items():
                page = document[page_number - 1]
                # Dense pages become unreadable when every PDF image fragment
                # gets a label. Keep these previews clean and list XREFs in HTML.
                if len(page_occurrences) <= MAX_ANNOTATED_OBJECTS_PER_PAGE:
                    for group, occurrence in page_occurrences:
                        seed = int(group["sha256"][:6], 16)
                        color = (((seed >> 16) & 255) / 255, ((seed >> 8) & 255) / 255, (seed & 255) / 255)
                        for index, placement in enumerate(occurrence.get("placements", []), 1):
                            if not placement.get("visible_on_page"):
                                continue
                            rect = pymupdf.Rect(placement["bbox_pdf_points"])
                            page.draw_rect(rect, color=color, width=1.5, overlay=True)
                            page.insert_text(pymupdf.Point(rect.x0, max(10, rect.y0 - 3)), f"xref {occurrence['image_index']} / {index}", fontsize=7, color=color, overlay=True)
                name = f"v{page_occurrences[0][1]['volume']}_p{page_number}_{RENDER_DPI}dpi.png"
                pix = page.get_pixmap(matrix=pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72), alpha=False)
                pix.save(page_dir / name)
                relative = f"high_res_pages/{name}"
                for _group, occurrence in page_occurrences:
                    occurrence["high_res_page_preview"] = relative
                logging.info("Rendered 300-DPI source page preview %s", page_dir / name)


def validate(input_path: Path, pdf_paths: list[Path]) -> dict[str, Any]:
    _configure_logging()
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Image chunks not found: {input_path}")
    missing_pdfs = [str(p) for p in pdf_paths if not p.is_file()]
    if missing_pdfs:
        raise FileNotFoundError("Missing source PDF(s): " + ", ".join(missing_pdfs))
    groups = [json.loads(line) for line in input_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    seen_ids: set[str] = set()
    for index, group in enumerate(groups, 1):
        if not isinstance(group, dict) or not isinstance(group.get("occurrences"), list):
            raise ValueError(f"Image record {index} must be an object with an occurrences array")
        asset_id = group.get("asset_group_id") or group.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError(f"Image record {index} has no valid group ID")
        if asset_id in seen_ids:
            raise ValueError(f"Duplicate image group ID: {asset_id}")
        seen_ids.add(asset_id)
        group["id"] = asset_id
        group["asset_group_id"] = asset_id
        group["modality"] = "image"
    _mark_visible_placements(groups, pdf_paths)
    for group in groups:
        _validate_group(group)
        group["min_effective_dpi"] = None
        group["status"], group["reason"] = _classify(group)
    _render_low_resolution(groups, pdf_paths)

    temp_path = input_path.with_name(input_path.name + ".tmp")
    try:
        _write_jsonl(temp_path, groups)
        temp_path.replace(input_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    _write_gallery(OUTPUT_DIR / "image_gallery.html", groups)
    count_by_status: dict[str, int] = defaultdict(int)
    occurrences_by_status: dict[str, int] = defaultdict(int)
    for group in groups:
        count_by_status[group["status"]] += 1
        occurrences_by_status[group["status"]] += len(group.get("occurrences", []))
    report = {
        "pipeline": "clean_validate_images",
        "version": "1.0",
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_and_output": str(input_path),
        "source_pdfs": [str(p) for p in pdf_paths],
        "unique_images": len(groups),
        "image_occurrences": sum(len(g.get("occurrences", [])) for g in groups),
        "groups_by_status": dict(sorted(count_by_status.items())),
        "occurrences_by_status": dict(sorted(occurrences_by_status.items())),
        "thresholds": {
            "tiny_if_shorter_side_below_px": TINY_THRESHOLD,
            "low_resolution_if_shorter_side_below_px": MIN_LOW_RESOLUTION_SIDE,
            "low_resolution_if_effective_dpi_at_or_below": MIN_EFFECTIVE_DPI,
            "exclude_if_no_visible_page_placement": True,
            "repeated_component_at_occurrences": REUSED_OCCURRENCE_THRESHOLD,
            "source_pdf_render_dpi": RENDER_DPI,
        },
        "log": str(LOG_PATH),
    }
    report_path = OUTPUT_DIR / "image_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("Validated %d unique images: %s", len(groups), json.dumps(report["groups_by_status"], sort_keys=True))
    logging.info("Report: %s", report_path)
    print(f"Validated {len(groups)} unique images; statuses: {json.dumps(report['groups_by_status'], sort_keys=True)}")
    print(f"Updated chunks: {input_path}")
    print(f"Gallery: {OUTPUT_DIR / 'image_gallery.html'}")
    print(f"Report: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Grouped image JSONL from src.images.chunk_images")
    parser.add_argument("--pdf", type=Path, nargs="+", default=[DEFAULT_PDF], help="Source PDF(s), same order/files used for extraction")
    args = parser.parse_args()
    try:
        validate(_resolve_from_root(args.input), [p.resolve() for p in args.pdf])
    except Exception as exc:
        logging.exception("Image validation failed: %s", exc)
        print(f"Image validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
