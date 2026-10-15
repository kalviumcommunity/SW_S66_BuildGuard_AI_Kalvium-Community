"""Extract, clean, and validate usable visuals directly from NBC PDFs."""
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
LOG_PATH = OUTPUT_DIR / "image_pipeline.log"
TINY_THRESHOLD = 16
MIN_EFFECTIVE_DPI = 150
REUSED_OCCURRENCE_THRESHOLD = 8
MIN_LOW_RESOLUTION_SIDE = 64
RENDER_DPI = 300
MAX_ANNOTATED_OBJECTS_PER_PAGE = 12
MAX_ASPECT_RATIO = 20.0
MIN_RENDERED_CROP_EDGE = 96
MIN_CROP_CONTRAST = 8.0
    

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
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        return "exclude_thin_fragment", f"aspect_ratio_above_{MAX_ASPECT_RATIO:g}"
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


def _validate_group(group: dict[str, Any], pdf_by_name: dict[str, Path]) -> None:
    """Validate dimensions and placements after source bytes were scanned once."""
    occurrence = next(iter(group.get("occurrences", [])), None)
    source = pdf_by_name.get(str(occurrence.get("source", ""))) if occurrence else None
    if not source:
        group["file_status"] = "source_pdf_not_available"
        return
    # _extract_groups_from_pdf has already extracted these bytes from the open
    # source PDF and computed sha256. Reopening the whole PDF for every group
    # is unnecessary and makes a volume take far longer than it should.
    width, height = int(group.get("width", 0)), int(group.get("height", 0))
    if width <= 0 or height <= 0:
        group["file_status"] = "invalid_image_dimensions"
        return
    group["file_status"] = "valid"

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
            src = html.escape(str(group.get("page_image_path", "")), quote=True)
            refs = "; ".join(
                f"p{o['page']} / xref{o['image_index']}"
                for o in group.get("occurrences", [])[:30]
            )
            if len(group.get("occurrences", [])) > 30:
                refs += f" … (+{len(group['occurrences']) - 30})"
            lines.extend([
                "<article class='card'>",
                (f"<a href='{src}' target='_blank' rel='noopener'><img loading='lazy' src='{src}' alt='Asset {html.escape(group['asset_group_id'])}'></a>" if src else "<div class='meta'><p>No retained page: excluded during validation.</p></div>"),
                "<div class='meta'>",
                f"<p><b>Group:</b> {html.escape(group['asset_group_id'])}</p>",
                f"<p><b>Status:</b> {html.escape(str(group.get('status')))} — {html.escape(str(group.get('reason')))}</p>",
                f"<p><b>Size:</b> {group.get('width')} × {group.get('height')} px</p>",
                f"<p><b>Occurrences:</b> {html.escape(refs)}</p>",
                "</div></article>",
            ])
        lines.append("</div>")

    append_group_cards("accepted", "Accepted visuals")
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
    for status in ("exclude_tiny", "exclude_thin_fragment", "exclude_reused_component", "exclude_off_page_component", "exclude_unreadable_crop", "invalid"):
        append_group_cards(status, status.replace("_", " ").title())
    lines.append("</body></html>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _crop_quality(page: pymupdf.Page, bbox: list[float]) -> tuple[str | None, dict[str, float | int]]:
    """Inspect a 300-DPI crop without writing a crop image to disk."""
    clip = pymupdf.Rect(bbox) & page.rect
    if clip.is_empty or clip.is_infinite:
        return "crop_outside_page", {}
    pix = page.get_pixmap(matrix=pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72), clip=clip, alpha=False)
    short_edge, long_edge = sorted((pix.width, pix.height))
    if short_edge < MIN_RENDERED_CROP_EDGE:
        return f"rendered_crop_shorter_side_below_{MIN_RENDERED_CROP_EDGE}px", {"width": pix.width, "height": pix.height}
    if long_edge / short_edge > MAX_ASPECT_RATIO:
        return f"rendered_crop_aspect_ratio_above_{MAX_ASPECT_RATIO:g}", {"width": pix.width, "height": pix.height}

    # Sample greyscale brightness to reject blank or near-uniform crops. This
    # is deliberately lightweight: it evaluates the temporary in-memory crop,
    # then lets the embedding model receive only meaningful visuals.
    samples = pix.samples
    pixel_count = pix.width * pix.height
    stride = max(1, pixel_count // 2500)
    values: list[float] = []
    for pixel in range(0, pixel_count, stride):
        offset = pixel * pix.n
        values.append((samples[offset] + samples[offset + 1] + samples[offset + 2]) / 3)
    mean = sum(values) / len(values)
    contrast = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    metrics: dict[str, float | int] = {"width": pix.width, "height": pix.height, "contrast": round(contrast, 2)}
    if contrast < MIN_CROP_CONTRAST:
        return f"rendered_crop_contrast_below_{MIN_CROP_CONTRAST:g}", metrics
    return None, metrics


def _render_accepted_pages(groups: list[dict[str, Any]], pdf_paths: list[Path]) -> None:
    """Keep only readable source pages; all candidate crops stay in memory."""
    pdf_by_name = {path.name: path for path in pdf_paths}
    documents = {name: pymupdf.open(path) for name, path in pdf_by_name.items()}
    try:
        for group in groups:
            if group.get("status") not in {"candidate", "low_resolution"}:
                continue
            selected: tuple[dict[str, Any], dict[str, Any], pymupdf.Page, dict[str, float | int]] | None = None
            for occurrence in group.get("occurrences", []):
                document = documents.get(occurrence.get("source", ""))
                page_number = int(occurrence.get("page", 0))
                if document is None or not 1 <= page_number <= document.page_count:
                    continue
                page = document[page_number - 1]
                for placement in occurrence.get("placements", []):
                    if not placement.get("visible_on_page"):
                        continue
                    reason, metrics = _crop_quality(page, placement["bbox_pdf_points"])
                    if reason is None:
                        selected = (occurrence, placement, page, metrics)
                        break
                if selected:
                    break
            if not selected:
                group["status"] = "exclude_unreadable_crop"
                group["reason"] = "rendered_pdf_crop_is_too_small_thin_blank_or_unreadable"
                continue

            occurrence, placement, page, metrics = selected
            volume = str(group.get("volume") or occurrence.get("volume"))
            page_dir = OUTPUT_DIR / volume
            page_dir.mkdir(parents=True, exist_ok=True)
            page_file = page_dir / f"page_{int(occurrence['page']):04d}.png"
            if not page_file.exists():
                page.get_pixmap(matrix=pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72), alpha=False).save(page_file)
            relative_path = f"{volume}/{page_file.name}"
            group.update({
                "status": "accepted",
                "reason": "rendered_pdf_crop_passed_readability_checks",
                "page_number": int(occurrence["page"]),
                "page_image_path": relative_path,
                "high_res_page_path": relative_path,
                "bbox_pdf_points": placement["bbox_pdf_points"],
                "render_dpi": RENDER_DPI,
                "render_scale": RENDER_DPI / 72,
                "rendered_crop_metrics": metrics,
            })
            occurrence["page_image_path"] = relative_path
            occurrence["high_res_page_path"] = relative_path
            occurrence["render_dpi"] = RENDER_DPI
            occurrence["render_scale"] = RENDER_DPI / 72
    finally:
        for document in documents.values():
            document.close()


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
    pdf_by_name = {path.name: path for path in pdf_paths}
    _mark_visible_placements(groups, pdf_paths)
    for group in groups:
        _validate_group(group, pdf_by_name)
        group["min_effective_dpi"] = None
        group["status"], group["reason"] = _classify(group)
    _render_accepted_pages(groups, pdf_paths)

    temp_path = input_path.with_name(input_path.name + ".tmp")
    try:
        _write_jsonl(temp_path, groups)
        temp_path.replace(input_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    volume = str(groups[0].get("volume", "volume_1")) if groups else "volume_1"
    _write_gallery(OUTPUT_DIR / f"image_gallery_{volume}.html", groups)
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
            "exclude_if_aspect_ratio_above": MAX_ASPECT_RATIO,
            "exclude_if_rendered_crop_shorter_side_below_px": MIN_RENDERED_CROP_EDGE,
            "exclude_if_rendered_crop_contrast_below": MIN_CROP_CONTRAST,
            "exclude_if_no_visible_page_placement": True,
            "repeated_component_at_occurrences": REUSED_OCCURRENCE_THRESHOLD,
            "source_pdf_render_dpi": RENDER_DPI,
        },
        "log": str(LOG_PATH),
    }
    report_path = OUTPUT_DIR / f"image_pipeline_report_{volume}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("Validated %d unique images: %s", len(groups), json.dumps(report["groups_by_status"], sort_keys=True))
    logging.info("Report: %s", report_path)
    print(f"Validated {len(groups)} unique images; statuses: {json.dumps(report['groups_by_status'], sort_keys=True)}")
    print(f"Updated chunks: {input_path}")
    print(f"Gallery: {OUTPUT_DIR / f'image_gallery_{volume}.html'}")
    print(f"Report: {report_path}")
    return report


def _volume_name(pdf_path: Path, fallback: int) -> str:
    match = re.search(r"nationalbuilding0?(\d+)", pdf_path.stem, re.I)
    return f"volume_{int(match.group(1)) if match else fallback}"


def _extract_groups_from_pdf(pdf_path: Path, volume: str) -> list[dict[str, Any]]:
    """Create temporary in-memory image groups; no raw image files are saved."""
    groups: dict[str, dict[str, Any]] = {}
    with pymupdf.open(pdf_path) as document:
        logging.info("Scanning %s: %d pages", pdf_path.name, document.page_count)
        image_cache: dict[int, tuple[str, int, int, str]] = {}
        for page in document:
            seen_xrefs: set[int] = set()
            for image_info in page.get_images(full=True):
                xref = int(image_info[0])
                if not xref or xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    if xref not in image_cache:
                        extracted = document.extract_image(xref)
                        image_cache[xref] = (
                            hashlib.sha256(extracted["image"]).hexdigest(),
                            int(extracted.get("width", 0)),
                            int(extracted.get("height", 0)),
                            str(extracted.get("ext", "")),
                        )
                    digest, width, height, extension = image_cache[xref]
                    occurrence = {
                        "source": pdf_path.name,
                        "volume": volume,
                        "page": page.number + 1,
                        "image_index": str(xref),
                        "placements": [
                            {"bbox_pdf_points": [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)]}
                            for rect in page.get_image_rects(xref)
                        ],
                    }
                    group = groups.setdefault(digest, {
                        "id": f"{volume}-asset-{digest[:16]}",
                        "asset_group_id": f"{volume}-asset-{digest[:16]}",
                        "modality": "image",
                        "volume": volume,
                        "sha256": digest,
                        "width": width,
                        "height": height,
                        "source_extension": extension,
                        "occurrences": [],
                    })
                    group["occurrences"].append(occurrence)
                except Exception as exc:
                    logging.warning("Could not inspect %s page %d xref %d: %s", pdf_path.name, page.number + 1, xref, exc)
            if (page.number + 1) % 25 == 0 or page.number + 1 == document.page_count:
                logging.info("Scanned %s: %d/%d pages", pdf_path.name, page.number + 1, document.page_count)
    return list(groups.values())


def clean_pdf(pdf_path: Path, volume: str) -> dict[str, Any]:
    """Validate one PDF and write only accepted visual records plus page images."""
    groups = _extract_groups_from_pdf(pdf_path, volume)
    logging.info("Validating %d unique image groups from %s", len(groups), pdf_path.name)
    _mark_visible_placements(groups, [pdf_path])
    pdf_by_name = {pdf_path.name: pdf_path}
    for group in groups:
        _validate_group(group, pdf_by_name)
        group["min_effective_dpi"] = None
        group["status"], group["reason"] = _classify(group)
    _render_accepted_pages(groups, [pdf_path])
    logging.info("Finished crop-quality checks for %s", pdf_path.name)

    accepted = [group for group in groups if group.get("status") == "accepted"]
    chunks: list[dict[str, Any]] = []
    for record in accepted:
        source = record.get("occurrences", [{}])[0].get("source")
        chunks.append({
            "image_id": str(record["id"]),
            "modality": "image",
            "volume": volume,
            "source_pdf": source,
            "page_number": int(record["page_number"]),
            "page_image_path": record["page_image_path"],
            "high_res_page_path": record["page_image_path"],
            "bbox_pdf_points": record["bbox_pdf_points"],
            "render_dpi": record["render_dpi"],
            "render_scale": record["render_scale"],
            "source_size_px": [record.get("width"), record.get("height")],
            "effective_dpi": record.get("min_effective_dpi"),
            "occurrence_count": len(record.get("occurrences", [])),
            "validation": record["reason"],
            "rendered_crop_metrics": record.get("rendered_crop_metrics", {}),
        })
    chunks_path = OUTPUT_DIR / f"image_chunks_{volume}.jsonl"
    _write_jsonl(chunks_path, chunks)
    gallery_path = OUTPUT_DIR / f"image_gallery_{volume}.html"
    _write_gallery(gallery_path, groups)
    count_by_status: dict[str, int] = defaultdict(int)
    for group in groups:
        count_by_status[str(group.get("status"))] += 1
    report = {
        "pipeline": "clean_validate_images",
        "volume": volume,
        "source_pdf": str(pdf_path),
        "unique_image_groups": len(groups),
        "accepted_images": len(accepted),
        "image_chunks": str(chunks_path),
        "groups_by_status": dict(sorted(count_by_status.items())),
        "gallery": str(gallery_path),
        "saved_page_directory": str(OUTPUT_DIR / volume),
        "validation": {
            "minimum_raw_short_edge_px": TINY_THRESHOLD,
            "maximum_aspect_ratio": MAX_ASPECT_RATIO,
            "minimum_rendered_crop_edge_px": MIN_RENDERED_CROP_EDGE,
            "minimum_rendered_crop_contrast": MIN_CROP_CONTRAST,
            "render_dpi": RENDER_DPI,
        },
    }
    report_path = OUTPUT_DIR / f"image_pipeline_report_{volume}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Validated {len(accepted)} usable images from {pdf_path.name}.")
    print(f"Image chunks: {chunks_path}")
    print(f"Gallery: {gallery_path}")
    print(f"Report: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, nargs="+", default=[DEFAULT_PDF], help="NBC PDF(s) to clean and validate")
    args = parser.parse_args()
    try:
        _configure_logging()
        for index, pdf_path in enumerate(args.pdf, start=1):
            resolved = _resolve_from_root(pdf_path).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Source PDF not found: {resolved}")
            clean_pdf(resolved, _volume_name(resolved, index))
    except Exception as exc:
        logging.exception("Image validation failed: %s", exc)
        print(f"Image validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
