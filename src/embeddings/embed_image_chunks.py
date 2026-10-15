"""Embed validated NBC images in a separate Chroma collection.

The script accepts a final per-volume ``outputs/images/image_chunks_volume_N.jsonl`` file.  An
image record may either refer to a retained image file, or (preferred) to a
high-resolution rendered PDF page plus ``bbox_pdf_points``.  In the latter
case the crop is made in memory for embedding and is never written to disk.

Run from the repository root:
    uv run python -m src.embeddings.embed_image_chunks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "images" / "image_chunks_volume_1.jsonl"
IMAGES_OUTPUT = PROJECT_ROOT / "outputs" / "images"
DEFAULT_DB = PROJECT_ROOT / "outputs" / "vector_db"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "embeddings" / "image_embedding_report.json"
DEFAULT_LOG = PROJECT_ROOT / "outputs" / "embeddings" / "image_embedding.log"
DEFAULT_COLLECTION = "nbc_v1_clip_images"
DEFAULT_MODEL = "sentence-transformers/clip-ViT-B-32"


@dataclass
class Candidate:
    identifier: str
    record: dict[str, Any]
    image_path: Path | None
    page_path: Path | None
    bbox: tuple[float, float, float, float] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--volume", default="volume_1", help="Fallback volume label for legacy chunk records.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed IDs already present in the image collection.",
    )
    return parser.parse_args()


def path_from_record(record: dict[str, Any], keys: Iterable[str]) -> Path | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value)
            if path.is_absolute():
                return path
            project_path = PROJECT_ROOT / path
            # Image chunk paths are relative to outputs/images; older records
            # can still use project-relative paths.
            return project_path if project_path.exists() else IMAGES_OUTPUT / path
    return None


def bbox_from_record(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = record.get("bbox_pdf_points") or record.get("bbox") or record.get("bounding_box")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, dict):
        value = [value.get(key) for key in ("x0", "y0", "x1", "y1")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def record_is_valid(record: dict[str, Any]) -> bool:
    """Respect validation decisions from either old or new image JSONL formats."""
    if record.get("excluded") is True or record.get("is_valid") is False:
        return False
    status = str(record.get("status", "")).strip().lower()
    return not status.startswith("exclude") and status not in {"excluded", "invalid", "rejected", "tiny", "duplicate"}


def load_candidates(input_path: Path, fallback_volume: str) -> tuple[list[Candidate], list[str]]:
    candidates: list[Candidate] = []
    warnings: list[str] = []
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(record, dict) or not record_is_valid(record):
                continue
            # The chunker stores one record per unique image group and keeps
            # page placements inside ``occurrences``.  Flatten its best visible
            # placement into the record used for embedding.
            occurrences = record.get("occurrences")
            if isinstance(occurrences, list) and occurrences:
                visible: list[tuple[dict[str, Any], dict[str, Any]]] = []
                for occurrence in occurrences:
                    if not isinstance(occurrence, dict):
                        continue
                    for placement in occurrence.get("placements", []):
                        if isinstance(placement, dict) and placement.get("visible_on_page", True):
                            visible.append((occurrence, placement))
                if not visible:
                    warnings.append(f"line {line_number}: no visible page placement")
                    continue
                occurrence, placement = next(
                    ((item, box) for item, box in visible if item.get("high_res_page_path")),
                    visible[0],
                )
                record = {**record, **occurrence}
                record["bbox_pdf_points"] = placement.get("bbox_pdf_points")
                record["page_number"] = occurrence.get("page")
            record.setdefault("volume", fallback_volume)
            raw_id = record.get("image_id") or record.get("id") or record.get("group_id")
            identifier = str(raw_id) if raw_id else f"image-{line_number:06d}"
            image_path = path_from_record(
                record,
                ("image_path", "file_path", "asset_path", "path"),
            )
            page_path = path_from_record(record, ("high_res_page_path", "page_image_path"))
            bbox = bbox_from_record(record)
            if not (image_path and image_path.exists()) and not (page_path and page_path.exists() and bbox):
                warnings.append(f"{identifier}: no usable image file or page crop")
                continue
            candidates.append(Candidate(identifier, record, image_path, page_path, bbox))
    return candidates, warnings


def open_candidate(candidate: Candidate):
    from PIL import Image

    # Recreate the crop from the retained page. No individual crop is saved.
    if candidate.page_path and candidate.page_path.exists() and candidate.bbox:
        with Image.open(candidate.page_path) as source:
            # PDF coordinates are points.  The renderer records either page
            # size in points or a scale/DPI so the bounding box maps to pixels.
            page_width_pt = candidate.record.get("page_width_pdf_points")
            page_height_pt = candidate.record.get("page_height_pdf_points")
            scale = candidate.record.get("render_scale")
            if page_width_pt and page_height_pt:
                sx = source.width / float(page_width_pt)
                sy = source.height / float(page_height_pt)
            elif scale:
                sx = sy = float(scale)
            else:
                # High-res pages are normally rendered at 300 DPI = 300/72 scale.
                dpi = float(candidate.record.get("render_dpi", 300))
                sx = sy = dpi / 72.0
            x0, y0, x1, y1 = candidate.bbox
            crop = source.crop((round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)))
            return crop.convert("RGB").copy()

    assert candidate.image_path is not None
    with Image.open(candidate.image_path) as source:
        return source.convert("RGB").copy()


def source_signature(image) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.size).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def metadata_for(candidate: Candidate, signature: str) -> dict[str, str | int | float | bool]:
    record = candidate.record
    metadata: dict[str, str | int | float | bool] = {
        "volume": str(record.get("volume", "volume_1")),
        "source_signature": signature,
    }
    for source_key, target_key in (
        ("page", "page"),
        ("page_number", "page_number"),
        ("pdf_page", "pdf_page"),
        ("image_type", "image_type"),
        ("group_id", "group_id"),
    ):
        value = record.get(source_key)
        if isinstance(value, (str, int, float, bool)):
            metadata[target_key] = value
    if candidate.page_path:
        metadata["high_res_page_path"] = str(candidate.page_path.relative_to(PROJECT_ROOT))
    if candidate.bbox:
        metadata["bbox_pdf_points"] = json.dumps(candidate.bbox)
    return metadata


def existing_ids(collection, identifiers: list[str]) -> set[str]:
    if not identifiers:
        return set()
    # Chroma limits a single get call; 500 is well below current defaults.
    found: set[str] = set()
    for index in range(0, len(identifiers), 500):
        result = collection.get(ids=identifiers[index : index + 500], include=[])
        found.update(result.get("ids", []))
    return found


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if not args.input.exists():
        raise SystemExit(f"Image chunk file not found: {args.input}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import chromadb
    from sentence_transformers import SentenceTransformer

    candidates, warnings = load_candidates(args.input, args.volume)
    client = chromadb.PersistentClient(path=str(args.db_path))
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine", "embedding_model": args.model, "modality": "image"},
    )
    present = set() if args.force else existing_ids(collection, [item.identifier for item in candidates])
    pending = [item for item in candidates if item.identifier not in present]

    model = SentenceTransformer(args.model)
    generated = skipped_quality = failed = 0
    failures: list[str] = []
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        images, usable = [], []
        for candidate in batch:
            try:
                image = open_candidate(candidate)
                short_edge, long_edge = sorted(image.size)
                # Safety net in case an old JSONL still contains a PDF fragment.
                if short_edge < 64 or long_edge / max(short_edge, 1) > 20:
                    skipped_quality += 1
                    continue
                images.append(image)
                usable.append(candidate)
            except Exception as exc:  # individual bad image must not stop the run
                failed += 1
                failures.append(f"{candidate.identifier}: {type(exc).__name__}: {exc}")
        if not usable:
            continue
        try:
            vectors = model.encode(images, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=False)
            signatures = [source_signature(image) for image in images]
            collection.upsert(
                ids=[item.identifier for item in usable],
                embeddings=vectors.tolist(),
                metadatas=[metadata_for(item, signature) for item, signature in zip(usable, signatures)],
                documents=["NBC visual reference" for _ in usable],
            )
            generated += len(usable)
        except Exception as exc:
            failed += len(usable)
            failures.extend(f"{item.identifier}: batch error {type(exc).__name__}: {exc}" for item in usable)

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(args.input),
        "collection": args.collection,
        "database_path": str(args.db_path),
        "model": args.model,
        "batch_size": args.batch_size,
        "valid_input_records": len(candidates),
        "embedded": generated,
        "already_embedded_skipped": len(present),
        "quality_skipped": skipped_quality,
        "failed": failed,
        "collection_count": collection.count(),
        "warnings": warnings,
        "failures": failures,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"warnings", "failures"}}, indent=2))
    print(f"Report: {args.report}")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
