"""QA artifacts and explicit human-review gates for GPT head generation runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

REVIEW_COLUMNS = [
    "custom_id", "filename", "bin", "pitch", "yaw", "cam",
    "photorealism", "intent_match", "framing", "roll_no_back", "body_integrity",
    "notes", "reviewed_sha256",
]
VALID_BINARY = {"pass", "fail"}
VALID_INTENT = {"match", "off-by-one-bin", "wrong"}

DEFAULT_QA_POLICY: dict[str, Any] = {
    "detector_score_threshold": 0.25,
    "head_height_ratio": {"min": 0.25, "max": 0.50},
    "margin_min_head_ratio": 0.50,
    "require_single_head": True,
    "require_body_detection": True,
    "reject_back": True,
    "reject_duplicates": True,
}

_FACE_CLASS = 16
_EYE_CLASS = 17
_NOSE_CLASS = 18
_MOUTH_CLASS = 19
_EAR_CLASS = 20


class ReviewError(RuntimeError):
    """A human-review workflow error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "batch_state.json"
    if not path.exists():
        raise ReviewError(f"batch state not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _read_plan(run_dir: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with (run_dir / state["plan_path"]).open(encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
    return records


def _iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _detector_annotations(run_dir: Path, records: list[dict[str, Any]],
                          model_path: Path, score_threshold: float) -> dict[str, dict[str, Any]]:
    if not model_path.exists():
        return {record["custom_id"]: {"detector_status": "model_missing"} for record in records}
    import cv2
    from ..pseudolabel.deimv2 import CLASS_BODY, CLASS_HEAD, DIR8_CLASSES, Deimv2Detector

    providers = (["CPUExecutionProvider"]
                 if os.environ.get("HRFFA_DEIMV2_CPU") == "1" else None)
    detector = Deimv2Detector(
        model_path, providers=providers, score_threshold=score_threshold
    )
    result: dict[str, dict[str, Any]] = {}
    present = [record for record in records if (run_dir / "images" / record["filename"]).exists()]
    for start in range(0, len(present), 8):
        chunk = present[start:start + 8]
        images = [cv2.imread(str(run_dir / "images" / record["filename"])) for record in chunk]
        valid_pairs = [(record, image) for record, image in zip(chunk, images) if image is not None]
        if not valid_pairs:
            continue
        detections = detector.infer_batch([image for _, image in valid_pairs])
        for (record, image), dets in zip(valid_pairs, detections):
            height, width = image.shape[:2]
            heads = [det for det in dets if int(det[0]) == CLASS_HEAD]
            if not heads:
                result[record["custom_id"]] = {
                    "detector_status": "no_head", "head_count": 0, "body_count": 0,
                    "direction": None,
                }
                continue
            head = max(heads, key=lambda det: det[5])
            box = [float(value) for value in head[1:5]]
            head_width, head_height = box[2] - box[0], box[3] - box[1]
            bodies = [det for det in dets if int(det[0]) == CLASS_BODY]
            body = max(bodies, key=lambda det: det[5]) if bodies else None
            directions = [det for det in dets if int(det[0]) in DIR8_CLASSES]
            direction = max(directions, key=lambda det: _iou(box, det[1:5])) if directions else None
            def feature_inside(det: Any) -> bool:
                center_x = (float(det[1]) + float(det[3])) / 2
                center_y = (float(det[2]) + float(det[4])) / 2
                return (box[0] <= center_x <= box[2]
                        and box[1] <= center_y <= box[3]
                        and float(det[5]) >= 0.6)

            result[record["custom_id"]] = {
                "detector_status": "ok",
                "head_count": len(heads),
                "body_count": len(bodies),
                "body_box_xyxy": (
                    [round(float(value), 2) for value in body[1:5]] if body else None
                ),
                "body_score": body[5] if body else None,
                "head_box_xyxy": [round(value, 2) for value in box],
                "head_score": head[5],
                "head_height_ratio": round(head_height / height, 4),
                "margin_left_head_ratio": round(box[0] / head_width, 4) if head_width else 0,
                "margin_right_head_ratio": round((width - box[2]) / head_width, 4) if head_width else 0,
                "margin_top_head_ratio": round(box[1] / head_height, 4) if head_height else 0,
                "margin_bottom_head_ratio": round((height - box[3]) / head_height, 4) if head_height else 0,
                "head_size_reference_pass": 0.25 <= head_height / height <= 0.50,
                "margin_reference_pass": all(value >= 0.5 for value in [
                    box[0] / head_width if head_width else 0,
                    (width - box[2]) / head_width if head_width else 0,
                    box[1] / head_height if head_height else 0,
                    (height - box[3]) / head_height if head_height else 0,
                ]),
                "direction": DIR8_CLASSES[int(direction[0])] if direction else None,
                "direction_score": direction[5] if direction else None,
                "back_reference": bool(direction and DIR8_CLASSES[int(direction[0])] == "back"),
                "has_face": any(
                    int(det[0]) == _FACE_CLASS and feature_inside(det) for det in dets
                ),
                "has_eye": any(
                    int(det[0]) == _EYE_CLASS and feature_inside(det) for det in dets
                ),
                "has_nose": any(
                    int(det[0]) == _NOSE_CLASS and feature_inside(det) for det in dets
                ),
                "has_mouth": any(
                    int(det[0]) == _MOUTH_CLASS and feature_inside(det) for det in dets
                ),
                "has_ear": any(
                    int(det[0]) == _EAR_CLASS and feature_inside(det) for det in dets
                ),
            }
    return result


def quality_gate_reasons(row: dict[str, Any], policy: dict[str, Any] | None = None) -> tuple[list[str], bool]:
    """Return hard-QA rejection reasons and whether every required check ran.

    Body detection catches gross omissions only.  It deliberately does not claim to
    detect texture melting or implausible occlusion; those remain in human_review.csv.
    """
    cfg = {**DEFAULT_QA_POLICY, **(policy or {})}
    reasons: list[str] = []
    image_valid = row.get("image_valid", row.get("png_valid", False))
    if not row.get("exists"):
        reasons.append("image_missing")
    elif not image_valid:
        reasons.append("invalid_image")
    elif not row.get("dimension_match"):
        reasons.append("wrong_dimensions")
    if cfg["reject_duplicates"] and row.get("duplicate_of"):
        reasons.append("duplicate_image")

    detector_status = row.get("detector_status")
    detector_complete = detector_status in {"ok", "no_head"}
    if detector_status == "no_head":
        reasons.append("head_not_detected")
    elif detector_status == "ok":
        if cfg["require_single_head"] and row.get("head_count") != 1:
            reasons.append("head_count_not_one")
        ratio = row.get("head_height_ratio")
        limits = cfg["head_height_ratio"]
        if ratio is None:
            reasons.append("head_size_unavailable")
        elif ratio < float(limits["min"]):
            reasons.append("head_too_small")
        elif ratio > float(limits["max"]):
            reasons.append("head_too_large")
        margin_limit = float(cfg["margin_min_head_ratio"])
        margins = [
            row.get("margin_left_head_ratio"), row.get("margin_right_head_ratio"),
            row.get("margin_top_head_ratio"), row.get("margin_bottom_head_ratio"),
        ]
        if any(value is None or float(value) < margin_limit for value in margins):
            reasons.append("insufficient_margin")
        if (cfg["reject_back"] and row.get("back_reference")
                and not any(row.get(key) for key in (
                    "has_face", "has_eye", "has_nose", "has_mouth", "has_ear"
                ))):
            reasons.append("back_of_head")
        if cfg["require_body_detection"] and not row.get("body_count"):
            reasons.append("body_not_detected")
    if row.get("intent_match_auto_hard_failure"):
        if row.get("intent_match_auto") == "off-by-one-bin":
            reasons.append("auto_intent_off_by_one")
        elif row.get("intent_match_auto") == "wrong":
            reasons.append("auto_intent_wrong")
    if row.get("roll_no_back_auto_hard_failure"):
        reasons.append("auto_roll_or_back")
    return reasons, detector_complete


def run_auto_qa(run_dir: Path, use_detector: bool = True,
                model_path: Path = Path("data/models/deimv2_wholebody49_boxes_only.onnx"),
                custom_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Run hard QA, reusing detector results for images whose bytes did not change.

    ``custom_ids`` is the set replaced since the last successful QA pass.  When
    the previous manifest is complete, only those records are sent through the
    detector.  Cheap, global duplicate/reason evaluation is still recomputed in
    plan order.  A missing or incomplete prior manifest fails safely to a full
    detector pass.
    """
    state = _read_state(run_dir)
    records = _read_plan(run_dir, state)
    policy = {**DEFAULT_QA_POLICY, **state.get("auto_correction", {})}
    qa_path = run_dir / "auto_qa.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if custom_ids is not None and qa_path.exists():
        try:
            with qa_path.open(encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    existing[row["custom_id"]] = row
        except (KeyError, json.JSONDecodeError, OSError):
            existing = {}
    planned_ids = {record["custom_id"] for record in records}
    incremental = (
        custom_ids is not None
        and set(existing) == planned_ids
        and set(custom_ids) <= planned_ids
    )
    refresh_ids = set(custom_ids or ()) if incremental else planned_ids
    detector_records = [
        record for record in records if record["custom_id"] in refresh_ids
    ]
    detector = _detector_annotations(
        run_dir, detector_records, model_path, float(policy["detector_score_threshold"])
    ) if use_detector else {}
    results: list[dict[str, Any]] = []
    for record in records:
        custom_id = record["custom_id"]
        if incremental and custom_id not in refresh_ids:
            results.append(dict(existing[custom_id]))
            continue
        path = run_dir / "images" / record["filename"]
        row: dict[str, Any] = {
            "custom_id": custom_id,
            "filename": record["filename"],
            "expected_size": record["size"],
            "exists": path.exists(),
            "image_valid": False,
            "storage_format": None,
            "dimension_match": False,
            "actual_size": None,
            "sha256": None,
            "duplicate_of": None,
        }
        if path.exists():
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    expected_format = "JPEG" if path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
                    row["image_valid"] = image.format == expected_format
                    row["storage_format"] = image.format
                    row["actual_size"] = f"{image.width}x{image.height}"
                row["dimension_match"] = row["actual_size"] == record["size"]
                digest = _sha256(path)
                row["sha256"] = digest
            except (OSError, UnidentifiedImageError):
                pass
        row.update(detector.get(custom_id, {
            "detector_status": "skipped" if not use_detector else "image_missing"
        }))
        results.append(row)

    # Replacing one image can add or remove a duplicate relationship for a later
    # untouched image, so duplicate assignment and hard-gate reasons remain a
    # global (but detector-free) pass.
    hashes: dict[str, str] = {}
    for row in results:
        digest = row.get("sha256")
        row["duplicate_of"] = hashes.get(digest) if digest else None
        if digest:
            hashes.setdefault(digest, row["custom_id"])
        reasons, complete = quality_gate_reasons(row, policy)
        row["quality_gate_complete"] = complete
        row["quality_gate_pass"] = complete and not reasons
        row["quality_gate_reasons"] = reasons
    _write_jsonl(qa_path, results)
    return results


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        return ImageFont.load_default()


def _contact_sheet(path: Path, records: list[dict[str, Any]], run_dir: Path,
                   columns: int = 5, tile_width: int = 256, tile_height: int = 220) -> None:
    label_height = 48
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font()
    for index, record in enumerate(records):
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        image_path = run_dir / "images" / record["filename"]
        if image_path.exists():
            try:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                    image.thumbnail((tile_width - 8, tile_height - 8), Image.Resampling.LANCZOS)
                    px = x + (tile_width - image.width) // 2
                    py = y + (tile_height - image.height) // 2
                    sheet.paste(image, (px, py))
            except (OSError, UnidentifiedImageError):
                draw.rectangle((x + 4, y + 4, x + tile_width - 4, y + tile_height - 4), fill="#ffd6d6")
        else:
            draw.rectangle((x + 4, y + 4, x + tile_width - 4, y + tile_height - 4), fill="#eeeeee")
        draw.text((x + 6, y + tile_height + 2), f"#{record['serial']} {record['bin']}", fill="black", font=font)
        draw.text((x + 6, y + tile_height + 23),
                  f"p{record['pitch']:+d} y{record['yaw']:+d} c{record['cam']:+d}",
                  fill="black", font=font)
    sheet.save(path, format="JPEG", quality=90, optimize=True)


def _overview_records(records: list[dict[str, Any]], count: int = 9) -> list[dict[str, Any]]:
    """Select deterministic, square-source representatives across pose bins."""
    if len(records) <= count:
        return records
    square = []
    for record in records:
        try:
            width, height = (int(value) for value in record["size"].split("x"))
        except (KeyError, ValueError):
            continue
        if 0.90 <= width / height <= 1.10:
            square.append(record)
    candidates = square if len(square) >= count else records
    bins: dict[str, list[dict[str, Any]]] = {}
    for record in candidates:
        bins.setdefault(str(record.get("bin", "unknown")), []).append(record)
    selected = [group[len(group) // 2] for group in bins.values() if group]
    if len(selected) > count:
        selected = selected[:count]
    selected_ids = {record["custom_id"] for record in selected}
    remaining = [record for record in candidates if record["custom_id"] not in selected_ids]
    while len(selected) < count and remaining:
        record = remaining.pop(len(remaining) // 2)
        selected.append(record)
    return selected


def regenerate_overview_contact_sheet(run_dir: Path) -> dict[str, Any]:
    state = _read_state(run_dir)
    records = _read_plan(run_dir, state)
    present = [record for record in records if (run_dir / "images" / record["filename"]).exists()]
    if not present:
        raise ReviewError("no collected images exist for the overview contact sheet")
    overview_records = _overview_records(present, count=9)
    path = run_dir / "contact_sheet.jpg"
    _contact_sheet(
        path, overview_records, run_dir,
        columns=3, tile_width=300, tile_height=252,
    )
    with Image.open(path) as image:
        dimensions = f"{image.width}x{image.height}"
    return {
        "contact_sheet": str(path),
        "images": len(overview_records),
        "dimensions": dimensions,
        "custom_ids": [record["custom_id"] for record in overview_records],
        "bins": [record["bin"] for record in overview_records],
        "source_sizes": [record["size"] for record in overview_records],
    }


def make_contact_sheets(run_dir: Path, records: list[dict[str, Any]],
                        custom_ids: set[str] | None = None) -> list[Path]:
    present = [record for record in records if (run_dir / "images" / record["filename"]).exists()]
    if not present:
        return []
    changed = set(custom_ids or ())
    paths: list[Path] = []
    if len(present) <= 500:
        path = run_dir / "contact_sheet.jpg"
        if custom_ids is None or not path.exists() or changed.intersection(
            record["custom_id"] for record in present
        ):
            _contact_sheet(path, present, run_dir)
        return [path]
    for index, start in enumerate(range(0, len(present), 500)):
        path = run_dir / f"contact_sheet_{index:03d}.jpg"
        chunk = present[start:start + 500]
        if custom_ids is None or not path.exists() or changed.intersection(
            record["custom_id"] for record in chunk
        ):
            _contact_sheet(path, chunk, run_dir)
        paths.append(path)
    # The conventional single sheet is a deterministic 3x3 overview. Prefer
    # square source images and cover every pose bin before adding a ninth view.
    overview = run_dir / "contact_sheet.jpg"
    overview_records = _overview_records(present, count=9)
    if custom_ids is None or not overview.exists() or changed.intersection(
        record["custom_id"] for record in overview_records
    ):
        _contact_sheet(
            overview, overview_records, run_dir,
            columns=3, tile_width=300, tile_height=252,
        )
    return [overview, *paths]


def prepare_review_csv(run_dir: Path, records: list[dict[str, Any]],
                       custom_ids: set[str] | None = None) -> Path:
    path = run_dir / "human_review.csv"
    existing: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                existing[row["custom_id"]] = row
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            old = existing.get(record["custom_id"], {})
            if custom_ids is not None and record["custom_id"] not in custom_ids and old:
                writer.writerow(old)
                continue
            image_path = run_dir / "images" / record["filename"]
            current_sha = _sha256(image_path) if image_path.exists() else ""
            keep_review = old.get("reviewed_sha256") == current_sha and bool(current_sha)
            writer.writerow({
                "custom_id": record["custom_id"],
                "filename": record["filename"],
                "bin": record["bin"],
                "pitch": record["pitch"],
                "yaw": record["yaw"],
                "cam": record["cam"],
                "photorealism": old.get("photorealism", "") if keep_review else "",
                "intent_match": old.get("intent_match", "") if keep_review else "",
                "framing": old.get("framing", "") if keep_review else "",
                "roll_no_back": old.get("roll_no_back", "") if keep_review else "",
                "body_integrity": old.get("body_integrity", "") if keep_review else "",
                "notes": old.get("notes", "") if keep_review else "",
                "reviewed_sha256": current_sha,
            })
    return path


def prepare_review_artifacts(run_dir: Path, use_detector: bool = True,
                             custom_ids: set[str] | None = None) -> dict[str, Any]:
    state = _read_state(run_dir)
    records = _read_plan(run_dir, state)
    qa = run_auto_qa(run_dir, use_detector=use_detector, custom_ids=custom_ids)
    sheets = make_contact_sheets(run_dir, records, custom_ids=custom_ids)
    review = prepare_review_csv(run_dir, records, custom_ids=custom_ids)
    return {
        "auto_qa": str(run_dir / "auto_qa.jsonl"),
        "review_csv": str(review),
        "contact_sheets": [str(path) for path in sheets],
        "valid_images": sum(row["image_valid"] and row["dimension_match"] for row in qa),
        "duplicates": sum(row["duplicate_of"] is not None for row in qa),
        "quality_pass": sum(row["quality_gate_pass"] for row in qa),
        "quality_failed": sum(
            row["quality_gate_complete"] and not row["quality_gate_pass"] for row in qa
        ),
        "quality_incomplete": sum(not row["quality_gate_complete"] for row in qa),
        "detector_images_evaluated": len(records) if custom_ids is None else len(custom_ids),
    }


def summarize_review(run_dir: Path) -> dict[str, Any]:
    state = _read_state(run_dir)
    plan = _read_plan(run_dir, state)
    path = run_dir / "human_review.csv"
    if not path.exists():
        raise ReviewError("human_review.csv does not exist; run prepare first")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    by_id = {row.get("custom_id", ""): row for row in rows}
    errors: list[str] = []
    for record in plan:
        custom_id = record["custom_id"]
        row = by_id.get(custom_id)
        if row is None:
            errors.append(f"{custom_id}: missing row")
            continue
        for field in ["photorealism", "framing", "roll_no_back", "body_integrity"]:
            if row.get(field) not in VALID_BINARY:
                errors.append(f"{custom_id}: {field} must be pass/fail")
        if row.get("intent_match") not in VALID_INTENT:
            errors.append(f"{custom_id}: intent_match must be match/off-by-one-bin/wrong")
    if errors:
        preview = "\n".join(errors[:20])
        raise ReviewError(f"human review is incomplete or invalid:\n{preview}")
    reviewed = [by_id[record["custom_id"]] for record in plan]
    summary = {
        "stage": state["stage"],
        "total": len(reviewed),
        "photorealism_pass": sum(row["photorealism"] == "pass" for row in reviewed),
        "intent_match": sum(row["intent_match"] == "match" for row in reviewed),
        "intent_off_by_one": sum(row["intent_match"] == "off-by-one-bin" for row in reviewed),
        "intent_wrong": sum(row["intent_match"] == "wrong" for row in reviewed),
        "framing_pass": sum(row["framing"] == "pass" for row in reviewed),
        "roll_no_back_pass": sum(row["roll_no_back"] == "pass" for row in reviewed),
        "body_integrity_pass": sum(row["body_integrity"] == "pass" for row in reviewed),
    }
    lines = [
        f"# Human review summary: {state['local_batch_id']}", "",
        f"- Stage: {summary['stage']}",
        f"- Reviewed: {summary['total']} / {summary['total']}",
        f"- Photorealism pass: {summary['photorealism_pass']}",
        f"- Intent: match {summary['intent_match']}, off-by-one-bin {summary['intent_off_by_one']}, wrong {summary['intent_wrong']}",
        f"- Framing pass: {summary['framing_pass']}",
        f"- Roll/no-back pass: {summary['roll_no_back_pass']}", "",
        f"- Body integrity pass: {summary['body_integrity_pass']}", "",
    ]
    if state["stage"] == "validation" and summary["intent_match"] < 6:
        lines.append("Gate: FAILED. Revise failing-bin anchors and create a new 10-image Validation Batch. Pilot is prohibited.")
    else:
        lines.append("Gate: review complete. The next stage remains prohibited until approval.json is explicitly created.")
    (run_dir / "review_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def approve_review(run_dir: Path, reviewer: str) -> dict[str, Any]:
    if not reviewer.strip():
        raise ReviewError("reviewer must be non-empty")
    summary = summarize_review(run_dir)
    if summary["stage"] == "validation" and summary["intent_match"] < 6:
        raise ReviewError("validation intent_match is below 6; approval is forbidden")
    # Validation is a feedback gate: >=6 intent matches may approve the improved
    # Pilot prompt even when individual test images exposed correctable failures.
    # Pilot approval is stricter because it authorizes Production.
    if summary["stage"] != "validation":
        for field in ["photorealism_pass", "framing_pass", "roll_no_back_pass", "body_integrity_pass"]:
            if summary[field] != summary["total"]:
                raise ReviewError(
                    f"{field} is below {summary['total']}; repair failed images before approval"
                )
        qa_path = run_dir / "auto_qa.jsonl"
        if not qa_path.exists():
            raise ReviewError("auto_qa.jsonl is missing; approval is forbidden")
        with qa_path.open(encoding="utf-8") as fh:
            qa_rows = [json.loads(line) for line in fh]
        failed_ids = [
            row.get("custom_id", "unknown") for row in qa_rows
            if not row.get("quality_gate_complete") or not row.get("quality_gate_pass")
        ]
        if len(qa_rows) != summary["total"] or failed_ids:
            preview = ", ".join(failed_ids[:10])
            raise ReviewError(f"machine quality gate is incomplete or failed: {preview}")
    review_path = run_dir / "human_review.csv"
    approval = {
        "approved": True,
        "stage": summary["stage"],
        "local_batch_id": _read_state(run_dir)["local_batch_id"],
        "reviewer": reviewer.strip(),
        "approved_at": datetime.now(UTC).isoformat(),
        "review_sha256": _sha256(review_path),
        **summary,
    }
    path = run_dir / "approval.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(approval, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return approval


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QA artifacts and explicit human-review gates for GPT head generation runs.")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--batch-dir", type=Path, required=True)
    prepare.add_argument("--skip-detector", action="store_true")
    overview = sub.add_parser("overview")
    overview.add_argument("--batch-dir", type=Path, required=True)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--batch-dir", type=Path, required=True)
    approve = sub.add_parser("approve")
    approve.add_argument("--batch-dir", type=Path, required=True)
    approve.add_argument("--reviewer", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_review_artifacts(args.batch_dir, not args.skip_detector)
        elif args.command == "overview":
            result = regenerate_overview_contact_sheet(args.batch_dir)
        elif args.command == "summarize":
            result = summarize_review(args.batch_dir)
        else:
            result = approve_review(args.batch_dir, args.reviewer)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ReviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
