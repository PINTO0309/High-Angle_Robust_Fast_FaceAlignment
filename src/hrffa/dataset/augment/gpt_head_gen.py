"""OpenAI Batch API based synthetic head-image generation pipeline.

The public CLI intentionally fixes gpt-image-2/low/opaque/n=1.  The original
head-pose profile uses PNG API responses and local JPEG q92 storage; an explicit
JPEG q92 API profile is also supported for large self-contained training runs.  It plans
deterministic requests, submits JSONL files to the Batch API, persists every remote
identifier, collects results without depending on response order, and retries only
missing custom IDs.

Examples:
    python -m hrffa.dataset.augment.gpt_head_gen plan \
        --stage validation --batch-id validation-v001
    python -m hrffa.dataset.augment.gpt_head_gen submit \
        --batch-dir data/imagegen/hrffa-heads/validation-v001
    python -m hrffa.dataset.augment.gpt_head_gen status --batch-dir ...
    python -m hrffa.dataset.augment.gpt_head_gen collect --batch-dir ...
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import gzip
import hashlib
import io
import json
import os
import random
import shutil
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image

MODEL = "gpt-image-2"
ENDPOINT = "/v1/images/generations"
EDIT_ENDPOINT = "/v1/images/edits"
COMPLETION_WINDOW = "24h"
QUALITY = "low"
OUTPUT_FORMAT = "png"
BACKGROUND = "opaque"
N_IMAGES = 1
ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}
STORAGE_FORMAT = "JPEG"
STORAGE_EXTENSION = ".jpg"
JPEG_QUALITY = 92
SUPPORTED_API_OUTPUT_FORMATS = {"png", "jpeg"}
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
ACTIVE_STATUSES = {"validating", "in_progress", "finalizing", "cancelling"}
STATE_NAME = "batch_state.json"
PLAN_NAME = "generation_plan.jsonl"
SCENARIO_LOG_NAME = "scenario_log.jsonl"
OUTPUT_PRUNE_MANIFEST_NAME = "batch_output_prune_manifest.json"
INPUT_ARCHIVE_NAME = "batch_inputs.zip"
INPUT_ARCHIVE_MANIFEST_NAME = "_batch_inputs_manifest.json"

DEFAULT_AUTO_CORRECTION = {
    "enabled_stages": ["pilot", "production"],
    "max_quality_retries": 2,
}

GEOMETRY_EDIT_REASONS = {
    "head_too_small", "head_too_large", "insufficient_margin", "human_framing",
}


def _default_api_profile() -> dict[str, Any]:
    return {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "completion_window": COMPLETION_WINDOW,
        "quality": QUALITY,
        "output_format": OUTPUT_FORMAT,
        "background": BACKGROUND,
        "n": N_IMAGES,
        "allowed_sizes": sorted(ALLOWED_SIZES),
    }


def _normalized_api_profile(value: dict[str, Any] | None) -> dict[str, Any]:
    profile = dict(value or _default_api_profile())
    profile.setdefault("allowed_sizes", sorted(ALLOWED_SIZES))
    return profile

QUALITY_CORRECTIONS = {
    "duplicate_image": "Create a genuinely different photograph while preserving the requested pose and attributes.",
    "head_not_detected": "Make exactly one complete human head unambiguous and clearly detectable.",
    "head_count_not_one": "Show exactly one person and exactly one head; no reflections, screens, posters, or bystanders.",
    "head_too_small": "Move the camera closer: target 32% to 40% head height and never fall below 25% of image height.",
    "head_too_large": "Move the camera slightly farther away: target 30% to 42% head height and never exceed 50%.",
    "insufficient_margin": "Recompose without cropping and leave empty margin on every side of at least half the head size.",
    "back_of_head": "Rotate the pose enough to preserve visible facial evidence; never show a pure back-of-head view.",
    "body_not_detected": "Make the person's neck, shoulders, and body visibly present and physically continuous.",
    "human_photorealism": "Replace the failed synthesis with a natural, unedited-looking real photograph without visible generative artifacts.",
    "human_framing": "Include the complete head, neck, shoulders, required margins, and the 25% to 50% hard head-height range.",
    "human_roll_or_back": "Keep roll visually at zero and retain clear facial evidence rather than a back-of-head view.",
    "human_body_integrity": "Show a continuous, anatomically coherent body, or hide it only behind a clearly visible natural occluder. Never dissolve or merge the torso or limbs into the floor, background, or furniture.",
    "human_intent_off_by_one": "Strengthen the requested pose angle so it lands in the intended bin rather than an adjacent bin.",
    "human_intent_wrong": "Rebuild the pose around the exact requested pitch, yaw, and camera elevation; that angle must dominate the scene.",
    "auto_intent_off_by_one": "Strengthen the physical upward-pose cue so the actual head lands inside the requested pitch bin, not an adjacent bin.",
    "auto_intent_wrong": "The prior image collapsed toward a level or incompatible pose. Rebuild the exact upward pitch and yaw around the physical scenario; the underside of nose, jaw, and throat must prove the angle.",
    "auto_roll_or_back": "Keep the image-plane head roll near zero and preserve visible facial evidence; never rotate into a back-of-head view.",
}


class PipelineError(RuntimeError):
    """A user-actionable pipeline error."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    return value


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    api = config.get("api", {})
    expected = {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "completion_window": COMPLETION_WINDOW,
        "quality": QUALITY,
        "background": BACKGROUND,
        "n": N_IMAGES,
    }
    for key, value in expected.items():
        if api.get(key) != value:
            raise PipelineError(f"config api.{key} must be fixed to {value!r}")
    if set(api.get("allowed_sizes", [])) != ALLOWED_SIZES:
        raise PipelineError(f"config sizes must be exactly {sorted(ALLOWED_SIZES)}")
    output_format = api.get("output_format")
    if output_format not in SUPPORTED_API_OUTPUT_FORMATS:
        raise PipelineError(
            f"config api.output_format must be one of {sorted(SUPPORTED_API_OUTPUT_FORMATS)}"
        )
    if output_format == "png" and "output_compression" in api:
        raise PipelineError("config api.output_compression is forbidden for PNG output")
    if output_format == "jpeg" and api.get("output_compression") != JPEG_QUALITY:
        raise PipelineError("config JPEG API output must use output_compression 92")
    storage = config.get("storage", {})
    if storage.get("format") != "jpeg" or storage.get("quality") != JPEG_QUALITY:
        raise PipelineError("config storage must be fixed to jpeg quality 92")
    correction = config.get("auto_correction", {})
    if int(correction.get("max_quality_retries", 0)) < 0:
        raise PipelineError("auto_correction.max_quality_retries must be non-negative")
    known_stages = {"validation", *config.get("stages", {})}
    if not set(correction.get("enabled_stages", [])) <= known_stages:
        raise PipelineError("auto_correction.enabled_stages contains an unknown stage")
    return config


def _validate_edit_images(value: Any) -> None:
    if (not isinstance(value, list) or len(value) != 1
            or not isinstance(value[0], dict)
            or set(value[0]) != {"image_url"}):
        raise PipelineError(
            "edit request body.images must contain exactly one image_url reference"
        )
    image_url = value[0]["image_url"]
    if not isinstance(image_url, str) or not image_url.startswith(
        "data:image/jpeg;base64,"
    ):
        raise PipelineError("edit request image_url must be an inline JPEG data URI")
    encoded = image_url.partition(",")[2]
    if not encoded:
        raise PipelineError("edit request image_url is empty")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PipelineError("edit request body.image is not valid base64") from exc
    if len(payload) >= 50 * 1024 * 1024:
        raise PipelineError("edit source image must be smaller than 50 MB")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
            if image.format != "JPEG":
                raise PipelineError("edit source data URI must contain a JPEG")
    except (OSError, ValueError) as exc:
        raise PipelineError("edit request body.image is not a valid JPEG") from exc


def validate_batch_request(request: dict[str, Any],
                           api_profile: dict[str, Any] | None = None,
                           expected_endpoint: str = ENDPOINT) -> None:
    profile = _normalized_api_profile(api_profile)
    if expected_endpoint not in {ENDPOINT, EDIT_ENDPOINT}:
        raise PipelineError(f"unsupported Batch endpoint: {expected_endpoint!r}")
    if request.get("method") != "POST" or request.get("url") != expected_endpoint:
        raise PipelineError(f"every request must POST to {expected_endpoint}")
    custom_id = request.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        raise PipelineError("custom_id must be a non-empty string")
    body = request.get("body")
    if not isinstance(body, dict):
        raise PipelineError("request body must be an object")
    fixed = {
        "model": MODEL,
        "n": N_IMAGES,
        "quality": QUALITY,
        "background": BACKGROUND,
        "output_format": profile["output_format"],
    }
    for key, value in fixed.items():
        if body.get(key) != value:
            raise PipelineError(f"request body.{key} must be {value!r}")
    if body.get("size") not in ALLOWED_SIZES:
        raise PipelineError(f"unsupported size: {body.get('size')!r}")
    if not isinstance(body.get("prompt"), str) or not body["prompt"].strip():
        raise PipelineError("prompt must be non-empty")
    expected_options = {
        "model", "prompt", "n", "size", "quality", "background", "output_format"
    }
    if expected_endpoint == EDIT_ENDPOINT:
        _validate_edit_images(body.get("images"))
        expected_options.add("images")
    if profile["output_format"] == "jpeg":
        if body.get("output_compression") != JPEG_QUALITY:
            raise PipelineError("request body.output_compression must be 92 for JPEG")
        expected_options.add("output_compression")
    unexpected = set(body) - expected_options
    if unexpected:
        raise PipelineError(f"unexpected image API options: {sorted(unexpected)}")


def _validate_jsonl_lines(lines: Iterable[str], label: str,
                          api_profile: dict[str, Any] | None = None,
                          expected_endpoint: str = ENDPOINT) -> list[str]:
    custom_ids: list[str] = []
    for number, line in enumerate(lines, 1):
        try:
            request = json.loads(line)
            validate_batch_request(request, api_profile, expected_endpoint)
        except (json.JSONDecodeError, PipelineError) as exc:
            raise PipelineError(f"{label}:{number}: {exc}") from exc
        custom_ids.append(request["custom_id"])
    if len(custom_ids) != len(set(custom_ids)):
        raise PipelineError(f"duplicate custom_id in {label}")
    if not custom_ids:
        raise PipelineError(f"empty Batch input: {label}")
    return custom_ids


def validate_jsonl(path: Path, api_profile: dict[str, Any] | None = None,
                   expected_endpoint: str = ENDPOINT) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return _validate_jsonl_lines(fh, str(path), api_profile, expected_endpoint)


def _validate_jsonl_bytes(payload: bytes, label: str,
                          api_profile: dict[str, Any] | None = None,
                          expected_endpoint: str = ENDPOINT) -> list[str]:
    try:
        text_payload = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PipelineError(f"Batch input is not UTF-8: {label}") from exc
    return _validate_jsonl_lines(
        io.StringIO(text_payload), label, api_profile, expected_endpoint
    )


def _sample_angle(spec: dict[str, Any], index: int, rng: random.Random, salt: int) -> int:
    mode = spec["mode"]
    if mode == "fixed":
        return int(spec["value"])
    low, high = int(spec["min"]), int(spec["max"])
    # Avoid clustering exactly at bin boundaries when the range permits it.
    inner_low, inner_high = (low + 2, high - 2) if high - low >= 6 else (low, high)
    magnitude = rng.randint(inner_low, inner_high)
    if mode == "fixed_sign":
        return int(spec["sign"]) * magnitude
    if mode == "signed_abs":
        return magnitude if (index + salt) % 2 == 0 else -magnitude
    raise PipelineError(f"unknown angle mode: {mode}")


def _exact_schedule(values: list[tuple[str, int]], total: int, rng: random.Random) -> list[str]:
    result: list[str] = []
    allocated = 0
    for index, (value, share) in enumerate(values):
        count = total - allocated if index == len(values) - 1 else total * share // 100
        result.extend([value] * count)
        allocated += count
    rng.shuffle(result)
    return result


def _boolean_schedule(total: int, percent: int, rng: random.Random) -> list[bool]:
    values = [True] * (total * percent // 100) + [False] * (total - total * percent // 100)
    rng.shuffle(values)
    return values


def _format_custom_id(pitch: int, yaw: int, cam: int, serial: int) -> str:
    return f"pitch{pitch:+04d}_yaw{yaw:+04d}_cam{cam:+03d}_{serial:06d}"


def _make_prompt(config: dict[str, Any], record: dict[str, Any]) -> str:
    prompt = config["prompt"]
    details = (
        f"Subject: a fictional adult {record['gender']} in their {record['age']}, "
        f"skin tone {record['skin_tone']}, {record['hair']}, wearing {record['clothing']}. "
        f"Scene: {record['context']} in {record['background']}; {record['lighting']}. "
        f"Camera character: {record['lens_feel']}. Accessories: {record['accessories']}."
    )
    pose = prompt["pose"].format(
        pitch=record["pitch"], yaw=record["yaw"], cam=record["cam"]
    )
    return " ".join([
        prompt["preamble"],
        f"Pose anchor: {record['anchor']}",
        pose,
        details,
        prompt["framing"],
        prompt["body_integrity"],
        prompt["realism"],
    ])


def _batch_request(record: dict[str, Any],
                   api_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = _normalized_api_profile(api_profile)
    request = {
        "custom_id": record["custom_id"],
        "method": "POST",
        "url": ENDPOINT,
        "body": {
            "model": MODEL,
            "prompt": record["prompt"],
            "n": N_IMAGES,
            "size": record["size"],
            "quality": QUALITY,
            "background": BACKGROUND,
            "output_format": profile["output_format"],
        },
    }
    if profile["output_format"] == "jpeg":
        request["body"]["output_compression"] = JPEG_QUALITY
    validate_batch_request(request, profile)
    return request


def _common_diversity(config: dict[str, Any], index: int, total: int,
                      rng: random.Random, schedules: dict[str, list[Any]]) -> dict[str, Any]:
    prompt = config["prompt"]
    accessories: list[str] = []
    if schedules["mask"][index]:
        accessories.append("a correctly worn medical mask")
    if schedules["hat"][index]:
        accessories.append("a plausible hat, cap, or hood")
    if schedules["glasses"][index]:
        accessories.append("eyeglasses")
    return {
        "gender": prompt["gender_presentations"][index % len(prompt["gender_presentations"])],
        "age": prompt["age_bands"][index % len(prompt["age_bands"])],
        "skin_tone": prompt["skin_tones"][index % len(prompt["skin_tones"])],
        "hair": prompt["hair"][index % len(prompt["hair"])],
        "clothing": prompt["clothing"][index % len(prompt["clothing"])],
        "lens_feel": prompt["lens_feel"][index % len(prompt["lens_feel"])],
        "background": prompt["backgrounds"][index % len(prompt["backgrounds"])],
        "lighting": schedules["lighting"][index],
        "accessories": ", ".join(accessories) if accessories else "none",
    }


def build_plan(config: dict[str, Any], stage: str, seed: int, *,
               bin_counts: list[int] | None = None,
               serial_offset: int = 0) -> list[dict[str, Any]]:
    known_stages = {"validation", *config.get("stages", {})}
    if stage not in known_stages:
        raise PipelineError(f"unknown stage: {stage}")
    rng = random.Random(seed)
    bins = {item["id"]: item for item in config["bins"]}
    if stage == "validation":
        if bin_counts is not None or serial_offset:
            raise PipelineError("validation does not support partial-plan overrides")
        source = list(config["validation"])
        total = len(source)
        if total != 10:
            raise PipelineError("validation must contain exactly 10 records")
        assignments = [item["bin"] for item in source]
    else:
        stage_config = config["stages"][stage]
        counts = list(bin_counts) if bin_counts is not None else stage_config["bin_counts"]
        expected_count = sum(counts) if bin_counts is not None else stage_config["count"]
        if len(counts) != len(config["bins"]) or sum(counts) != expected_count:
            raise PipelineError(f"invalid {stage} bin counts")
        early = stage_config.get("early_drift_check") or {}
        early_counts = early.get("bin_counts")
        if early_counts is not None:
            early_counts = [int(value) for value in early_counts]
            if (len(early_counts) != len(counts)
                    or sum(early_counts) != int(early.get("count", 0))
                    or any(early_count < 0 or early_count > count
                           for early_count, count in zip(early_counts, counts))):
                raise PipelineError(f"invalid {stage} early-drift bin counts")
            assignments = [
                bin_cfg["id"]
                for bin_cfg, count in zip(config["bins"], early_counts)
                for _ in range(count)
            ]
            rng.shuffle(assignments)
            remaining = [
                bin_cfg["id"]
                for bin_cfg, count, early_count in zip(config["bins"], counts, early_counts)
                for _ in range(count - early_count)
            ]
            rng.shuffle(remaining)
            assignments.extend(remaining)
        else:
            assignments = [
                bin_cfg["id"]
                for bin_cfg, count in zip(config["bins"], counts)
                for _ in range(count)
            ]
            rng.shuffle(assignments)
        total = len(assignments)
        source = []

    size_values = [("1024x1536", 50), ("1024x1024", 30), ("1536x1024", 20)]
    sizes = _exact_schedule(size_values, total, rng)
    lighting_values = [(item["name"], int(item["share"])) for item in config["prompt"]["lighting"]]
    schedules: dict[str, list[Any]] = {
        "lighting": _exact_schedule(lighting_values, total, rng),
        "mask": _boolean_schedule(total, 15, rng),
        "hat": _boolean_schedule(total, 10, rng),
        "glasses": _boolean_schedule(total, 15, rng),
    }
    records: list[dict[str, Any]] = []
    local_indices = {bin_id: 0 for bin_id in bins}
    for index, bin_id in enumerate(assignments):
        bin_cfg = bins[bin_id]
        local_index = local_indices[bin_id]
        local_indices[bin_id] += 1
        diversity = _common_diversity(config, index, total, rng, schedules)
        if stage == "validation":
            item = source[index]
            pitch, yaw, cam = int(item["pitch"]), int(item["yaw"]), int(item["cam"])
            size = item["size"]
            context = item["context"]
            if item.get("hair"):
                diversity["hair"] = item["hair"]
            if item.get("accessory"):
                diversity["accessories"] = item["accessory"]
        else:
            pitch = _sample_angle(bin_cfg["pitch"], local_index, rng, 0)
            yaw = _sample_angle(bin_cfg["yaw"], local_index, rng, 1)
            cam = _sample_angle(bin_cfg["cam"], local_index, rng, 0)
            size = sizes[index]
            postures = config["prompt"]["postures"][bin_id]
            posture = postures[local_index % len(postures)]
            if isinstance(posture, dict):
                context = str(posture["scenario"])
                compatible_backgrounds = posture.get("backgrounds") or []
                if compatible_backgrounds:
                    diversity["background"] = compatible_backgrounds[
                        local_index % len(compatible_backgrounds)
                    ]
            else:
                context = str(posture)
        serial = serial_offset + index + 1
        record: dict[str, Any] = {
            "serial": serial,
            "stage": stage,
            "bin": bin_id,
            "pitch": pitch,
            "yaw": yaw,
            "cam": cam,
            "roll": 0,
            "size": size,
            "context": context,
            "scenario": context,
            "anchor": bin_cfg["anchor"],
            **diversity,
        }
        record["custom_id"] = _format_custom_id(pitch, yaw, cam, serial)
        record["filename"] = record["custom_id"] + STORAGE_EXTENSION
        record["prompt"] = _make_prompt(config, record)
        records.append(record)
    if len({record["custom_id"] for record in records}) != len(records):
        raise PipelineError("planner produced duplicate custom_ids")
    return records


def build_production_plan_with_pilot(config: dict[str, Any], seed: int,
                                     pilot_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build 5,000 final records with the approved 500-image Pilot as a subset."""
    pilot_cfg = config["stages"]["pilot"]
    production_cfg = config["stages"]["production"]
    if len(pilot_records) != int(pilot_cfg["count"]):
        raise PipelineError("approved Pilot plan must contain exactly 500 records")

    bin_ids = [item["id"] for item in config["bins"]]
    pilot_histogram = {bin_id: 0 for bin_id in bin_ids}
    for record in pilot_records:
        bin_id = record.get("bin")
        if bin_id not in pilot_histogram:
            raise PipelineError(f"approved Pilot contains an unknown bin: {bin_id!r}")
        pilot_histogram[bin_id] += 1
    expected_pilot = dict(zip(bin_ids, pilot_cfg["bin_counts"]))
    if pilot_histogram != expected_pilot:
        raise PipelineError("approved Pilot bin counts do not match the configured 500-image subset")

    size_histogram = {size: 0 for size in ALLOWED_SIZES}
    for record in pilot_records:
        size = record.get("size")
        if size not in size_histogram:
            raise PipelineError(f"approved Pilot contains an unsupported size: {size!r}")
        size_histogram[size] += 1
    if size_histogram != {"1024x1536": 250, "1024x1024": 150, "1536x1024": 100}:
        raise PipelineError("approved Pilot size counts do not match the 10% Production subset")

    remaining_counts = [
        int(production) - int(pilot)
        for production, pilot in zip(production_cfg["bin_counts"], pilot_cfg["bin_counts"])
    ]
    if any(count < 0 for count in remaining_counts):
        raise PipelineError("Pilot bin count exceeds the final Production quota")

    inherited: list[dict[str, Any]] = []
    for record in pilot_records:
        copied = dict(record)
        copied.update({
            "stage": "production",
            "source_stage": "pilot",
            "reused_from_pilot": True,
            "filename": copied["custom_id"] + STORAGE_EXTENSION,
        })
        inherited.append(copied)
    generated = build_plan(
        config,
        "production",
        seed,
        bin_counts=remaining_counts,
        serial_offset=len(inherited),
    )
    records = inherited + generated
    if len(records) != int(production_cfg["count"]):
        raise PipelineError("inclusive Production plan must contain exactly 5,000 records")
    if len({record["custom_id"] for record in records}) != len(records):
        raise PipelineError("inclusive Production plan produced duplicate custom_ids")
    final_histogram = {bin_id: 0 for bin_id in bin_ids}
    for record in records:
        final_histogram[record["bin"]] += 1
    if final_histogram != dict(zip(bin_ids, production_cfg["bin_counts"])):
        raise PipelineError("inclusive Production bin counts do not match the final quotas")
    final_sizes = {size: 0 for size in ALLOWED_SIZES}
    for record in records:
        final_sizes[record["size"]] += 1
    if final_sizes != {"1024x1536": 2500, "1024x1024": 1500, "1536x1024": 1000}:
        raise PipelineError("inclusive Production size counts do not match the final quotas")
    return records


def _approval_ok(parent_dir: Path, required_stage: str) -> dict[str, Any]:
    approval_path = parent_dir / "approval.json"
    review_path = parent_dir / "human_review.csv"
    if not approval_path.exists():
        raise PipelineError(f"{required_stage} approval missing: {approval_path}")
    with approval_path.open(encoding="utf-8") as fh:
        approval = json.load(fh)
    if approval.get("stage") != required_stage or approval.get("approved") is not True:
        raise PipelineError(f"approval is not for completed {required_stage} review")
    if not review_path.exists() or approval.get("review_sha256") != sha256_file(review_path):
        raise PipelineError("human_review.csv changed after approval")
    if required_stage == "validation" and int(approval.get("intent_match", 0)) < 6:
        raise PipelineError("validation has fewer than 6 intent matches")
    return approval


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def create_plan(config_path: Path, stage: str, batch_id: str, output_root: Path,
                seed: int, approved_batch_dir: Path | None = None) -> Path:
    if not batch_id or any(char in batch_id for char in "/\\"):
        raise PipelineError("batch-id must be a safe single path component")
    parent_approval: dict[str, Any] | None = None
    if stage == "pilot":
        if approved_batch_dir is None:
            raise PipelineError("pilot planning requires --approved-batch-dir")
        parent_approval = _approval_ok(approved_batch_dir, "validation")
    elif stage == "production":
        if approved_batch_dir is None:
            raise PipelineError("production planning requires --approved-batch-dir")
        parent_approval = _approval_ok(approved_batch_dir, "pilot")

    config = load_config(config_path)
    reused_sources: dict[str, tuple[Path, str]] = {}
    if stage == "production":
        assert approved_batch_dir is not None
        pilot_plan_path = approved_batch_dir / PLAN_NAME
        if not pilot_plan_path.exists():
            raise PipelineError(f"approved Pilot plan is missing: {pilot_plan_path}")
        with pilot_plan_path.open(encoding="utf-8") as fh:
            pilot_records = [json.loads(line) for line in fh]
        records = build_production_plan_with_pilot(config, seed, pilot_records)
        for record in records:
            if not record.get("reused_from_pilot"):
                continue
            source = approved_batch_dir / "images_jpeg_q92" / record["filename"]
            if not _valid_existing_image(source, record["size"]):
                raise PipelineError(f"approved Pilot image is missing or invalid: {source}")
            reused_sources[record["custom_id"]] = (source, sha256_file(source))
    else:
        records = build_plan(config, stage, seed)
    run_dir = output_root / batch_id
    state_path = run_dir / STATE_NAME
    if state_path.exists():
        raise PipelineError(f"run already exists; refusing to overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "images").mkdir()
    for custom_id, (source, _digest) in reused_sources.items():
        target = run_dir / "images" / (custom_id + STORAGE_EXTENSION)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    _write_jsonl(run_dir / PLAN_NAME, records)
    _write_jsonl(run_dir / SCENARIO_LOG_NAME, ({
        "custom_id": record["custom_id"],
        "filename": record["filename"],
        "bin": record["bin"],
        "pitch": record["pitch"],
        "yaw": record["yaw"],
        "cam": record["cam"],
        "scenario": record.get("scenario", record["context"]),
        "background": record["background"],
    } for record in records))

    stage_config = config["stages"][stage]
    request_records = [record for record in records if not record.get("reused_from_pilot")]
    shard_sizes = stage_config.get("shard_sizes")
    if shard_sizes is not None:
        shard_sizes = [int(value) for value in shard_sizes]
        if not shard_sizes or any(value <= 0 for value in shard_sizes):
            raise PipelineError(f"invalid {stage} shard_sizes")
        if sum(shard_sizes) != len(request_records):
            raise PipelineError(
                f"{stage} shard_sizes total does not match request count"
            )
    else:
        shard_size = int(stage_config["shard_size"])
        shard_sizes = [
            min(shard_size, len(request_records) - start)
            for start in range(0, len(request_records), shard_size)
        ]
    shards: list[dict[str, Any]] = []
    start = 0
    api_profile = _normalized_api_profile(config["api"])
    early_cfg = stage_config.get("early_drift_check") or {}
    for shard_index, current_size in enumerate(shard_sizes):
        shard_records = request_records[start:start + current_size]
        start += current_size
        input_name = f"batch_input_{shard_index:03d}_attempt_00.jsonl"
        input_path = run_dir / input_name
        _write_jsonl(
            input_path,
            (_batch_request(record, api_profile) for record in shard_records),
        )
        custom_ids = validate_jsonl(input_path, api_profile)
        shards.append({
            "index": shard_index,
            "submission_phase": "early" if early_cfg and shard_index == 0 else "main",
            "custom_ids": custom_ids,
            "attempts": [{
                "number": 0,
                "kind": "generation",
                "input_path": input_name,
                "input_sha256": sha256_file(input_path),
                "custom_ids": custom_ids,
                "input_file_id": None,
                "batch_id": None,
                "status": "planned",
                "output_file_id": None,
                "error_file_id": None,
                "request_counts": None,
                "history": [{"at": utc_now(), "status": "planned"}],
            }],
        })
    state = {
        "schema_version": 2,
        "local_batch_id": batch_id,
        "stage": stage,
        "status": "planned",
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "auto_correction": config.get("auto_correction", {}),
        "api_request": api_profile,
        "pseudo_label": config.get("pseudo_label", {}),
        "plan_path": PLAN_NAME,
        "plan_sha256": sha256_file(run_dir / PLAN_NAME),
        "scenario_log_path": SCENARIO_LOG_NAME,
        "scenario_log_sha256": sha256_file(run_dir / SCENARIO_LOG_NAME),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "parent_batch_dir": str(approved_batch_dir.resolve()) if approved_batch_dir else None,
        "parent_approval_sha256": (
            sha256_file(approved_batch_dir / "approval.json") if parent_approval else None
        ),
        "target_count": len(records),
        "request_count": len(request_records),
        "reused_count": len(reused_sources),
        "items": {
            record["custom_id"]: ({
                "status": "success",
                "sha256": reused_sources[record["custom_id"]][1],
                "reused_from_pilot": True,
                "source_path": str(reused_sources[record["custom_id"]][0].resolve()),
                "filename": record["filename"],
            } if record["custom_id"] in reused_sources else {
                "status": "planned", "filename": record["filename"]
            })
            for record in records
        },
        "shards": shards,
    }
    if early_cfg:
        early_count = int(early_cfg.get("count", 0))
        if not shards or len(shards[0]["custom_ids"]) != early_count:
            raise PipelineError("early-drift count must match the first shard")
        state["submission_gate"] = {
            "type": "early_drift",
            "count": early_count,
            "custom_ids": list(shards[0]["custom_ids"]),
            "minimum_match_rate": float(early_cfg.get("minimum_match_rate", 0.80)),
            "minimum_bin_match_rate": float(
                early_cfg.get("minimum_bin_match_rate", 0.60)
            ),
            "released": False,
            "report_path": "early_drift_report.json",
        }
    _atomic_json(state_path, state)
    return run_dir


def load_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / STATE_NAME
    if not path.exists():
        raise PipelineError(f"state not found: {path}")
    with path.open(encoding="utf-8") as fh:
        state = json.load(fh)
    plan_path = run_dir / state["plan_path"]
    if sha256_file(plan_path) != state["plan_sha256"]:
        raise PipelineError("generation plan changed after creation")
    scenario_path = state.get("scenario_log_path")
    if scenario_path:
        path = run_dir / scenario_path
        if not path.exists() or sha256_file(path) != state.get("scenario_log_sha256"):
            raise PipelineError("scenario log changed after creation")
    return state


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    attempts = [attempt for shard in state.get("shards", []) for attempt in shard.get("attempts", [])]
    attempt_statuses = [attempt.get("status", "unknown") for attempt in attempts]
    items = state.get("items", {})
    if items and all(item.get("status") == "success" for item in items.values()):
        state["status"] = "collected"
    elif any(status in ACTIVE_STATUSES for status in attempt_statuses):
        state["status"] = next(status for status in attempt_statuses if status in ACTIVE_STATUSES)
    elif any(status == "planned" for status in attempt_statuses):
        state["status"] = "planned"
    elif attempt_statuses and all(status == "completed" for status in attempt_statuses):
        state["status"] = "completed"
    elif attempt_statuses and all(status in TERMINAL_STATUSES for status in attempt_statuses):
        state["status"] = "terminal_with_failures"
    state["updated_at"] = utc_now()
    _atomic_json(run_dir / STATE_NAME, state)


def read_plan(run_dir: Path, state: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    state = state or load_state(run_dir)
    records: dict[str, dict[str, Any]] = {}
    with (run_dir / state["plan_path"]).open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            records[record["custom_id"]] = record
    return records


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Replace a JSONL file only after its complete new contents are durable."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def revise_held_prompts(run_dir: Path, config_path: Path) -> dict[str, Any]:
    """Apply a reviewed prompt revision only to requests held behind an early gate.

    The already-submitted early sample is immutable.  A revision is allowed only
    while every held request has neither a remote file ID nor a Batch ID.  This
    makes the operation useful after an early drift review without weakening the
    duplicate-submission guarantees.
    """
    state = load_state(run_dir)
    if (run_dir / "approval.json").exists():
        raise PipelineError("run is already approved")
    gate = state.get("submission_gate") or {}
    if gate.get("type") != "early_drift" or gate.get("released") is True:
        raise PipelineError("held prompts can be revised only while the early-drift gate is closed")

    config = load_config(config_path)
    stage = str(state.get("stage"))
    if stage not in config.get("stages", {}):
        raise PipelineError(f"revision config does not define stage {stage!r}")
    api_profile = _normalized_api_profile(config.get("api"))
    if api_profile != _normalized_api_profile(state.get("api_request")):
        raise PipelineError("revision config changes the immutable API request profile")

    held_shards = [
        shard for shard in state.get("shards", [])
        if not _submission_allowed(state, shard)
    ]
    if not held_shards:
        raise PipelineError("no requests are held behind the early-drift gate")

    generation_attempts: dict[int, dict[str, Any]] = {}
    held_ids: list[str] = []
    for shard in held_shards:
        shard_ids = list(shard.get("custom_ids") or [])
        if not shard_ids:
            raise PipelineError(f"held shard {shard.get('index')} is empty")
        active_attempts = [
            attempt for attempt in shard.get("attempts", [])
            if not attempt.get("superseded")
        ]
        if any(attempt.get("batch_id") or attempt.get("input_file_id")
               for attempt in active_attempts):
            raise PipelineError(
                f"held shard {shard.get('index')} already has remote OpenAI state"
            )
        originals = [
            attempt for attempt in active_attempts
            if attempt.get("kind") == "generation" and int(attempt.get("number", -1)) == 0
        ]
        if len(active_attempts) != 1 or len(originals) != 1:
            raise PipelineError(
                f"held shard {shard.get('index')} is not a pristine unsent generation shard"
            )
        attempt = originals[0]
        if list(attempt.get("custom_ids") or []) != shard_ids:
            raise PipelineError(f"held shard {shard.get('index')} custom_ids are inconsistent")
        generation_attempts[int(shard["index"])] = attempt
        held_ids.extend(shard_ids)

    if len(held_ids) != len(set(held_ids)):
        raise PipelineError("held requests contain duplicate custom_ids")
    held_set = set(held_ids)
    early_set = set(gate.get("custom_ids") or [])
    if held_set & early_set:
        raise PipelineError("held requests overlap the immutable early sample")
    for custom_id in held_ids:
        item = state.get("items", {}).get(custom_id)
        if item is None or item.get("status") != "planned":
            raise PipelineError(f"held request is not pristine planned state: {custom_id}")
        filename = str(item.get("filename") or custom_id + STORAGE_EXTENSION)
        if (run_dir / "images" / filename).exists():
            raise PipelineError(f"held request already has a local image: {custom_id}")

    plan_path = run_dir / state["plan_path"]
    with plan_path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    if set(record["custom_id"] for record in records) != set(state.get("items", {})):
        raise PipelineError("generation plan and state items are inconsistent")

    bins = {str(item["id"]): item for item in config["bins"]}
    occurrence: dict[str, int] = {bin_id: 0 for bin_id in bins}
    revised: list[dict[str, Any]] = []
    changed = 0
    for original in records:
        record = dict(original)
        bin_id = str(record.get("bin"))
        if bin_id not in bins:
            raise PipelineError(f"revision config does not define plan bin {bin_id!r}")
        local_index = occurrence[bin_id]
        occurrence[bin_id] += 1
        if record["custom_id"] in held_set:
            bin_cfg = bins[bin_id]
            postures = config["prompt"]["postures"][bin_id]
            posture = postures[local_index % len(postures)]
            if isinstance(posture, dict):
                context = str(posture["scenario"])
                compatible = posture.get("backgrounds") or []
                if compatible:
                    record["background"] = compatible[local_index % len(compatible)]
            else:
                context = str(posture)
            record["context"] = context
            record["scenario"] = context
            record["anchor"] = str(bin_cfg["anchor"])
            record["prompt"] = _make_prompt(config, record)
            if record != original:
                changed += 1
        revised.append(record)

    if changed == 0:
        current_config_sha = sha256_file(config_path)
        if (state.get("config_sha256") == current_config_sha
                and state.get("auto_correction") == config.get("auto_correction", {})):
            return {
                "revised_requests": 0,
                "held_requests": len(held_ids),
                "config_sha256": current_config_sha,
                "idempotent": True,
            }

    by_id = {record["custom_id"]: record for record in revised}
    scenario_rows = [{
        "custom_id": record["custom_id"],
        "filename": record["filename"],
        "bin": record["bin"],
        "pitch": record["pitch"],
        "yaw": record["yaw"],
        "cam": record["cam"],
        "scenario": record.get("scenario", record["context"]),
        "background": record["background"],
    } for record in revised]

    old_plan_sha = state["plan_sha256"]
    old_scenario_sha = state.get("scenario_log_sha256")
    old_config_sha = state.get("config_sha256")
    input_changes: list[dict[str, Any]] = []
    _atomic_write_jsonl(plan_path, revised)
    scenario_path = run_dir / state.get("scenario_log_path", SCENARIO_LOG_NAME)
    _atomic_write_jsonl(scenario_path, scenario_rows)
    for shard in held_shards:
        attempt = generation_attempts[int(shard["index"])]
        input_path = _safe_run_file(run_dir, str(attempt["input_path"]))
        old_input_sha = attempt["input_sha256"]
        _atomic_write_jsonl(
            input_path,
            (_batch_request(by_id[custom_id], api_profile)
             for custom_id in shard["custom_ids"]),
        )
        custom_ids = validate_jsonl(input_path, api_profile)
        if custom_ids != list(shard["custom_ids"]):
            raise PipelineError(f"revised held input order changed for shard {shard['index']}")
        new_input_sha = sha256_file(input_path)
        attempt["input_sha256"] = new_input_sha
        attempt.setdefault("history", []).append({
            "at": utc_now(),
            "status": "planned",
            "event": "held_prompt_revision",
            "old_input_sha256": old_input_sha,
            "new_input_sha256": new_input_sha,
        })
        input_changes.append({
            "shard": int(shard["index"]),
            "requests": len(custom_ids),
            "old_input_sha256": old_input_sha,
            "new_input_sha256": new_input_sha,
        })

    state["plan_sha256"] = sha256_file(plan_path)
    state["scenario_log_sha256"] = sha256_file(scenario_path)
    state["config_path"] = str(config_path)
    state["config_sha256"] = sha256_file(config_path)
    state["auto_correction"] = config.get("auto_correction", {})
    state["pseudo_label"] = config.get("pseudo_label", {})
    held_id_sha = hashlib.sha256("\n".join(held_ids).encode("utf-8")).hexdigest()
    revision = {
        "at": utc_now(),
        "scope": "unsent_requests_held_behind_early_drift_gate",
        "revised_requests": changed,
        "held_requests": len(held_ids),
        "held_custom_ids_sha256": held_id_sha,
        "immutable_early_requests": len(early_set),
        "old_config_sha256": old_config_sha,
        "new_config_sha256": state["config_sha256"],
        "old_plan_sha256": old_plan_sha,
        "new_plan_sha256": state["plan_sha256"],
        "old_scenario_log_sha256": old_scenario_sha,
        "new_scenario_log_sha256": state["scenario_log_sha256"],
        "input_changes": input_changes,
    }
    state.setdefault("prompt_revision_history", []).append(revision)
    save_state(run_dir, state)
    return {**revision, "idempotent": False}


def revise_unsent_prompts(run_dir: Path, instruction: str) -> dict[str, Any]:
    """Append an audited rolling-QA correction to pristine unsent generations."""
    instruction = instruction.strip()
    if not instruction:
        raise PipelineError("unsent prompt correction must not be empty")
    state = load_state(run_dir)
    if (run_dir / "approval.json").exists():
        raise PipelineError("run is already approved")
    for shard in state.get("shards", []):
        for attempt in shard.get("attempts", []):
            if not attempt.get("superseded") and attempt.get("status") in ACTIVE_STATUSES:
                raise PipelineError(f"Batch {attempt.get('batch_id')} is still {attempt['status']}")

    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for shard in state.get("shards", []):
        if shard.get("retired"):
            continue
        for attempt in shard.get("attempts", []):
            if attempt.get("superseded"):
                continue
            if (attempt.get("status") == "planned"
                    and not attempt.get("batch_id")
                    and not attempt.get("input_file_id")
                    and attempt.get("kind") in {"generation", "token_limit_reshard"}):
                pending.append((shard, attempt))
    if not pending:
        raise PipelineError("no pristine unsent generation input is available to revise")

    prefix = " ROLLING QA COMPOSITION CORRECTION (highest priority): "
    suffix = prefix + instruction
    pending_ids = {
        custom_id
        for _shard, attempt in pending
        for custom_id in attempt.get("custom_ids", [])
    }
    plan_path = run_dir / state["plan_path"]
    plan_rows = [
        json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row["custom_id"]: row for row in plan_rows}
    if pending_ids - set(by_id):
        raise PipelineError("unsent prompt correction references unknown plan IDs")
    already = {
        custom_id for custom_id in pending_ids
        if suffix in str(by_id[custom_id].get("prompt", ""))
    }
    if already and already != pending_ids:
        raise PipelineError("unsent prompt correction is only partially applied")
    if already == pending_ids:
        return {
            "revised_requests": len(pending_ids),
            "inputs": len(pending),
            "instruction": instruction,
            "idempotent": True,
        }

    old_plan_sha = sha256_file(plan_path)
    for custom_id in pending_ids:
        by_id[custom_id]["prompt"] = str(by_id[custom_id]["prompt"]) + suffix
    _atomic_write_jsonl(plan_path, plan_rows)
    input_changes: list[dict[str, Any]] = []
    for shard, attempt in pending:
        input_path = _safe_run_file(run_dir, str(attempt["input_path"]))
        payload = _input_payload(run_dir, state, attempt)
        rows = [
            json.loads(line) for line in payload.decode("utf-8").splitlines()
            if line.strip()
        ]
        if [row.get("custom_id") for row in rows] != attempt.get("custom_ids"):
            raise PipelineError(f"unsent Batch input IDs changed: {attempt['input_path']}")
        old_sha = str(attempt["input_sha256"])
        for row in rows:
            row["body"]["prompt"] = by_id[row["custom_id"]]["prompt"]
        _atomic_write_jsonl(input_path, rows)
        validate_jsonl(input_path, state.get("api_request"))
        new_sha = sha256_file(input_path)
        attempt["input_sha256"] = new_sha
        attempt.setdefault("history", []).append({
            "at": utc_now(),
            "status": "planned",
            "event": "rolling_qa_prompt_revision",
            "old_input_sha256": old_sha,
            "new_input_sha256": new_sha,
        })
        input_changes.append({
            "shard": int(shard["index"]),
            "attempt": int(attempt["number"]),
            "requests": len(rows),
            "old_input_sha256": old_sha,
            "new_input_sha256": new_sha,
        })
    state["plan_sha256"] = sha256_file(plan_path)
    event = {
        "at": utc_now(),
        "scope": "pristine_unsent_generation_requests",
        "revised_requests": len(pending_ids),
        "inputs": len(pending),
        "instruction": instruction,
        "old_plan_sha256": old_plan_sha,
        "new_plan_sha256": state["plan_sha256"],
        "input_changes": input_changes,
    }
    state.setdefault("rolling_qa_prompt_revision_history", []).append(event)
    save_state(run_dir, state)
    return {**event, "idempotent": False}


def rollback_latest_unsent_prompt_revision(run_dir: Path) -> dict[str, Any]:
    """Remove the latest rolling-QA suffix from pristine unsent requests only.

    Prompts already uploaded to OpenAI remain untouched in the plan so their
    provenance stays truthful.  This is intended for rolling QA that proves a
    newly appended instruction harmful before later sequential shards submit.
    """
    state = load_state(run_dir)
    if (run_dir / "approval.json").exists():
        raise PipelineError("run is already approved")
    for shard in state.get("shards", []):
        for attempt in shard.get("attempts", []):
            if not attempt.get("superseded") and attempt.get("status") in ACTIVE_STATUSES:
                raise PipelineError(f"Batch {attempt.get('batch_id')} is still {attempt['status']}")
    history = state.get("rolling_qa_prompt_revision_history", [])
    if not history:
        raise PipelineError("no rolling-QA prompt revision exists to roll back")
    instruction = str(history[-1].get("instruction", "")).strip()
    if not instruction:
        raise PipelineError("latest rolling-QA prompt revision has no instruction")

    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for shard in state.get("shards", []):
        if shard.get("retired"):
            continue
        for attempt in shard.get("attempts", []):
            if (not attempt.get("superseded")
                    and attempt.get("status") == "planned"
                    and not attempt.get("batch_id")
                    and not attempt.get("input_file_id")
                    and attempt.get("kind") in {"generation", "token_limit_reshard"}):
                pending.append((shard, attempt))
    if not pending:
        raise PipelineError("no pristine unsent generation input is available to roll back")
    pending_ids = {
        custom_id for _shard, attempt in pending
        for custom_id in attempt.get("custom_ids", [])
    }
    suffix = " ROLLING QA COMPOSITION CORRECTION (highest priority): " + instruction
    plan_path = run_dir / state["plan_path"]
    plan_rows = [
        json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row["custom_id"]: row for row in plan_rows}
    missing = [
        custom_id for custom_id in pending_ids
        if not str(by_id.get(custom_id, {}).get("prompt", "")).endswith(suffix)
    ]
    if missing:
        raise PipelineError(
            f"latest rolling-QA suffix is missing from {len(missing)} unsent prompt(s)"
        )
    old_plan_sha = sha256_file(plan_path)
    for custom_id in pending_ids:
        by_id[custom_id]["prompt"] = str(by_id[custom_id]["prompt"])[:-len(suffix)]
    _atomic_write_jsonl(plan_path, plan_rows)

    input_changes: list[dict[str, Any]] = []
    for shard, attempt in pending:
        input_path = _safe_run_file(run_dir, str(attempt["input_path"]))
        rows = [
            json.loads(line) for line in _input_payload(run_dir, state, attempt).decode("utf-8").splitlines()
            if line.strip()
        ]
        if [row.get("custom_id") for row in rows] != attempt.get("custom_ids"):
            raise PipelineError(f"unsent Batch input IDs changed: {attempt['input_path']}")
        old_sha = str(attempt["input_sha256"])
        for row in rows:
            row["body"]["prompt"] = by_id[row["custom_id"]]["prompt"]
        _atomic_write_jsonl(input_path, rows)
        validate_jsonl(input_path, state.get("api_request"))
        new_sha = sha256_file(input_path)
        attempt["input_sha256"] = new_sha
        attempt.setdefault("history", []).append({
            "at": utc_now(),
            "status": "planned",
            "event": "rolling_qa_prompt_revision_rollback",
            "old_input_sha256": old_sha,
            "new_input_sha256": new_sha,
        })
        input_changes.append({
            "shard": int(shard["index"]),
            "attempt": int(attempt["number"]),
            "requests": len(rows),
            "old_input_sha256": old_sha,
            "new_input_sha256": new_sha,
        })
    state["plan_sha256"] = sha256_file(plan_path)
    event = {
        "at": utc_now(),
        "scope": "pristine_unsent_generation_requests",
        "rolled_back_requests": len(pending_ids),
        "inputs": len(pending),
        "instruction": instruction,
        "old_plan_sha256": old_plan_sha,
        "new_plan_sha256": state["plan_sha256"],
        "input_changes": input_changes,
    }
    state.setdefault("rolling_qa_prompt_rollback_history", []).append(event)
    save_state(run_dir, state)
    return event


def _input_payload(run_dir: Path, state: dict[str, Any],
                   attempt: dict[str, Any]) -> bytes:
    input_path = _safe_run_file(run_dir, str(attempt["input_path"]))
    if input_path.exists():
        return input_path.read_bytes()
    archive = state.get("input_archiving") or {}
    archive_value = archive.get("archive")
    member = attempt.get("input_archive_member")
    if not isinstance(archive_value, str) or not isinstance(member, str):
        raise PipelineError(f"Batch input is missing: {input_path}")
    archive_path = _safe_run_file(run_dir, archive_value)
    if not archive_path.exists():
        raise PipelineError(f"Batch input archive is missing: {archive_path}")
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            payload = bundle.read(member)
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise PipelineError(f"cannot read archived Batch input {member}: {exc}") from exc
    if hashlib.sha256(payload).hexdigest() != attempt.get("input_sha256"):
        raise PipelineError(f"archived Batch input checksum mismatch: {member}")
    return payload


def _client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise PipelineError("OPENAI_API_KEY is not set")
    from openai import OpenAI
    return OpenAI()


def _sync_attempt_from_batch(attempt: dict[str, Any], batch: Any) -> None:
    status = str(_get(batch, "status", "unknown"))
    attempt.update({
        "batch_id": _get(batch, "id", attempt.get("batch_id")),
        "status": status,
        "output_file_id": _get(batch, "output_file_id"),
        "error_file_id": _get(batch, "error_file_id"),
        "request_counts": _jsonable(_get(batch, "request_counts")),
        "batch_errors": _jsonable(_get(batch, "errors")),
    })
    history = attempt.setdefault("history", [])
    if not history or history[-1].get("status") != status:
        history.append({"at": utc_now(), "status": status})


def _attempt_has_batch_error(attempt: dict[str, Any], code: str) -> bool:
    errors = attempt.get("batch_errors") or {}
    rows = errors.get("data", []) if isinstance(errors, dict) else []
    return any(
        isinstance(row, dict) and row.get("code") == code
        for row in rows
    )


def _find_remote_duplicate(client: Any, expected_metadata: dict[str, str]) -> Any | None:
    try:
        page = client.batches.list(limit=100)
    except Exception:  # Reconciliation is best effort; normal idempotency is local state.
        return None
    for batch in _get(page, "data", []) or []:
        metadata = _get(batch, "metadata", {}) or {}
        if all(metadata.get(key) == value for key, value in expected_metadata.items()):
            return batch
    return None


def _validate_current_stage_gate(state: dict[str, Any]) -> None:
    required = {"pilot": "validation", "production": "pilot"}.get(state["stage"])
    if required is None:
        return
    parent_value = state.get("parent_batch_dir")
    if not parent_value:
        raise PipelineError(f"{state['stage']} state has no parent approval")
    parent_dir = Path(parent_value)
    _approval_ok(parent_dir, required)
    approval_path = parent_dir / "approval.json"
    if sha256_file(approval_path) != state.get("parent_approval_sha256"):
        raise PipelineError("parent approval.json changed after this plan was created")


def _submission_allowed(state: dict[str, Any], shard: dict[str, Any]) -> bool:
    if shard.get("retired"):
        return False
    sequential_group = shard.get("sequential_group")
    if sequential_group:
        position = int(shard.get("sequence_position", 0))
        for prior in state.get("shards", []):
            if (prior.get("sequential_group") != sequential_group
                    or int(prior.get("sequence_position", 0)) >= position):
                continue
            attempts = [
                attempt for attempt in prior.get("attempts", [])
                if not attempt.get("superseded")
            ]
            if not attempts or attempts[-1].get("status") != "completed":
                return False
            if any(
                state.get("items", {}).get(custom_id, {}).get("status") != "success"
                for custom_id in prior.get("custom_ids", [])
            ):
                return False
    gate = state.get("submission_gate") or {}
    if gate.get("type") != "early_drift" or gate.get("released") is True:
        return True
    early_ids = set(gate.get("custom_ids") or [])
    shard_ids = set(shard.get("custom_ids") or [])
    return bool(shard_ids) and shard_ids <= early_ids


def _supersede_closed_gate_resume_artifacts(state: dict[str, Any]) -> int:
    """Neutralize unsent retries accidentally planned for intentionally held items."""
    gate = state.get("submission_gate") or {}
    if gate.get("type") != "early_drift" or gate.get("released") is True:
        return 0
    changed = 0
    for shard in state.get("shards", []):
        if _submission_allowed(state, shard):
            continue
        for attempt in shard.get("attempts", []):
            if (attempt.get("kind") == "api_retry"
                    and not attempt.get("batch_id")
                    and not attempt.get("input_file_id")
                    and not attempt.get("superseded")):
                attempt["superseded"] = True
                attempt["status"] = "superseded"
                attempt.setdefault("history", []).append({
                    "at": utc_now(),
                    "status": "superseded",
                    "reason": "request was intentionally held behind the early-drift gate",
                })
                changed += 1
    return changed


def _attempt_endpoint(attempt: dict[str, Any]) -> str:
    endpoint = str(attempt.get("endpoint", ENDPOINT))
    if endpoint not in {ENDPOINT, EDIT_ENDPOINT}:
        raise PipelineError(f"unsupported attempt endpoint: {endpoint!r}")
    return endpoint


def submit_pending(run_dir: Path, client: Any | None = None) -> list[str]:
    client = client or _client()
    state = load_state(run_dir)
    _validate_current_stage_gate(state)
    if _supersede_closed_gate_resume_artifacts(state):
        save_state(run_dir, state)
    remote_ids: list[str] = []
    for shard in state["shards"]:
        if not _submission_allowed(state, shard):
            continue
        for attempt in shard["attempts"]:
            if attempt.get("superseded"):
                continue
            endpoint = _attempt_endpoint(attempt)
            input_payload = _input_payload(run_dir, state, attempt)
            custom_ids = _validate_jsonl_bytes(
                input_payload, str(attempt["input_path"]), state.get("api_request"),
                endpoint,
            )
            if (custom_ids != attempt["custom_ids"]
                    or hashlib.sha256(input_payload).hexdigest() != attempt["input_sha256"]):
                raise PipelineError(f"Batch input changed: {attempt['input_path']}")
            if attempt.get("batch_id"):
                batch = client.batches.retrieve(attempt["batch_id"])
                _sync_attempt_from_batch(attempt, batch)
                save_state(run_dir, state)
                remote_ids.append(attempt["batch_id"])
                continue
            if (attempt.get("kind") in {"quality_retry", "fresh_replacement", "image_edit"}
                    and not attempt.get("archive_complete")):
                raise PipelineError(
                    f"quality retry archive is incomplete for shard {shard['index']} "
                    f"attempt {attempt['number']}"
                )

            metadata = {
                "local_batch_id": state["local_batch_id"][:64],
                "stage": state["stage"],
                "shard": str(shard["index"]),
                "attempt": str(attempt["number"]),
                "input_sha256": attempt["input_sha256"],
            }
            duplicate = _find_remote_duplicate(client, metadata)
            if duplicate is not None:
                _sync_attempt_from_batch(attempt, duplicate)
                attempt["history"].append({"at": utc_now(), "status": "reconciled_existing_batch"})
                save_state(run_dir, state)
                remote_ids.append(attempt["batch_id"])
                continue

            if not attempt.get("input_file_id"):
                upload = io.BytesIO(input_payload)
                upload.name = str(attempt["input_path"])
                uploaded = client.files.create(file=upload, purpose="batch")
                attempt["input_file_id"] = _get(uploaded, "id")
                attempt["history"].append({"at": utc_now(), "status": "input_uploaded"})
                save_state(run_dir, state)
            batch = client.batches.create(
                input_file_id=attempt["input_file_id"],
                endpoint=endpoint,
                completion_window=COMPLETION_WINDOW,
                metadata=metadata,
            )
            _sync_attempt_from_batch(attempt, batch)
            save_state(run_dir, state)
            remote_ids.append(attempt["batch_id"])
    return remote_ids


def refresh_status(run_dir: Path, client: Any | None = None) -> dict[str, Any]:
    client = client or _client()
    state = load_state(run_dir)
    summary: dict[str, Any] = {"stage": state["stage"], "batches": []}
    for shard in state["shards"]:
        for attempt in shard["attempts"]:
            if not attempt.get("batch_id"):
                continue
            batch = client.batches.retrieve(attempt["batch_id"])
            _sync_attempt_from_batch(attempt, batch)
            save_state(run_dir, state)
            if attempt.get("superseded"):
                # Preserve its remote terminal state for audit, but an
                # explicitly superseded race retry must not hold the current
                # generation cycle open or participate in repair decisions.
                continue
            summary["batches"].append({
                "shard": shard["index"],
                "attempt": attempt["number"],
                "batch_id": attempt["batch_id"],
                "status": attempt["status"],
                "request_counts": attempt.get("request_counts"),
            })
    summary["local_success"] = sum(item.get("status") == "success" for item in state["items"].values())
    summary["total"] = len(state["items"])
    return summary


def watch_until_terminal(run_dir: Path, interval_seconds: int = 60,
                         collect_on_terminal: bool = True,
                         detector: bool = True,
                         auto_repair: bool = False,
                         auto_resume: bool = False,
                         client: Any | None = None) -> dict[str, Any]:
    if interval_seconds < 5:
        raise PipelineError("watch interval must be at least 5 seconds")
    client = client or _client()
    while True:
        summary = refresh_status(run_dir, client)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        statuses = [batch["status"] for batch in summary["batches"]]
        if not statuses:
            raise PipelineError("no submitted Batch exists in this run")
        active_batches = [status for status in statuses if status in ACTIVE_STATUSES]
        current_state = load_state(run_dir)
        token_limit_exceeded = any(
            attempt.get("status") == "failed"
            and not attempt.get("superseded")
            and _attempt_has_batch_error(attempt, "token_limit_exceeded")
            for shard in current_state.get("shards", [])
            for attempt in shard.get("attempts", [])
        )
        resumable_terminal = any(
            status in {"failed", "expired", "cancelled"} for status in statuses
        )
        if (auto_resume and not token_limit_exceeded
                and len(active_batches) <= 1 and resumable_terminal):
            partial = collect_results(
                run_dir, client, prepare_review=False, detector=False
            )
            retry_requests = prepare_resume(run_dir)
            if retry_requests:
                summary["api_resume"] = {
                    "retry_requests": retry_requests,
                    "batch_ids": submit_pending(run_dir, client),
                    "local_success": partial["success"],
                }
                print(json.dumps(summary, ensure_ascii=False), flush=True)
                continue
        if all(status in TERMINAL_STATUSES for status in statuses):
            if token_limit_exceeded:
                summary["token_limit_exceeded"] = True
                summary["next_action"] = "split-token-limit"
                return summary
            current_gate = current_state.get("submission_gate") or {}
            if (current_gate.get("type") == "early_drift"
                    and current_gate.get("released") is not True):
                if collect_on_terminal:
                    summary["collection"] = collect_results(
                        run_dir, client, detector=detector
                    )
                summary["submission_gate"] = {
                    "type": "early_drift",
                    "released": False,
                    "held_requests": len(current_state["items"]) - int(
                        current_gate.get("count", 0)
                    ),
                }
                return summary
            if auto_resume:
                partial = collect_results(
                    run_dir, client, prepare_review=False, detector=False
                )
                if partial["missing"]:
                    retry_requests = prepare_resume(run_dir)
                    if not retry_requests:
                        raise PipelineError(
                            f"{partial['missing']} images are missing but no API resume was planned"
                        )
                    summary["api_resume"] = {
                        "retry_requests": retry_requests,
                        "batch_ids": submit_pending(run_dir, client),
                    }
                    print(json.dumps(summary, ensure_ascii=False), flush=True)
                    continue
            if collect_on_terminal:
                summary["collection"] = collect_results(run_dir, client, detector=detector)
            if auto_repair:
                repair = prepare_quality_retry(run_dir)
                summary["quality_repair"] = repair
                if repair["retry_requests"]:
                    summary["quality_repair"]["batch_ids"] = submit_pending(run_dir, client)
                    continue
                # Once cumulative quality-correction prompts are exhausted,
                # automatically switch to a fresh image sampled from the
                # immutable base prompt plus the current QA reasons.  The
                # configured round limit remains a hard cost/safety bound.
                repair_state = load_state(run_dir)
                fresh_rounds = int(repair_state.get("fresh_replacement_rounds", 0))
                fresh_limit = int(repair_state.get("fresh_replacement_max_rounds", 3))
                if repair["exhausted"] and fresh_rounds < fresh_limit:
                    replacement = prepare_fresh_replacement(run_dir)
                    summary["fresh_replacement"] = replacement
                    if replacement["retry_requests"]:
                        summary["fresh_replacement"]["batch_ids"] = submit_pending(
                            run_dir, client
                        )
                        continue
            return summary
        time.sleep(interval_seconds)


def _download_file(client: Any, file_id: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with client.files.with_streaming_response.content(file_id) as response:
        response.stream_to_file(tmp)
    tmp.replace(path)


def _gzip_jsonl(path: Path) -> Path:
    compressed = path.with_suffix(path.suffix + ".gz")
    tmp = compressed.with_suffix(compressed.suffix + ".tmp")
    with path.open("rb") as source, gzip.open(tmp, "wb", compresslevel=6) as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    tmp.replace(compressed)
    path.unlink()
    return compressed


def _valid_existing_image(path: Path, expected_size: str) -> bool:
    if not path.exists():
        return False
    expected = tuple(int(value) for value in expected_size.split("x"))
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            suffixes = {suffix.lower() for suffix in path.suffixes}
            expected_format = "JPEG" if suffixes & {".jpg", ".jpeg"} else "PNG"
            return image.format == expected_format and image.size == expected
    except Exception:  # noqa: BLE001 - corrupt images may raise several Pillow errors
        return False


def _extract_success(row: dict[str, Any]) -> tuple[str, str]:
    custom_id = row.get("custom_id")
    response = row.get("response") or {}
    if response.get("status_code") != 200:
        raise PipelineError(f"response status {response.get('status_code')}")
    data = (response.get("body") or {}).get("data") or []
    if len(data) != 1 or not isinstance(data[0].get("b64_json"), str):
        raise PipelineError("response body does not contain exactly one b64_json image")
    return custom_id, data[0]["b64_json"]


def process_output_jsonl(path: Path, run_dir: Path, state: dict[str, Any],
                         plan: dict[str, dict[str, Any]],
                         replace_existing: bool = False,
                         eligible_ids: set[str] | None = None) -> set[str]:
    images_dir = run_dir / "images"
    changed_ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            custom_id: str | None = None
            try:
                row = json.loads(line)
                custom_id, encoded = _extract_success(row)
                if custom_id not in plan:
                    raise PipelineError(f"unknown custom_id {custom_id!r}")
                if eligible_ids is not None and custom_id not in eligible_ids:
                    # A newer quality/fresh/API-retry attempt owns this ID.  An
                    # old output may need to be re-downloaded for audit, but it
                    # must never restore stale pixels while the newer attempt is
                    # pending or after it has completed.
                    continue
                target = images_dir / plan[custom_id]["filename"]
                if not replace_existing and _valid_existing_image(target, plan[custom_id]["size"]):
                    state["items"][custom_id].update({
                        "status": "success",
                        "sha256": sha256_file(target),
                        "reused_existing": True,
                    })
                    continue
                raw = base64.b64decode(encoded, validate=True)
                tmp = target.with_suffix(target.suffix + ".tmp")
                expected = tuple(int(value) for value in plan[custom_id]["size"].split("x"))
                api_format = (state.get("api_request") or {}).get(
                    "output_format", OUTPUT_FORMAT
                )
                expected_api_format = "JPEG" if api_format == "jpeg" else "PNG"
                with Image.open(io.BytesIO(raw)) as source:
                    if source.format != expected_api_format or source.size != expected:
                        raise PipelineError(
                            "decoded API image is corrupt, has the wrong format, or wrong size"
                        )
                    if (target.suffix.lower() in {".jpg", ".jpeg"}
                            and expected_api_format == "JPEG"):
                        with tmp.open("wb") as output:
                            output.write(raw)
                    elif target.suffix.lower() in {".jpg", ".jpeg"}:
                        source.convert("RGB").save(
                            tmp,
                            STORAGE_FORMAT,
                            quality=JPEG_QUALITY,
                            optimize=True,
                        )
                    else:
                        # Compatibility for an in-flight plan created before JPEG
                        # q92 storage was enabled. It is compacted after approval.
                        with tmp.open("wb") as output:
                            output.write(raw)
                if not _valid_existing_image(tmp, plan[custom_id]["size"]):
                    tmp.unlink(missing_ok=True)
                    raise PipelineError("JPEG q92 storage conversion failed validation")
                tmp.replace(target)
                changed_ids.add(custom_id)
                state["items"][custom_id].update({
                    "status": "success",
                    "sha256": sha256_file(target),
                    "collected_at": utc_now(),
                })
            except (json.JSONDecodeError, PipelineError, binascii.Error, OSError, ValueError) as exc:
                if custom_id in state["items"]:
                    state["items"][custom_id].update({"status": "collect_error", "error": str(exc)})
                else:
                    state.setdefault("collection_errors", []).append({
                        "file": path.name, "line": line_number, "error": str(exc)
                    })
    return changed_ids


def process_error_jsonl(path: Path, state: dict[str, Any]) -> None:
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                state.setdefault("collection_errors", []).append({
                    "file": path.name, "line": line_number, "error": str(exc)
                })
                continue
            custom_id = row.get("custom_id")
            if custom_id in state["items"] and state["items"][custom_id].get("status") != "success":
                state["items"][custom_id].update({"status": "api_error", "api_error": row.get("error")})


def _write_hash_manifest(run_dir: Path, state: dict[str, Any]) -> None:
    seen: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for custom_id, item in sorted(state["items"].items()):
        path = run_dir / "images" / item.get("filename", custom_id + STORAGE_EXTENSION)
        if item.get("status") != "success" or not path.exists():
            continue
        digest = sha256_file(path)
        duplicate_of = seen.get(digest)
        seen.setdefault(digest, custom_id)
        item["sha256"] = digest
        item["duplicate_of"] = duplicate_of
        rows.append({
            "custom_id": custom_id,
            "filename": path.name,
            "sha256": digest,
            "duplicate_of": duplicate_of,
        })
    _write_jsonl(run_dir / "image_sha256.jsonl", rows)


def collect_results(run_dir: Path, client: Any | None = None,
                    prepare_review: bool = True, detector: bool = True) -> dict[str, int]:
    client = client or _client()
    state = load_state(run_dir)
    plan = read_plan(run_dir, state)
    pending_qa_ids = set(state.get("pending_qa_ids", []))
    latest_attempt_by_id: dict[str, dict[str, Any]] = {}
    latest_key_by_id: dict[str, tuple[str, int, int]] = {}
    for candidate_shard in state["shards"]:
        for candidate in candidate_shard["attempts"]:
            if candidate.get("superseded"):
                continue
            history = candidate.get("history") or []
            planned_at = str(history[0].get("at", "")) if history else ""
            key = (
                planned_at,
                int(candidate_shard["index"]),
                int(candidate.get("number", 0)),
            )
            for custom_id in candidate.get("custom_ids", []):
                if key >= latest_key_by_id.get(custom_id, ("", -1, -1)):
                    latest_key_by_id[custom_id] = key
                    latest_attempt_by_id[custom_id] = candidate
    for shard in state["shards"]:
        for attempt in shard["attempts"]:
            if not attempt.get("batch_id"):
                continue
            batch = client.batches.retrieve(attempt["batch_id"])
            _sync_attempt_from_batch(attempt, batch)
            save_state(run_dir, state)
            prefix = f"shard_{shard['index']:03d}_attempt_{attempt['number']:02d}"
            if attempt.get("output_file_id") and not attempt.get("local_output_pruned"):
                # A deliberately pruned response is not downloaded again.  The
                # error-file branch below remains independent and auditable.
                prior_output = attempt.get("local_output_path")
                if not prior_output or not (run_dir / prior_output).exists():
                    output_path = run_dir / f"{prefix}_output.jsonl"
                    _download_file(client, attempt["output_file_id"], output_path)
                    changed_ids = process_output_jsonl(
                        output_path, run_dir, state, plan,
                        replace_existing=(
                            attempt.get("kind") in {
                                "quality_retry", "fresh_replacement", "image_edit"
                            }
                            or bool(attempt.get("replace_existing"))
                        ),
                        eligible_ids={
                            custom_id for custom_id in attempt.get("custom_ids", [])
                            if latest_attempt_by_id.get(custom_id) is attempt
                        },
                    )
                    pending_qa_ids.update(changed_ids)
                    state["pending_qa_ids"] = sorted(pending_qa_ids)
                    compressed = _gzip_jsonl(output_path)
                    attempt["local_output_path"] = compressed.name
                    save_state(run_dir, state)
            if attempt.get("error_file_id"):
                prior_error = attempt.get("local_error_path")
                if not prior_error or not (run_dir / prior_error).exists():
                    error_path = run_dir / f"{prefix}_errors.jsonl"
                    _download_file(client, attempt["error_file_id"], error_path)
                    process_error_jsonl(error_path, state)
                    attempt["local_error_path"] = error_path.name
                    save_state(run_dir, state)
    _write_hash_manifest(run_dir, state)
    save_state(run_dir, state)
    counts = {
        "success": sum(item.get("status") == "success" for item in state["items"].values()),
        "total": len(state["items"]),
    }
    counts["missing"] = counts["total"] - counts["success"]
    if prepare_review and counts["success"]:
        from ..qa.gpt_head_review import prepare_review_artifacts
        # The initial collection has no QA manifest and therefore runs a full
        # detector pass.  Later corrections evaluate only newly replaced IDs;
        # the review code still recomputes global duplicate relationships.
        incremental_ids = pending_qa_ids if (run_dir / "auto_qa.jsonl").exists() else None
        review = prepare_review_artifacts(
            run_dir, use_detector=detector, custom_ids=incremental_ids
        )
        state["pending_qa_ids"] = []
        state["last_qa"] = {
            "completed_at": utc_now(),
            "mode": "incremental" if incremental_ids is not None else "full",
            "detector_images_evaluated": review["detector_images_evaluated"],
        }
        save_state(run_dir, state)
        counts.update({
            "quality_pass": review["quality_pass"],
            "quality_failed": review["quality_failed"],
            "quality_incomplete": review["quality_incomplete"],
        })
    return counts


def reprocess_local_output(run_dir: Path, shard_index: int,
                           attempt_number: int) -> dict[str, Any]:
    """Recover a downloaded output file locally without another API request."""
    state = load_state(run_dir)
    plan = read_plan(run_dir, state)
    try:
        shard = next(item for item in state["shards"] if item["index"] == shard_index)
        attempt = next(item for item in shard["attempts"] if item["number"] == attempt_number)
    except StopIteration as exc:
        raise PipelineError("requested shard/attempt is not recorded") from exc
    value = attempt.get("local_output_path")
    if not value:
        raise PipelineError("attempt has no downloaded local output")
    if attempt.get("local_output_pruned"):
        raise PipelineError(
            f"attempt output was deliberately pruned; see {OUTPUT_PRUNE_MANIFEST_NAME}"
        )
    source = run_dir / value
    if not source.exists():
        raise PipelineError(f"downloaded local output is missing: {source}")
    temporary = run_dir / f"reprocess_shard_{shard_index:03d}_attempt_{attempt_number:02d}.jsonl"
    if source.suffix == ".gz":
        with gzip.open(source, "rb") as compressed, temporary.open("wb") as output:
            shutil.copyfileobj(compressed, output, length=1024 * 1024)
    else:
        shutil.copy2(source, temporary)
    try:
        changed_ids = process_output_jsonl(
            temporary,
            run_dir,
            state,
            plan,
            replace_existing=attempt.get("kind") in {
                "quality_retry", "fresh_replacement", "image_edit"
            },
        )
    finally:
        temporary.unlink(missing_ok=True)
    _write_hash_manifest(run_dir, state)
    pending_qa_ids = set(state.get("pending_qa_ids", []))
    pending_qa_ids.update(changed_ids)
    state["pending_qa_ids"] = sorted(pending_qa_ids)
    save_state(run_dir, state)
    from ..qa.gpt_head_review import prepare_review_artifacts
    incremental_ids = pending_qa_ids if (run_dir / "auto_qa.jsonl").exists() else None
    review = prepare_review_artifacts(
        run_dir, use_detector=True, custom_ids=incremental_ids
    )
    state["pending_qa_ids"] = []
    state["last_qa"] = {
        "completed_at": utc_now(),
        "mode": "incremental" if incremental_ids is not None else "full",
        "detector_images_evaluated": review["detector_images_evaluated"],
    }
    save_state(run_dir, state)
    return {
        "success": sum(item.get("status") == "success" for item in state["items"].values()),
        "total": len(state["items"]),
        "quality_pass": review["quality_pass"],
        "quality_failed": review["quality_failed"],
        "quality_incomplete": review["quality_incomplete"],
    }


def compact_legacy_png_run(run_dir: Path) -> dict[str, Any]:
    """Convert an already approved PNG run to JPEG q92 and gzip API outputs.

    The immutable review artifacts continue to document the source PNG bytes.  A
    separate manifest maps each source SHA-256 to its verified JPEG derivative.
    Source PNGs are removed only after every derivative passes format/dimension
    validation.
    """
    if not (run_dir / "approval.json").exists():
        raise PipelineError("legacy compaction requires an approved run")
    state = load_state(run_dir)
    plan = read_plan(run_dir, state)
    target_dir = run_dir / "images_jpeg_q92"
    target_dir.mkdir(exist_ok=True)
    conversions: list[tuple[Path, Path, dict[str, Any]]] = []

    for custom_id, record in plan.items():
        source = run_dir / "images" / record["filename"]
        target = target_dir / f"{custom_id}{STORAGE_EXTENSION}"
        if not source.exists():
            raise PipelineError(f"approved source image is missing before compaction: {source}")
        expected = record["size"]
        tmp = target.with_suffix(target.suffix + ".tmp")
        with Image.open(source) as image:
            if image.format != "PNG" or f"{image.width}x{image.height}" != expected:
                raise PipelineError(f"legacy source is not the expected PNG: {source}")
            image.convert("RGB").save(
                tmp, STORAGE_FORMAT, quality=JPEG_QUALITY, optimize=True
            )
        if not _valid_existing_image(tmp, expected):
            tmp.unlink(missing_ok=True)
            raise PipelineError(f"JPEG q92 derivative failed validation: {target}")
        tmp.replace(target)
        conversions.append((source, target, {
            "custom_id": custom_id,
            "source_path": str(source.relative_to(run_dir)),
            "source_sha256": sha256_file(source),
            "target_path": str(target.relative_to(run_dir)),
            "target_sha256": sha256_file(target),
            "size": expected,
            "format": "JPEG",
            "quality": JPEG_QUALITY,
        }))

    rejected_root = run_dir / "rejected"
    if rejected_root.exists():
        for source in sorted(rejected_root.rglob("*.png")):
            target = source.with_suffix(STORAGE_EXTENSION)
            tmp = target.with_suffix(target.suffix + ".tmp")
            with Image.open(source) as image:
                size = f"{image.width}x{image.height}"
                if image.format != "PNG":
                    raise PipelineError(f"rejected source is not PNG: {source}")
                image.convert("RGB").save(
                    tmp, STORAGE_FORMAT, quality=JPEG_QUALITY, optimize=True
                )
            if not _valid_existing_image(tmp, size):
                tmp.unlink(missing_ok=True)
                raise PipelineError(f"rejected JPEG derivative failed validation: {target}")
            tmp.replace(target)
            conversions.append((source, target, {
                "custom_id": source.stem,
                "source_path": str(source.relative_to(run_dir)),
                "source_sha256": sha256_file(source),
                "target_path": str(target.relative_to(run_dir)),
                "target_sha256": sha256_file(target),
                "size": size,
                "format": "JPEG",
                "quality": JPEG_QUALITY,
                "rejected": True,
            }))

    # Every derivative is now durable and verified; only now remove the larger PNGs.
    for source, _target, _row in conversions:
        source.unlink()
    _write_jsonl(run_dir / "jpeg_q92_manifest.jsonl", (row for _, _, row in conversions))

    compressed_outputs = 0
    for shard in state["shards"]:
        for attempt in shard["attempts"]:
            value = attempt.get("local_output_path")
            if not value or value.endswith(".gz"):
                continue
            output = run_dir / value
            if output.exists() and output.suffix == ".jsonl":
                attempt["local_output_path"] = _gzip_jsonl(output).name
                compressed_outputs += 1
    state["derived_storage"] = {
        "format": "JPEG",
        "quality": JPEG_QUALITY,
        "accepted_images": len(plan),
        "converted_files": len(conversions),
        "manifest": "jpeg_q92_manifest.jsonl",
        "compacted_at": utc_now(),
    }
    save_state(run_dir, state)
    return {
        "accepted_images": len(plan),
        "converted_files": len(conversions),
        "compressed_outputs": compressed_outputs,
        "target_dir": str(target_dir),
    }


def _safe_run_file(run_dir: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.name != value:
        raise PipelineError(f"unsafe run-relative output path: {value!r}")
    return run_dir / relative


def _verify_prune_readiness(run_dir: Path, state: dict[str, Any]) -> None:
    active = [
        attempt.get("batch_id")
        for shard in state["shards"]
        for attempt in shard["attempts"]
        if attempt.get("status") in ACTIVE_STATUSES
    ]
    if active:
        raise PipelineError(f"cannot prune while Batch is active: {active[0]}")
    if state.get("pending_qa_ids"):
        raise PipelineError("cannot prune while replacement images are pending QA")
    if any(item.get("status") != "success" for item in state["items"].values()):
        raise PipelineError("cannot prune before every planned image is collected")

    expected_ids = set(state["items"])
    allow_approved_validation_failures = state.get("stage") == "validation"
    validation_approval_checked = False
    qa_path = run_dir / "auto_qa.jsonl"
    if not qa_path.exists():
        raise PipelineError("cannot prune without auto_qa.jsonl")
    qa_ids: set[str] = set()
    with qa_path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"invalid auto_qa.jsonl line {number}: {exc}") from exc
            custom_id = row.get("custom_id")
            if custom_id in qa_ids:
                raise PipelineError(f"duplicate QA custom_id: {custom_id}")
            qa_ids.add(custom_id)
            if not row.get("quality_gate_complete"):
                raise PipelineError(f"cannot prune before hard QA completes: {custom_id}")
            if not row.get("quality_gate_pass"):
                if not allow_approved_validation_failures:
                    raise PipelineError(f"cannot prune before hard QA passes: {custom_id}")
                if not validation_approval_checked:
                    _approval_ok(run_dir, "validation")
                    validation_approval_checked = True
    if qa_ids != expected_ids:
        raise PipelineError("auto_qa.jsonl does not cover the complete plan")

    derived = state.get("derived_storage") or {}
    if derived:
        manifest_path = run_dir / str(derived.get("manifest", ""))
        if not manifest_path.exists():
            raise PipelineError("JPEG derivative manifest is missing")
        verified_ids: set[str] = set()
        with manifest_path.open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, 1):
                row = json.loads(line)
                if row.get("rejected"):
                    continue
                custom_id = row.get("custom_id")
                target_value = row.get("target_path")
                if custom_id in verified_ids or not isinstance(target_value, str):
                    raise PipelineError(f"invalid JPEG manifest line {number}")
                target = run_dir / target_value
                if not target.exists() or sha256_file(target) != row.get("target_sha256"):
                    raise PipelineError(f"retained JPEG checksum mismatch: {target}")
                verified_ids.add(custom_id)
    else:
        manifest_path = run_dir / "image_sha256.jsonl"
        if not manifest_path.exists():
            raise PipelineError("image_sha256.jsonl is missing")
        verified_ids = set()
        with manifest_path.open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, 1):
                row = json.loads(line)
                custom_id = row.get("custom_id")
                filename = row.get("filename")
                if custom_id in verified_ids or not isinstance(filename, str):
                    raise PipelineError(f"invalid image checksum manifest line {number}")
                target = run_dir / "images" / filename
                if not target.exists() or sha256_file(target) != row.get("sha256"):
                    raise PipelineError(f"retained image checksum mismatch: {target}")
                verified_ids.add(custom_id)
    if verified_ids != expected_ids:
        raise PipelineError("retained image manifest does not cover the complete plan")


def _output_audit(path: Path) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_text_tokens": 0,
        "input_image_tokens": 0,
        "output_text_tokens": 0,
        "output_image_tokens": 0,
    }
    custom_ids = hashlib.sha256()
    rows = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for number, line in enumerate(fh, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PipelineError(f"invalid output JSONL {path.name}:{number}: {exc}") from exc
                rows += 1
                custom_id = row.get("custom_id")
                if not isinstance(custom_id, str):
                    raise PipelineError(f"missing custom_id in {path.name}:{number}")
                custom_ids.update(custom_id.encode("utf-8") + b"\n")
                response = row.get("response") or {}
                status = str(response.get("status_code", "missing"))
                status_counts[status] = status_counts.get(status, 0) + 1
                usage = (response.get("body") or {}).get("usage") or {}
                input_details = usage.get("input_tokens_details") or {}
                output_details = usage.get("output_tokens_details") or {}
                usage_totals["input_tokens"] += int(usage.get("input_tokens") or 0)
                usage_totals["output_tokens"] += int(usage.get("output_tokens") or 0)
                usage_totals["total_tokens"] += int(usage.get("total_tokens") or 0)
                usage_totals["input_text_tokens"] += int(input_details.get("text_tokens") or 0)
                usage_totals["input_image_tokens"] += int(input_details.get("image_tokens") or 0)
                usage_totals["output_text_tokens"] += int(output_details.get("text_tokens") or 0)
                usage_totals["output_image_tokens"] += int(output_details.get("image_tokens") or 0)
    except (OSError, EOFError) as exc:
        raise PipelineError(f"corrupt gzip output {path}: {exc}") from exc
    if rows == 0:
        raise PipelineError(f"refusing to prune empty output file: {path}")
    return {
        "compressed_bytes": path.stat().st_size,
        "compressed_sha256": sha256_file(path),
        "rows": rows,
        "custom_ids_ordered_sha256": custom_ids.hexdigest(),
        "response_status_counts": dict(sorted(status_counts.items())),
        "usage": usage_totals,
    }


def _finish_output_prune(run_dir: Path, state: dict[str, Any],
                         manifest: dict[str, Any]) -> dict[str, Any]:
    for entry in manifest["files"]:
        path = _safe_run_file(run_dir, entry["path"])
        if path.exists():
            if sha256_file(path) != entry["compressed_sha256"]:
                raise PipelineError(f"output changed after prune audit: {path}")
            path.unlink()
    completed_at = utc_now()
    manifest["status"] = "completed"
    manifest["completed_at"] = completed_at
    _atomic_json(run_dir / OUTPUT_PRUNE_MANIFEST_NAME, manifest)
    manifest_sha256 = sha256_file(run_dir / OUTPUT_PRUNE_MANIFEST_NAME)
    for shard in state["shards"]:
        for attempt in shard["attempts"]:
            if attempt.get("local_output_pruned"):
                attempt["local_output_pruned_at"] = completed_at
    state["output_pruning"] = {
        "status": "completed",
        "completed_at": completed_at,
        "manifest": OUTPUT_PRUNE_MANIFEST_NAME,
        "manifest_sha256": manifest_sha256,
        "files": manifest["aggregate"]["files"],
        "bytes_freed": manifest["aggregate"]["compressed_bytes"],
    }
    save_state(run_dir, state)
    return {
        "files_deleted": manifest["aggregate"]["files"],
        "bytes_freed": manifest["aggregate"]["compressed_bytes"],
        "rows_preserved_in_manifest": manifest["aggregate"]["rows"],
        "manifest": str(run_dir / OUTPUT_PRUNE_MANIFEST_NAME),
    }


def prune_batch_outputs(run_dir: Path) -> dict[str, Any]:
    """Audit and remove downloaded Batch output payloads without re-downloads."""
    state = load_state(run_dir)
    manifest_path = run_dir / OUTPUT_PRUNE_MANIFEST_NAME
    pruning = state.get("output_pruning") or {}
    if pruning.get("status") in {"deleting", "completed"}:
        if not manifest_path.exists():
            raise PipelineError("output prune state exists but its manifest is missing")
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        if pruning.get("status") == "completed":
            if any(_safe_run_file(run_dir, entry["path"]).exists()
                   for entry in manifest["files"]):
                raise PipelineError("a pruned output unexpectedly exists again")
            return {
                "files_deleted": manifest["aggregate"]["files"],
                "bytes_freed": manifest["aggregate"]["compressed_bytes"],
                "rows_preserved_in_manifest": manifest["aggregate"]["rows"],
                "manifest": str(manifest_path),
                "already_pruned": True,
            }
        return _finish_output_prune(run_dir, state, manifest)

    _verify_prune_readiness(run_dir, state)
    attempt_by_path: dict[str, tuple[int, dict[str, Any]]] = {}
    for shard in state["shards"]:
        for attempt in shard["attempts"]:
            value = attempt.get("local_output_path")
            if not value:
                continue
            path = _safe_run_file(run_dir, value)
            if path.suffixes[-2:] != [".jsonl", ".gz"]:
                raise PipelineError(f"refusing to prune non-gzip output: {path}")
            if value in attempt_by_path:
                raise PipelineError(f"duplicate local output reference: {value}")
            attempt_by_path[value] = (int(shard["index"]), attempt)
    disk_paths = {path.name for path in run_dir.glob("shard_*_output.jsonl.gz")}
    if disk_paths != set(attempt_by_path):
        raise PipelineError("on-disk Batch outputs do not exactly match state references")
    if not disk_paths:
        raise PipelineError("no downloaded Batch output files exist to prune")

    files: list[dict[str, Any]] = []
    aggregate_usage: dict[str, int] = {}
    total_bytes = 0
    total_rows = 0
    for value in sorted(disk_paths):
        shard_index, attempt = attempt_by_path[value]
        audit = _output_audit(run_dir / value)
        entry = {
            "path": value,
            "shard": shard_index,
            "attempt": int(attempt["number"]),
            "batch_id": attempt.get("batch_id"),
            "output_file_id": attempt.get("output_file_id"),
            **audit,
        }
        files.append(entry)
        total_bytes += audit["compressed_bytes"]
        total_rows += audit["rows"]
        for key, value_count in audit["usage"].items():
            aggregate_usage[key] = aggregate_usage.get(key, 0) + int(value_count)
    manifest = {
        "schema_version": 1,
        "status": "verified_pending_delete",
        "created_at": utc_now(),
        "local_batch_id": state["local_batch_id"],
        "stage": state["stage"],
        "files": files,
        "aggregate": {
            "files": len(files),
            "compressed_bytes": total_bytes,
            "rows": total_rows,
            "usage": aggregate_usage,
        },
        "retained_artifacts": {
            name: sha256_file(run_dir / name)
            for name in [state["plan_path"], "auto_qa.jsonl", "image_sha256.jsonl"]
            if (run_dir / name).exists()
        },
    }
    _atomic_json(manifest_path, manifest)
    entry_by_path = {entry["path"]: entry for entry in files}
    for value, (_shard_index, attempt) in attempt_by_path.items():
        entry = entry_by_path[value]
        attempt.update({
            "local_output_pruned": True,
            "local_output_sha256": entry["compressed_sha256"],
            "local_output_bytes": entry["compressed_bytes"],
            "local_output_rows": entry["rows"],
            "local_output_prune_manifest": OUTPUT_PRUNE_MANIFEST_NAME,
        })
    state["output_pruning"] = {
        "status": "deleting",
        "started_at": utc_now(),
        "manifest": OUTPUT_PRUNE_MANIFEST_NAME,
        "files": len(files),
        "bytes_to_free": total_bytes,
    }
    save_state(run_dir, state)
    return _finish_output_prune(run_dir, state, manifest)


def _read_input_archive_manifest(archive_path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            corrupt = bundle.testzip()
            if corrupt:
                raise PipelineError(f"corrupt member in Batch input archive: {corrupt}")
            manifest = json.loads(bundle.read(INPUT_ARCHIVE_MANIFEST_NAME))
            expected = {entry["member"] for entry in manifest.get("files", [])}
            actual = set(bundle.namelist()) - {INPUT_ARCHIVE_MANIFEST_NAME}
            if actual != expected:
                raise PipelineError("Batch input archive members do not match its manifest")
            for entry in manifest["files"]:
                payload = bundle.read(entry["member"])
                if (len(payload) != entry["bytes"]
                        or hashlib.sha256(payload).hexdigest() != entry["sha256"]):
                    raise PipelineError(
                        f"Batch input archive checksum mismatch: {entry['member']}"
                    )
    except (KeyError, OSError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise PipelineError(f"invalid Batch input archive {archive_path}: {exc}") from exc
    return manifest


def _finish_input_archive(run_dir: Path, state: dict[str, Any],
                          manifest: dict[str, Any]) -> dict[str, Any]:
    entry_by_member = {entry["member"]: entry for entry in manifest["files"]}
    for shard in state["shards"]:
        for attempt in shard["attempts"]:
            member = str(attempt["input_path"])
            entry = entry_by_member.get(member)
            if entry is None:
                raise PipelineError(f"attempt input missing from archive manifest: {member}")
            path = _safe_run_file(run_dir, member)
            if path.exists():
                if sha256_file(path) != entry["sha256"]:
                    raise PipelineError(f"Batch input changed after archive verification: {path}")
                path.unlink()
            attempt["input_archive_member"] = member
    archive_path = run_dir / INPUT_ARCHIVE_NAME
    completed_at = utc_now()
    state["input_archiving"] = {
        "status": "completed",
        "completed_at": completed_at,
        "archive": INPUT_ARCHIVE_NAME,
        "archive_sha256": sha256_file(archive_path),
        "files": manifest["aggregate"]["files"],
        "original_bytes": manifest["aggregate"]["original_bytes"],
        "archive_bytes": archive_path.stat().st_size,
    }
    save_state(run_dir, state)
    return {
        "files_archived": manifest["aggregate"]["files"],
        "original_bytes": manifest["aggregate"]["original_bytes"],
        "archive_bytes": archive_path.stat().st_size,
        "archive": str(archive_path),
    }


def archive_batch_inputs(run_dir: Path) -> dict[str, Any]:
    """Combine per-attempt Batch JSONL inputs into one exact, readable ZIP."""
    state = load_state(run_dir)
    if any(
        attempt.get("status") in ACTIVE_STATUSES
        for shard in state["shards"] for attempt in shard["attempts"]
    ):
        raise PipelineError("cannot archive Batch inputs while a Batch is active")
    attempts = [
        (int(shard["index"]), attempt)
        for shard in state["shards"] for attempt in shard["attempts"]
    ]
    members = [str(attempt["input_path"]) for _shard, attempt in attempts]
    if len(members) != len(set(members)):
        raise PipelineError("duplicate Batch input paths cannot be archived")

    archive_path = run_dir / INPUT_ARCHIVE_NAME
    archiving = state.get("input_archiving") or {}
    original_paths = [_safe_run_file(run_dir, member) for member in members]
    if archiving.get("status") == "completed" and not any(
        path.exists() for path in original_paths
    ):
        if not archive_path.exists():
            raise PipelineError("Batch input archive state exists but archive is missing")
        if sha256_file(archive_path) != archiving.get("archive_sha256"):
            raise PipelineError("Batch input archive checksum changed")
        manifest = _read_input_archive_manifest(archive_path)
        return {
            "files_archived": manifest["aggregate"]["files"],
            "original_bytes": manifest["aggregate"]["original_bytes"],
            "archive_bytes": archive_path.stat().st_size,
            "archive": str(archive_path),
            "already_archived": True,
        }
    if archiving.get("status") == "deleting":
        if not archive_path.exists():
            raise PipelineError("Batch input archive is missing during crash recovery")
        if sha256_file(archive_path) != archiving.get("archive_sha256"):
            raise PipelineError("Batch input archive changed during crash recovery")
        return _finish_input_archive(
            run_dir, state, _read_input_archive_manifest(archive_path)
        )

    files: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    total_bytes = 0
    total_rows = 0
    for shard_index, attempt in attempts:
        member = str(attempt["input_path"])
        payload = _input_payload(run_dir, state, attempt)
        endpoint = str(attempt.get("endpoint") or ENDPOINT)
        custom_ids = _validate_jsonl_bytes(
            payload, member, state.get("api_request"), endpoint
        )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != attempt.get("input_sha256") or custom_ids != attempt.get("custom_ids"):
            raise PipelineError(f"Batch input does not match state: {member}")
        payloads[member] = payload
        files.append({
            "member": member,
            "sha256": digest,
            "bytes": len(payload),
            "rows": len(custom_ids),
            "shard": shard_index,
            "attempt": int(attempt["number"]),
            "batch_id": attempt.get("batch_id"),
            "input_file_id": attempt.get("input_file_id"),
        })
        total_bytes += len(payload)
        total_rows += len(custom_ids)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "local_batch_id": state["local_batch_id"],
        "stage": state["stage"],
        "files": files,
        "aggregate": {
            "files": len(files),
            "rows": total_rows,
            "original_bytes": total_bytes,
        },
    }
    tmp = archive_path.with_suffix(archive_path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as bundle:
            for member in sorted(payloads):
                bundle.writestr(member, payloads[member])
            bundle.writestr(
                INPUT_ARCHIVE_MANIFEST_NAME,
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
        tmp.replace(archive_path)
    finally:
        tmp.unlink(missing_ok=True)
    _read_input_archive_manifest(archive_path)
    archive_sha256 = sha256_file(archive_path)
    for _shard_index, attempt in attempts:
        attempt["input_archive_member"] = str(attempt["input_path"])
    state["input_archiving"] = {
        "status": "deleting",
        "started_at": utc_now(),
        "archive": INPUT_ARCHIVE_NAME,
        "archive_sha256": archive_sha256,
        "files": len(files),
        "original_bytes": total_bytes,
        "archive_bytes": archive_path.stat().st_size,
    }
    save_state(run_dir, state)
    return _finish_input_archive(run_dir, state, manifest)


def _missing_ids(run_dir: Path, state: dict[str, Any], plan: dict[str, dict[str, Any]]) -> set[str]:
    missing: set[str] = set()
    for custom_id, record in plan.items():
        path = run_dir / "images" / record["filename"]
        if not _valid_existing_image(path, record["size"]):
            missing.add(custom_id)
        else:
            state["items"][custom_id].update({"status": "success", "sha256": sha256_file(path)})
    return missing


def _latest_request_for_id(run_dir: Path, state: dict[str, Any], shard: dict[str, Any],
                           custom_id: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Reuse a corrected prompt when an API retry follows a quality retry."""
    for attempt in reversed(shard["attempts"]):
        path = _safe_run_file(run_dir, str(attempt["input_path"]))
        if not path.exists() and not attempt.get("input_archive_member"):
            continue
        payload = _input_payload(run_dir, state, attempt)
        for line in payload.decode("utf-8").splitlines():
            request = json.loads(line)
            if request.get("custom_id") == custom_id:
                validate_batch_request(
                    request, state.get("api_request"), _attempt_endpoint(attempt)
                )
                return request
    return _batch_request(fallback, state.get("api_request"))


def prepare_resume(run_dir: Path) -> int:
    state = load_state(run_dir)
    if _supersede_closed_gate_resume_artifacts(state):
        save_state(run_dir, state)
    token_limited = [
        shard for shard in state.get("shards", [])
        if not shard.get("retired")
        and any(
            not attempt.get("superseded")
            and attempt.get("status") == "failed"
            and _attempt_has_batch_error(attempt, "token_limit_exceeded")
            for attempt in shard.get("attempts", [])
        )
    ]
    if token_limited:
        indices = ", ".join(str(shard["index"]) for shard in token_limited)
        raise PipelineError(
            "Batch enqueued-token limit rejected shard(s) "
            f"{indices}; run split-token-limit instead of retrying the same input"
        )
    plan = read_plan(run_dir, state)
    missing = _missing_ids(run_dir, state, plan)
    # The same custom_id can occur in its original shard and in a later
    # aggregate correction shard.  Resume only the globally newest attempt
    # that owns each missing ID.  In particular, never create an original-
    # prompt retry while a newer quality/fresh replacement is still active.
    owner_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    owner_key_by_id: dict[str, tuple[str, int, int]] = {}
    for candidate_shard in state["shards"]:
        for candidate in candidate_shard["attempts"]:
            if candidate.get("superseded"):
                continue
            history = candidate.get("history") or []
            planned_at = str(history[0].get("at", "")) if history else ""
            key = (
                planned_at,
                int(candidate_shard["index"]),
                int(candidate.get("number", 0)),
            )
            for custom_id in candidate.get("custom_ids", []):
                if key >= owner_key_by_id.get(custom_id, ("", -1, -1)):
                    owner_key_by_id[custom_id] = key
                    owner_by_id[custom_id] = (candidate_shard, candidate)
    created = 0
    for shard in state["shards"]:
        if not _submission_allowed(state, shard):
            continue
        if not shard["attempts"]:
            # A repair-only shard may be durably visible before its first
            # quality-retry attempt is written.  Let auto-repair reconstruct
            # that attempt instead of turning an interrupted save into an
            # IndexError or an original-prompt API retry.
            continue
        retry_ids = []
        latest_by_id: dict[str, dict[str, Any]] = {}
        for custom_id in shard["custom_ids"]:
            owner = owner_by_id.get(custom_id)
            if custom_id not in missing or owner is None or owner[0] is not shard:
                continue
            owner_attempt = owner[1]
            if owner_attempt.get("status") in ACTIVE_STATUSES:
                continue
            retry_ids.append(custom_id)
            latest_by_id[custom_id] = owner_attempt
        if not retry_ids:
            continue
        pending_attempts = {
            id(attempt): attempt for attempt in latest_by_id.values()
            if not attempt.get("batch_id") and attempt.get("status") == "planned"
        }
        if pending_attempts:
            if len(pending_attempts) != 1:
                raise PipelineError(
                    f"missing IDs for shard {shard['index']} have multiple unsent retry owners"
                )
            pending = next(iter(pending_attempts.values()))
            if set(pending["custom_ids"]) != set(retry_ids):
                raise PipelineError(
                    f"unsent retry input for shard {shard['index']} no longer matches missing IDs"
                )
            created += len(retry_ids)
            continue
        number = len(shard["attempts"])
        input_name = f"batch_input_{shard['index']:03d}_attempt_{number:02d}.jsonl"
        input_path = run_dir / input_name
        retry_requests = [
            _latest_request_for_id(run_dir, state, shard, custom_id, plan[custom_id])
            for custom_id in retry_ids
        ]
        endpoints = {str(request.get("url")) for request in retry_requests}
        if len(endpoints) != 1 or next(iter(endpoints)) not in {ENDPOINT, EDIT_ENDPOINT}:
            raise PipelineError(
                f"API retry for shard {shard['index']} mixes unsupported endpoints"
            )
        endpoint = next(iter(endpoints))
        _write_jsonl(input_path, retry_requests)
        validate_jsonl(input_path, state.get("api_request"), endpoint)
        shard["attempts"].append({
            "number": number,
            "kind": "api_retry",
            "endpoint": endpoint,
            "replace_existing": (
                any(attempt.get("kind") in {
                        "quality_retry", "fresh_replacement", "image_edit"
                    }
                    or bool(attempt.get("replace_existing"))
                    for attempt in latest_by_id.values())
            ),
            "input_path": input_name,
            "input_sha256": sha256_file(input_path),
            "custom_ids": retry_ids,
            "input_file_id": None,
            "batch_id": None,
            "status": "planned",
            "output_file_id": None,
            "error_file_id": None,
            "request_counts": None,
            "history": [{"at": utc_now(), "status": "planned_retry"}],
        })
        created += len(retry_ids)
    save_state(run_dir, state)
    return created


def split_token_limited_shard(run_dir: Path, max_requests: int = 500) -> dict[str, Any]:
    """Replace one zero-progress token-limited shard with sequential sub-shards."""
    if max_requests < 1 or max_requests > 500:
        raise PipelineError("max requests per token-limit shard must be between 1 and 500")
    state = load_state(run_dir)
    if (run_dir / "approval.json").exists():
        raise PipelineError("run is already approved")
    for shard in state.get("shards", []):
        for attempt in shard.get("attempts", []):
            if not attempt.get("superseded") and attempt.get("status") in ACTIVE_STATUSES:
                raise PipelineError(f"Batch {attempt.get('batch_id')} is still {attempt['status']}")

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for shard in state.get("shards", []):
        if shard.get("retired"):
            continue
        attempts = [
            attempt for attempt in shard.get("attempts", [])
            if not attempt.get("superseded")
        ]
        if not attempts:
            continue
        latest = attempts[-1]
        counts = latest.get("request_counts") or {}
        if (latest.get("status") == "failed"
                and _attempt_has_batch_error(latest, "token_limit_exceeded")
                and int(counts.get("completed", 0) or 0) == 0
                and int(counts.get("failed", 0) or 0) == 0):
            candidates.append((shard, latest))
    if len(candidates) != 1:
        raise PipelineError(
            "split-token-limit requires exactly one unsuperseded, zero-progress "
            f"token-limited shard; found {len(candidates)}"
        )

    source_shard, source_attempt = candidates[0]
    payload = _input_payload(run_dir, state, source_attempt)
    custom_ids = _validate_jsonl_bytes(
        payload, str(source_attempt["input_path"]), state.get("api_request")
    )
    if custom_ids != source_shard.get("custom_ids"):
        raise PipelineError("token-limited shard input IDs do not match durable state")
    rows = [
        json.loads(line) for line in payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(custom_ids):
        raise PipelineError("token-limited shard input row count is inconsistent")

    first_index = max(
        (int(shard["index"]) for shard in state.get("shards", [])), default=-1
    ) + 1
    chunks = [rows[start:start + max_requests] for start in range(0, len(rows), max_requests)]
    group = (
        f"token-limit-{source_shard['index']}-"
        f"{str(source_attempt['input_sha256'])[:12]}"
    )
    new_shards: list[dict[str, Any]] = []
    for position, chunk in enumerate(chunks):
        shard_index = first_index + position
        input_name = f"batch_input_{shard_index:03d}_attempt_00.jsonl"
        input_path = run_dir / input_name
        _atomic_write_jsonl(input_path, chunk)
        validate_jsonl(input_path, state.get("api_request"))
        chunk_ids = [str(row["custom_id"]) for row in chunk]
        attempt = {
            "number": 0,
            "kind": "token_limit_reshard",
            "input_path": input_name,
            "input_sha256": sha256_file(input_path),
            "custom_ids": chunk_ids,
            "input_file_id": None,
            "batch_id": None,
            "status": "planned",
            "output_file_id": None,
            "error_file_id": None,
            "request_counts": None,
            "history": [{
                "at": utc_now(),
                "status": "planned_token_limit_reshard",
                "source_shard": int(source_shard["index"]),
            }],
        }
        new_shards.append({
            "index": shard_index,
            "custom_ids": chunk_ids,
            "attempts": [attempt],
            "token_limit_reshard": True,
            "sequential_group": group,
            "sequence_position": position,
            "sequence_total": len(chunks),
            "source_shard": int(source_shard["index"]),
        })

    replacement_indices = [int(shard["index"]) for shard in new_shards]
    source_shard.update({
        "retired": True,
        "retired_reason": "token_limit_exceeded",
        "replacement_shard_indices": replacement_indices,
    })
    for attempt in source_shard.get("attempts", []):
        attempt["superseded"] = True
        attempt.setdefault("history", []).append({
            "at": utc_now(),
            "status": "superseded",
            "reason": "zero-progress Batch exceeded the organization enqueued-token limit",
            "replacement_shard_indices": replacement_indices,
        })
    state["shards"].extend(new_shards)
    event = {
        "at": utc_now(),
        "source_shard": int(source_shard["index"]),
        "source_requests": len(rows),
        "max_requests": max_requests,
        "sequential_group": group,
        "replacement_shards": replacement_indices,
        "replacement_counts": [len(shard["custom_ids"]) for shard in new_shards],
    }
    state.setdefault("token_limit_reshard_history", []).append(event)
    save_state(run_dir, state)
    return event


def _read_auto_qa(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "auto_qa.jsonl"
    if not path.exists():
        raise PipelineError("auto_qa.jsonl is missing; collect results with the detector first")
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"{path}:{number}: {exc}") from exc
            rows[row["custom_id"]] = row
    return rows


def _human_quality_failures(run_dir: Path) -> dict[str, list[str]]:
    path = run_dir / "human_review.csv"
    if not path.exists():
        return {}
    failures: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            custom_id = row.get("custom_id", "")
            image_path = run_dir / "images" / row.get("filename", "")
            if not custom_id or not image_path.exists():
                continue
            # A review decision is valid only for the exact bytes that were reviewed.
            if row.get("reviewed_sha256") != sha256_file(image_path):
                continue
            reasons: list[str] = []
            if row.get("photorealism") == "fail":
                reasons.append("human_photorealism")
            if row.get("framing") == "fail":
                reasons.append("human_framing")
            if row.get("roll_no_back") == "fail":
                reasons.append("human_roll_or_back")
            if row.get("body_integrity") == "fail":
                reasons.append("human_body_integrity")
            if row.get("intent_match") == "off-by-one-bin":
                reasons.append("human_intent_off_by_one")
            elif row.get("intent_match") == "wrong":
                reasons.append("human_intent_wrong")
            if reasons:
                failures[custom_id] = reasons
    return failures


def _add_repair_only_shards(state: dict[str, Any], custom_ids: list[str],
                             *, shard_size: int = 500) -> None:
    """Give reused plan items a shard so selective retries can address them.

    Production's initial nine shards intentionally contain only the 4,500 new
    requests.  A reused Pilot image can nevertheless fail a later full-run QA.
    Such an item has no shard membership, so add a durable repair-only shard
    before constructing its first retry attempt.
    """
    assigned = {
        custom_id
        for shard in state["shards"]
        for custom_id in shard["custom_ids"]
    }
    unassigned = [custom_id for custom_id in custom_ids if custom_id not in assigned]
    if not unassigned:
        return
    next_index = max((shard["index"] for shard in state["shards"]), default=-1) + 1
    for start in range(0, len(unassigned), shard_size):
        state["shards"].append({
            "index": next_index,
            "custom_ids": unassigned[start:start + shard_size],
            "attempts": [],
            "repair_only": True,
        })
        next_index += 1


def _quality_retry_request(record: dict[str, Any], reasons: list[str], *,
                           base_request: dict[str, Any] | None = None,
                           round_number: int = 1,
                           extra_instruction: str = "",
                           api_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    corrected = dict(record)
    geometry_reasons = {"head_too_small", "head_too_large", "insufficient_margin"}
    instructions = [
        QUALITY_CORRECTIONS[reason] for reason in reasons
        if reason in QUALITY_CORRECTIONS and reason not in geometry_reasons
    ]
    # Always restart a quality correction from the immutable plan prompt.  Using
    # the previous corrected request caused instructions and numeric ranges to
    # accumulate across rounds.  Also neutralize plans created before the base
    # framing text stopped telling every scene to move the camera closer.
    del base_request  # API resume still reuses the exact corrective request.
    base_prompt = record["prompt"].replace(
        "Move the camera closer instead of using a distant wide composition.",
        "Choose camera distance solely to satisfy the numeric framing constraints.",
    )
    reason_set = set(reasons)
    geometry = ""
    if "head_too_large" in reason_set:
        geometry = (
            " COMPOSITION OVERRIDE (takes priority over earlier framing language): "
            "this must be an upper-torso portrait, never a face close-up. Move the camera "
            "significantly farther back and target complete-head height of 28% to 32% of "
            "the image. Keep the subject centered, with at least 20% of image height empty "
            "above the crown and below the chin and at least one-half head width clear on "
            "both sides. Do not aim near the 50% hard maximum."
        )
    elif "head_too_small" in reason_set:
        geometry = (
            " COMPOSITION OVERRIDE (takes priority over earlier framing language): "
            "use a medium-close upper-torso portrait, never a full-body or distant view. "
            "Move the camera closer and target complete-head height of 36% to 40% of the "
            "image, while keeping the full crown and chin inside frame with clear margin. "
            "Never fall below the 25% hard minimum."
        )
    elif "insufficient_margin" in reason_set:
        geometry = (
            " COMPOSITION OVERRIDE (takes priority over earlier framing language): "
            "recenter the complete head and use an upper-torso portrait with head height "
            "of 30% to 34% of the image. Leave at least 20% of image height empty above "
            "the crown and below the chin and at least one-half head width clear on both "
            "sides; no part of the hair, skull, or chin may approach a frame edge."
        )
    corrected["prompt"] = (
        base_prompt
        + f" QUALITY CORRECTION ROUND {round_number}: "
        + " ".join(instructions)
        + geometry
        + (
            " RUN-SPECIFIC CORRECTION (highest priority; supersedes any conflicting "
            f"numeric range above): {extra_instruction.strip()}"
            if extra_instruction.strip() else ""
        )
    )
    return _batch_request(corrected, api_profile)


def set_retry_prompt_override(run_dir: Path, custom_ids: list[str],
                              instruction: str) -> dict[str, Any]:
    """Persist an audited prompt correction for only the named requests."""
    instruction = instruction.strip()
    if not instruction:
        raise PipelineError("retry prompt override instruction must not be empty")
    state = load_state(run_dir)
    if (run_dir / "approval.json").exists():
        raise PipelineError("run is already approved")
    requested_ids = list(dict.fromkeys(custom_ids))
    unknown = [custom_id for custom_id in requested_ids if custom_id not in state["items"]]
    if unknown:
        raise PipelineError(f"unknown custom_id(s): {', '.join(unknown)}")
    overrides = state.setdefault("retry_prompt_overrides", {})
    for custom_id in requested_ids:
        overrides[custom_id] = instruction
    state.setdefault("retry_prompt_override_history", []).append({
        "at": utc_now(),
        "custom_ids": requested_ids,
        "instruction": instruction,
    })
    save_state(run_dir, state)
    return {"custom_ids": requested_ids, "instruction": instruction}


def extend_quality_retry_limit(run_dir: Path, max_retries: int) -> dict[str, Any]:
    """Record an explicit, bounded manual extension after automatic retries exhaust."""
    if max_retries < 1 or max_retries > 6:
        raise PipelineError("extended max retries must be between 1 and 6")
    state = load_state(run_dir)
    if (run_dir / "approval.json").exists():
        raise PipelineError("run is already approved")
    for shard in state["shards"]:
        attempts = [attempt for attempt in shard["attempts"] if not attempt.get("superseded")]
        if not attempts:
            continue
        latest = attempts[-1]
        if latest.get("status") in ACTIVE_STATUSES:
            raise PipelineError(f"Batch {latest.get('batch_id')} is still {latest['status']}")
    policy = state.setdefault("auto_correction", {})
    previous = int(policy.get("max_quality_retries", 0))
    if max_retries <= previous:
        raise PipelineError(f"extended max retries must exceed current limit {previous}")
    policy["max_quality_retries"] = max_retries
    state.setdefault("manual_retry_extensions", []).append({
        "at": utc_now(),
        "previous_max": previous,
        "new_max": max_retries,
        "reason": "replace machine-QA failures before Pilot approval",
    })
    save_state(run_dir, state)
    return {"previous_max": previous, "new_max": max_retries}


def extend_fresh_replacement_limit(run_dir: Path, max_rounds: int) -> dict[str, Any]:
    """Explicitly extend the bounded fresh-replacement limit with an audit trail."""
    if max_rounds < 1 or max_rounds > 99:
        raise PipelineError("extended fresh replacement limit must be between 1 and 99")
    state = load_state(run_dir)
    if (run_dir / "approval.json").exists():
        raise PipelineError("run is already approved")
    for shard in state["shards"]:
        attempts = [attempt for attempt in shard["attempts"] if not attempt.get("superseded")]
        if not attempts:
            continue
        latest = attempts[-1]
        if latest.get("status") in ACTIVE_STATUSES:
            raise PipelineError(f"Batch {latest.get('batch_id')} is still {latest['status']}")
    previous = int(state.get("fresh_replacement_max_rounds", 3))
    completed = int(state.get("fresh_replacement_rounds", 0))
    if max_rounds <= max(previous, completed):
        raise PipelineError(
            f"extended fresh replacement limit must exceed current limit {max(previous, completed)}"
        )
    state["fresh_replacement_max_rounds"] = max_rounds
    state.setdefault("fresh_replacement_limit_history", []).append({
        "at": utc_now(),
        "previous_max": previous,
        "new_max": max_rounds,
        "reason": "retry run-specific framing failures without replacing passing images",
    })
    save_state(run_dir, state)
    return {"previous_max": previous, "new_max": max_rounds}


def prepare_fresh_replacement(run_dir: Path) -> dict[str, Any]:
    """Reset stubborn failures in one aggregate Batch.

    Fresh replacement counts are normally small.  Keeping them in one Batch
    avoids queue latency and per-shard polling overhead while preserving the
    original shard attempts as immutable history.
    """
    state = load_state(run_dir)
    if (run_dir / "approval.json").exists():
        raise PipelineError("run is already approved")
    for shard in state["shards"]:
        attempts = [attempt for attempt in shard["attempts"] if not attempt.get("superseded")]
        if not attempts:
            continue
        latest = attempts[-1]
        if latest.get("status") in ACTIVE_STATUSES:
            raise PipelineError(f"Batch {latest.get('batch_id')} is still {latest['status']}")
    fresh_round = int(state.get("fresh_replacement_rounds", 0)) + 1
    max_fresh_rounds = int(state.get("fresh_replacement_max_rounds", 3))
    if fresh_round > max_fresh_rounds:
        raise PipelineError(
            f"fresh replacement limit of {max_fresh_rounds} rounds is exhausted"
        )
    plan = read_plan(run_dir, state)
    prompt_overrides = state.get("retry_prompt_overrides", {})
    qa = _read_auto_qa(run_dir)
    failures = {
        custom_id: [
            reason for reason in row.get("quality_gate_reasons", [])
            if reason in QUALITY_CORRECTIONS
        ]
        for custom_id, row in qa.items()
        if not row.get("quality_gate_pass")
    }
    failures = {custom_id: reasons for custom_id, reasons in failures.items() if reasons}
    for custom_id, reasons in _human_quality_failures(run_dir).items():
        failures.setdefault(custom_id, []).extend(reasons)
    failures = {
        custom_id: list(dict.fromkeys(reasons))
        for custom_id, reasons in failures.items()
        if custom_id in plan and (run_dir / "images" / plan[custom_id]["filename"]).exists()
    }
    if not failures:
        return {"retry_requests": 0, "fresh_round": fresh_round - 1, "reasons": {}}
    retry_ids = [custom_id for custom_id in plan if custom_id in failures]
    shard_index = max((shard["index"] for shard in state["shards"]), default=-1) + 1
    shard = {
        "index": shard_index,
        "custom_ids": retry_ids,
        "attempts": [],
        "repair_only": True,
        "aggregate_correction": True,
    }
    state["shards"].append(shard)
    input_name = f"batch_input_{shard_index:03d}_attempt_00.jsonl"
    input_path = run_dir / input_name
    requests = [
        _quality_retry_request(
            plan[custom_id],
            failures[custom_id],
            round_number=fresh_round,
            extra_instruction=prompt_overrides.get(custom_id, ""),
            api_profile=state.get("api_request"),
        )
        for custom_id in retry_ids
    ]
    _write_jsonl(input_path, requests)
    validate_jsonl(input_path, state.get("api_request"))
    attempt = {
        "number": 0,
        "kind": "fresh_replacement",
        "fresh_round": fresh_round,
        "input_path": input_name,
        "input_sha256": sha256_file(input_path),
        "custom_ids": retry_ids,
        "qa_reasons": {custom_id: failures[custom_id] for custom_id in retry_ids},
        "archive_dir": f"rejected/shard_{shard_index:03d}_attempt_00",
        "archive_complete": False,
        "input_file_id": None,
        "batch_id": None,
        "status": "planned",
        "output_file_id": None,
        "error_file_id": None,
        "request_counts": None,
        "history": [{"at": utc_now(), "status": "planned_fresh_replacement"}],
    }
    shard["attempts"].append(attempt)
    save_state(run_dir, state)
    _finish_quality_archive(run_dir, state, attempt, plan)
    created = attempt["qa_reasons"]
    state["fresh_replacement_rounds"] = fresh_round
    state.setdefault("fresh_replacement_history", []).append({
        "at": utc_now(), "round": fresh_round, "requests": len(created)
    })
    save_state(run_dir, state)
    return {"retry_requests": len(created), "fresh_round": fresh_round, "reasons": created}


def _finish_quality_archive(run_dir: Path, state: dict[str, Any],
                            attempt: dict[str, Any],
                            plan: dict[str, dict[str, Any]]) -> None:
    archive_dir = run_dir / attempt["archive_dir"]
    archive_dir.mkdir(parents=True, exist_ok=True)
    for custom_id in attempt["custom_ids"]:
        filename = plan[custom_id]["filename"]
        source = run_dir / "images" / filename
        destination = archive_dir / filename
        if source.exists() and destination.exists():
            raise PipelineError(f"both accepted and rejected copies exist for {custom_id}")
        if source.exists():
            shutil.move(str(source), str(destination))
        if not destination.exists():
            raise PipelineError(f"cannot finish quality archive for {custom_id}")
        state["items"][custom_id].update({
            "status": "quality_retry_planned",
            "rejected_sha256": sha256_file(destination),
            "rejected_path": str(destination.relative_to(run_dir)),
            "qa_reasons": attempt["qa_reasons"][custom_id],
        })
        save_state(run_dir, state)
    attempt["archive_complete"] = True
    save_state(run_dir, state)


def _image_edit_retry_request(record: dict[str, Any], reasons: list[str],
                              source_path: Path, *, round_number: int,
                              extra_instruction: str,
                              api_profile: dict[str, Any] | None) -> dict[str, Any]:
    """Build a Batch-compatible image edit with an inline source JPEG.

    Batch input is JSON rather than multipart, so the edit endpoint's JSON
    ``images`` array carries one ``image_url`` base64 data URI.  Keeping the
    source bytes inside the immutable JSONL also makes API-resume retries
    byte-for-byte reproducible.
    """
    edit_directive = (
        "EDIT THE SUPPLIED SOURCE PHOTOGRAPH; do not create an unrelated person or scene. "
        "Preserve the same fictional adult, exact head pitch and yaw, facial expression, "
        "clothing, lighting, and natural background. Change only camera framing and canvas "
        "composition as required by the correction. Outpaint coherent background, furniture, "
        "shoulders, or upper torso where additional margin is needed. Do not place the source "
        "inside a border, inset, collage, or picture frame."
    )
    combined_override = " ".join(
        value for value in [edit_directive, extra_instruction.strip()] if value
    )
    request = _quality_retry_request(
        record,
        reasons,
        round_number=round_number,
        extra_instruction=combined_override,
        api_profile=api_profile,
    )
    request["url"] = EDIT_ENDPOINT
    request["body"]["images"] = [{
        "image_url": (
            "data:image/jpeg;base64,"
            + base64.b64encode(source_path.read_bytes()).decode("ascii")
        )
    }]
    validate_batch_request(request, api_profile, EDIT_ENDPOINT)
    return request


def prepare_image_edit_retry(run_dir: Path) -> dict[str, Any]:
    """Aggregate geometry-only QA failures into one source-image edit Batch.

    Each distinct source image receives at most one edit attempt by default.
    If an edit still misses QA, callers fall back to a fresh text generation;
    that newly generated source may itself receive one bounded edit.  Counting
    by source SHA-256 avoids repeatedly editing the same bad composition while
    still allowing efficient local correction after a fresh replacement.
    """
    state = load_state(run_dir)
    if (run_dir / "approval.json").exists():
        raise PipelineError("run is already approved")
    for shard in state["shards"]:
        attempts = [
            attempt for attempt in shard.get("attempts", [])
            if not attempt.get("superseded")
        ]
        if attempts and attempts[-1].get("status") in ACTIVE_STATUSES:
            raise PipelineError(
                f"Batch {attempts[-1].get('batch_id')} is still {attempts[-1]['status']}"
            )

    plan = read_plan(run_dir, state)
    pending = [
        attempt for shard in state["shards"] for attempt in shard.get("attempts", [])
        if attempt.get("kind") == "image_edit"
        and attempt.get("status") == "planned"
        and not attempt.get("batch_id")
        and not attempt.get("superseded")
    ]
    if pending:
        if len(pending) != 1:
            raise PipelineError("multiple unsent image-edit Batches exist")
        attempt = pending[0]
        _finish_quality_archive(run_dir, state, attempt, plan)
        return {
            "retry_requests": len(attempt["custom_ids"]),
            "reasons": attempt.get("qa_reasons", {}),
            "exhausted": [],
            "skipped_non_geometry": [],
            "reused_pending": True,
        }

    qa = _read_auto_qa(run_dir)
    failures: dict[str, list[str]] = {}
    for custom_id, row in qa.items():
        reasons = [
            reason for reason in row.get("quality_gate_reasons", [])
            if reason in QUALITY_CORRECTIONS
        ]
        if reasons:
            failures[custom_id] = reasons
    for custom_id, reasons in _human_quality_failures(run_dir).items():
        failures.setdefault(custom_id, []).extend(reasons)
    failures = {
        custom_id: list(dict.fromkeys(reasons))
        for custom_id, reasons in failures.items()
        if custom_id in plan
        and (run_dir / "images" / plan[custom_id]["filename"]).exists()
    }
    geometry = {
        custom_id: reasons for custom_id, reasons in failures.items()
        if reasons and set(reasons) <= GEOMETRY_EDIT_REASONS
    }
    skipped_non_geometry = [
        custom_id for custom_id in plan
        if custom_id in failures and custom_id not in geometry
    ]
    max_edits = int(state.get("image_edit_max_retries_per_source", 1))
    retry_ids: list[str] = []
    exhausted: list[str] = []
    current_source_sha256: dict[str, str] = {}
    for custom_id in plan:
        if custom_id not in geometry:
            continue
        source = run_dir / "images" / plan[custom_id]["filename"]
        source_sha = sha256_file(source)
        current_source_sha256[custom_id] = source_sha
        prior = sum(
            attempt.get("kind") == "image_edit"
            and custom_id in attempt.get("custom_ids", [])
            and attempt.get("source_sha256", {}).get(custom_id) == source_sha
            for shard in state["shards"]
            for attempt in shard.get("attempts", [])
        )
        if prior >= max_edits:
            exhausted.append(custom_id)
        else:
            retry_ids.append(custom_id)
    if not retry_ids:
        return {
            "retry_requests": 0,
            "reasons": {},
            "exhausted": exhausted,
            "skipped_non_geometry": skipped_non_geometry,
            "reused_pending": False,
        }

    shard_index = max((int(shard["index"]) for shard in state["shards"]), default=-1) + 1
    input_name = f"batch_input_{shard_index:03d}_attempt_00.jsonl"
    input_path = run_dir / input_name
    overrides = state.get("retry_prompt_overrides", {})
    requests: list[dict[str, Any]] = []
    source_sha256: dict[str, str] = {}
    for custom_id in retry_ids:
        record = plan[custom_id]
        source = run_dir / "images" / record["filename"]
        if not _valid_existing_image(source, record["size"]):
            raise PipelineError(f"image-edit source is invalid: {source}")
        source_sha = current_source_sha256[custom_id]
        prior_for_source = sum(
            attempt.get("kind") == "image_edit"
            and custom_id in attempt.get("custom_ids", [])
            and attempt.get("source_sha256", {}).get(custom_id) == source_sha
            for shard in state["shards"]
            for attempt in shard.get("attempts", [])
        )
        total_prior = sum(
            attempt.get("kind") == "image_edit"
            and custom_id in attempt.get("custom_ids", [])
            for shard in state["shards"]
            for attempt in shard.get("attempts", [])
        )
        if prior_for_source >= max_edits:
            raise PipelineError(f"image-edit retry limit reached for source: {custom_id}")
        source_sha256[custom_id] = source_sha
        requests.append(_image_edit_retry_request(
            record,
            geometry[custom_id],
            source,
            round_number=total_prior + 1,
            extra_instruction=overrides.get(custom_id, ""),
            api_profile=state.get("api_request"),
        ))
    _write_jsonl(input_path, requests)
    validate_jsonl(input_path, state.get("api_request"), EDIT_ENDPOINT)
    attempt = {
        "number": 0,
        "kind": "image_edit",
        "endpoint": EDIT_ENDPOINT,
        "replace_existing": True,
        "input_path": input_name,
        "input_sha256": sha256_file(input_path),
        "custom_ids": retry_ids,
        "qa_reasons": {custom_id: geometry[custom_id] for custom_id in retry_ids},
        "source_sha256": source_sha256,
        "archive_dir": f"rejected/shard_{shard_index:03d}_attempt_00",
        "archive_complete": False,
        "input_file_id": None,
        "batch_id": None,
        "status": "planned",
        "output_file_id": None,
        "error_file_id": None,
        "request_counts": None,
        "history": [{"at": utc_now(), "status": "planned_image_edit"}],
    }
    shard = {
        "index": shard_index,
        "custom_ids": retry_ids,
        "attempts": [attempt],
        "repair_only": True,
        "aggregate_correction": True,
        "image_edit": True,
    }
    state["shards"].append(shard)
    save_state(run_dir, state)
    _finish_quality_archive(run_dir, state, attempt, plan)
    state.setdefault("image_edit_history", []).append({
        "at": utc_now(),
        "shard": shard_index,
        "requests": len(retry_ids),
        "source_sha256": source_sha256,
    })
    save_state(run_dir, state)
    return {
        "retry_requests": len(retry_ids),
        "reasons": attempt["qa_reasons"],
        "exhausted": exhausted,
        "skipped_non_geometry": skipped_non_geometry,
        "reused_pending": False,
    }


def apply_local_crop_repair(run_dir: Path, target_head_height: float = 0.30) -> dict[str, Any]:
    """Deterministically crop head-too-small images without another API request.

    This repair is deliberately narrow: the current QA reason must be exactly
    ``head_too_small``, the detector head box must be present, and the computed
    crop must preserve at least half a head width on both sides.  The original
    JPEG and its SHA-256 are retained for audit, while the accepted derivative
    is re-encoded at the run's fixed JPEG q92 setting and original dimensions.
    """
    if not 0.25 < target_head_height <= 0.35:
        raise PipelineError("local crop target head height must be in (0.25, 0.35]")
    state = load_state(run_dir)
    if (run_dir / "approval.json").exists():
        raise PipelineError("run is already approved")
    for shard in state["shards"]:
        attempts = [attempt for attempt in shard.get("attempts", []) if not attempt.get("superseded")]
        if attempts and attempts[-1].get("status") in ACTIVE_STATUSES:
            raise PipelineError(
                f"Batch {attempts[-1].get('batch_id')} is still {attempts[-1]['status']}"
            )

    plan = read_plan(run_dir, state)
    qa = _read_auto_qa(run_dir)
    prior_hashes: dict[str, set[str]] = {}
    for event in state.get("local_crop_history", []):
        for entry in event.get("entries", []):
            custom_id = str(entry.get("custom_id", ""))
            hashes = prior_hashes.setdefault(custom_id, set())
            for key in ("source_sha256", "output_sha256"):
                if entry.get(key):
                    hashes.add(str(entry[key]))
    candidates = [
        custom_id for custom_id in plan
        if qa.get(custom_id, {}).get("quality_gate_reasons") == ["head_too_small"]
        and (source := run_dir / "images" / plan[custom_id]["filename"]).exists()
        and sha256_file(source) not in prior_hashes.get(custom_id, set())
    ]
    if not candidates:
        return {"repaired": 0, "custom_ids": [], "skipped": []}

    crop_round = len(state.get("local_crop_history", [])) + 1
    archive_dir = run_dir / "rejected" / f"local_crop_round_{crop_round:02d}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    repaired: list[str] = []
    skipped: list[dict[str, str]] = []
    entries: list[dict[str, Any]] = []
    for custom_id in candidates:
        record = plan[custom_id]
        row = qa[custom_id]
        box = row.get("head_box_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            skipped.append({"custom_id": custom_id, "reason": "head_box_missing"})
            continue
        width, height = (int(value) for value in record["size"].split("x"))
        x1, y1, x2, y2 = (float(value) for value in box)
        head_width, head_height = x2 - x1, y2 - y1
        if head_width <= 0 or head_height <= 0:
            skipped.append({"custom_id": custom_id, "reason": "head_box_invalid"})
            continue

        # A centered crop needs at least 2x head width to provide half a head
        # of background on each side.  Keep 4% safety for detector jitter.
        aspect = width / height
        side_safe_target = head_height * aspect / (2.08 * head_width)
        effective_target = min(target_head_height, side_safe_target)
        if effective_target < 0.255:
            skipped.append({"custom_id": custom_id, "reason": "crop_cannot_preserve_margin"})
            continue
        crop_height = min(float(height), head_height / effective_target)
        crop_width = crop_height * aspect
        if crop_width > width:
            crop_width = float(width)
            crop_height = crop_width / aspect
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        left = min(max(center_x - crop_width / 2, 0.0), width - crop_width)
        top = min(max(center_y - crop_height / 2, 0.0), height - crop_height)
        right, bottom = left + crop_width, top + crop_height
        if min(x1 - left, right - x2) < 0.50 * head_width:
            skipped.append({"custom_id": custom_id, "reason": "crop_margin_check_failed"})
            continue
        if min(y1 - top, bottom - y2) < 0.50 * head_height:
            skipped.append({"custom_id": custom_id, "reason": "crop_vertical_margin_failed"})
            continue

        source = run_dir / "images" / record["filename"]
        destination = archive_dir / record["filename"]
        temp = source.with_suffix(source.suffix + ".crop.tmp")
        source_sha = sha256_file(source)
        with Image.open(source) as image:
            derivative = image.convert("RGB").crop((left, top, right, bottom)).resize(
                (width, height), Image.Resampling.LANCZOS
            )
            derivative.save(temp, "JPEG", quality=JPEG_QUALITY, optimize=True)
        if not _valid_existing_image(temp, record["size"]):
            temp.unlink(missing_ok=True)
            raise PipelineError(f"local crop derivative is invalid: {custom_id}")
        if destination.exists():
            temp.unlink(missing_ok=True)
            raise PipelineError(f"local crop archive already exists: {destination}")
        shutil.move(str(source), str(destination))
        temp.replace(source)
        output_sha = sha256_file(source)
        repaired.append(custom_id)
        entries.append({
            "custom_id": custom_id,
            "source_sha256": source_sha,
            "output_sha256": output_sha,
            "source_path": str(destination.relative_to(run_dir)),
            "crop_xyxy": [round(value, 2) for value in (left, top, right, bottom)],
            "target_head_height": round(effective_target, 4),
        })
        state["items"][custom_id]["status"] = "success"

    if repaired:
        state.setdefault("local_crop_history", []).append({
            "at": utc_now(),
            "round": crop_round,
            "custom_ids": repaired,
            "archive_dir": str(archive_dir.relative_to(run_dir)),
            "entries": entries,
        })
        _write_hash_manifest(run_dir, state)
        save_state(run_dir, state)
    return {"repaired": len(repaired), "custom_ids": repaired, "skipped": skipped}


def restore_failed_image_edit_sources(run_dir: Path,
                                      custom_ids: list[str]) -> dict[str, Any]:
    """Restore archived source JPEGs after a terminal image-edit rejection."""
    state = load_state(run_dir)
    if (run_dir / "approval.json").exists():
        raise PipelineError("run is already approved")
    for shard in state["shards"]:
        attempts = [attempt for attempt in shard.get("attempts", []) if not attempt.get("superseded")]
        if attempts and attempts[-1].get("status") in ACTIVE_STATUSES:
            raise PipelineError(
                f"Batch {attempts[-1].get('batch_id')} is still {attempts[-1]['status']}"
            )
    plan = read_plan(run_dir, state)
    requested = list(dict.fromkeys(custom_ids))
    unknown = [custom_id for custom_id in requested if custom_id not in plan]
    if unknown:
        raise PipelineError(f"unknown custom_id(s): {', '.join(unknown)}")
    restored: list[str] = []
    entries: list[dict[str, Any]] = []
    for custom_id in requested:
        destination = run_dir / "images" / plan[custom_id]["filename"]
        if destination.exists():
            if not _valid_existing_image(destination, plan[custom_id]["size"]):
                raise PipelineError(f"existing restore destination is invalid: {destination}")
            continue
        origins = [
            attempt for shard in state["shards"] for attempt in shard.get("attempts", [])
            if attempt.get("kind") == "image_edit"
            and custom_id in attempt.get("custom_ids", [])
            and attempt.get("archive_complete")
            and attempt.get("archive_dir")
        ]
        if not origins:
            raise PipelineError(f"no archived image-edit source exists for {custom_id}")
        origin = origins[-1]
        source = run_dir / origin["archive_dir"] / plan[custom_id]["filename"]
        if not _valid_existing_image(source, plan[custom_id]["size"]):
            raise PipelineError(f"archived image-edit source is invalid: {source}")
        shutil.copy2(source, destination)
        restored_sha = sha256_file(destination)
        expected_sha = origin.get("source_sha256", {}).get(custom_id)
        if expected_sha and restored_sha != expected_sha:
            destination.unlink(missing_ok=True)
            raise PipelineError(f"restored image-edit source SHA mismatch: {custom_id}")
        origin.setdefault("restored_custom_ids", []).append(custom_id)
        state["items"][custom_id]["status"] = "success"
        restored.append(custom_id)
        entries.append({
            "custom_id": custom_id,
            "source_path": str(source.relative_to(run_dir)),
            "restored_sha256": restored_sha,
        })
    if restored:
        state.setdefault("image_edit_restore_history", []).append({
            "at": utc_now(),
            "custom_ids": restored,
            "entries": entries,
        })
        save_state(run_dir, state)
    return {"restored": len(restored), "custom_ids": restored}


def prepare_quality_retry(run_dir: Path) -> dict[str, Any]:
    """Plan a bounded retry Batch for machine- or human-QA failures.

    Rejected images are retained under ``rejected/``.  This function performs no
    network call; callers explicitly use ``submit_pending`` after state is durable.
    """
    state = load_state(run_dir)
    policy = {**DEFAULT_AUTO_CORRECTION, **state.get("auto_correction", {})}
    if state["stage"] not in policy.get("enabled_stages", []):
        raise PipelineError(f"automatic correction is disabled for {state['stage']}")
    if (run_dir / "approval.json").exists():
        raise PipelineError("this run is already approved; refusing to change approved images")
    for shard in state["shards"]:
        attempts = [attempt for attempt in shard["attempts"] if not attempt.get("superseded")]
        if not attempts:
            continue
        latest = attempts[-1]
        if latest.get("status") in ACTIVE_STATUSES:
            raise PipelineError(f"Batch {latest.get('batch_id')} is still {latest['status']}")
    plan = read_plan(run_dir, state)
    prompt_overrides = state.get("retry_prompt_overrides", {})

    # Crash-safe reuse: never archive a second time or create another unsent retry.
    pending = [
        attempt for shard in state["shards"] for attempt in shard["attempts"]
        if attempt.get("kind") == "quality_retry"
        and attempt.get("status") == "planned" and not attempt.get("batch_id")
        and not attempt.get("superseded")
    ]
    if pending:
        for attempt in pending:
            _finish_quality_archive(run_dir, state, attempt, plan)
        reasons = {
            custom_id: values
            for attempt in pending
            for custom_id, values in attempt.get("qa_reasons", {}).items()
        }
        return {
            "retry_requests": len(reasons), "exhausted": [],
            "incomplete_machine_qa": [], "reasons": reasons,
        }

    qa = _read_auto_qa(run_dir)
    if any("quality_gate_complete" not in row for row in qa.values()):
        from ..qa.gpt_head_review import prepare_review_artifacts
        prepare_review_artifacts(run_dir, use_detector=True)
        qa = _read_auto_qa(run_dir)
    failures: dict[str, list[str]] = {}
    incomplete = [
        custom_id for custom_id, row in qa.items()
        if not row.get("quality_gate_complete")
    ]
    for custom_id, row in qa.items():
        reasons = [
            reason for reason in row.get("quality_gate_reasons", [])
            if reason in QUALITY_CORRECTIONS
        ]
        if reasons:
            failures[custom_id] = reasons
    for custom_id, reasons in _human_quality_failures(run_dir).items():
        failures.setdefault(custom_id, []).extend(reasons)
    failures = {
        custom_id: list(dict.fromkeys(reasons))
        for custom_id, reasons in failures.items()
        if custom_id in plan and (run_dir / "images" / plan[custom_id]["filename"]).exists()
    }

    # Initial Production shards exclude the 500 approved Pilot images.  If a
    # reused image fails the final 5,000-image QA, place it in a repair-only
    # shard so it is corrected just like a newly generated image.
    _add_repair_only_shards(state, list(failures))

    max_retries = int(policy.get("max_quality_retries", 0))
    exhausted: list[str] = []
    created_reasons: dict[str, list[str]] = {}
    for shard in state["shards"]:
        if shard.get("aggregate_correction") or shard.get("retired"):
            continue
        retry_ids: list[str] = []
        for custom_id in shard["custom_ids"]:
            if custom_id not in failures:
                continue
            prior = sum(
                attempt.get("kind") == "quality_retry" and custom_id in attempt.get("custom_ids", [])
                for attempt in shard["attempts"]
            )
            if prior >= max_retries:
                exhausted.append(custom_id)
                state["items"][custom_id]["status"] = "quality_retry_exhausted"
            else:
                retry_ids.append(custom_id)
        if not retry_ids:
            continue
        number = len(shard["attempts"])
        input_name = f"batch_input_{shard['index']:03d}_attempt_{number:02d}.jsonl"
        input_path = run_dir / input_name
        requests: list[dict[str, Any]] = []
        for custom_id in retry_ids:
            prior = sum(
                attempt.get("kind") == "quality_retry"
                and custom_id in attempt.get("custom_ids", [])
                for attempt in shard["attempts"]
            )
            latest_request = _latest_request_for_id(
                run_dir, state, shard, custom_id, plan[custom_id]
            )
            requests.append(_quality_retry_request(
                plan[custom_id],
                failures[custom_id],
                base_request=latest_request,
                round_number=prior + 1,
                extra_instruction=prompt_overrides.get(custom_id, ""),
                api_profile=state.get("api_request"),
            ))
        _write_jsonl(input_path, requests)
        validate_jsonl(input_path, state.get("api_request"))
        attempt = {
            "number": number,
            "kind": "quality_retry",
            "input_path": input_name,
            "input_sha256": sha256_file(input_path),
            "custom_ids": retry_ids,
            "qa_reasons": {custom_id: failures[custom_id] for custom_id in retry_ids},
            "archive_dir": f"rejected/shard_{shard['index']:03d}_attempt_{number:02d}",
            "archive_complete": False,
            "input_file_id": None,
            "batch_id": None,
            "status": "planned",
            "output_file_id": None,
            "error_file_id": None,
            "request_counts": None,
            "history": [{"at": utc_now(), "status": "planned_quality_retry"}],
        }
        shard["attempts"].append(attempt)
        save_state(run_dir, state)
        _finish_quality_archive(run_dir, state, attempt, plan)
        for custom_id in retry_ids:
            created_reasons[custom_id] = failures[custom_id]
    state["quality_gate"] = {
        "evaluated_at": utc_now(),
        "retry_requests": len(created_reasons),
        "exhausted": exhausted,
        "incomplete_machine_qa": incomplete,
        "reasons": created_reasons,
    }
    save_state(run_dir, state)
    return {
        "retry_requests": len(created_reasons),
        "exhausted": exhausted,
        "incomplete_machine_qa": incomplete,
        "reasons": created_reasons,
    }


def cancel_batch(run_dir: Path, batch_id: str, client: Any | None = None) -> str:
    client = client or _client()
    state = load_state(run_dir)
    found: dict[str, Any] | None = None
    for shard in state["shards"]:
        for attempt in shard["attempts"]:
            if attempt.get("batch_id") == batch_id:
                found = attempt
                break
    if found is None:
        raise PipelineError(f"explicit Batch ID is not recorded in this run: {batch_id}")
    current = client.batches.retrieve(batch_id)
    _sync_attempt_from_batch(found, current)
    if found["status"] in TERMINAL_STATUSES:
        raise PipelineError(f"Batch {batch_id} is already {found['status']}")
    cancelled = client.batches.cancel(batch_id)
    _sync_attempt_from_batch(found, cancelled)
    save_state(run_dir, state)
    return found["status"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI Batch API based synthetic head-image generation pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="create a deterministic generation plan and Batch JSONL")
    plan.add_argument(
        "--stage", choices=["validation", "pilot", "production", "lookup"], required=True
    )
    plan.add_argument("--batch-id", required=True, help="local immutable run identifier")
    plan.add_argument("--config", type=Path, default=Path("configs/head_image_generation.yaml"))
    plan.add_argument("--output-root", type=Path, default=Path("data/imagegen/hrffa-heads"))
    plan.add_argument("--seed", type=int, default=20260822)
    plan.add_argument("--approved-batch-dir", type=Path)
    for name in [
        "submit", "status", "collect", "resume", "repair", "compact", "prune-outputs",
        "archive-inputs",
    ]:
        command = sub.add_parser(name)
        command.add_argument("--batch-dir", type=Path, required=True)
        if name == "collect":
            command.add_argument("--skip-detector", action="store_true")
            command.add_argument("--auto-repair", action="store_true",
                                 help="submit a bounded Batch for machine-QA failures")
        elif name == "status":
            command.add_argument("--watch", action="store_true",
                                 help="poll until terminal and collect immediately")
            command.add_argument("--interval-seconds", type=int, default=60)
            command.add_argument("--skip-detector", action="store_true")
            command.add_argument("--auto-resume", action="store_true",
                                 help="re-submit only missing requests after terminal API failures")
            command.add_argument("--auto-repair", action="store_true",
                                 help="while watching, repair machine-QA failures until pass/limit")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("--batch-dir", type=Path, required=True)
    cancel.add_argument("--batch-id", required=True, help="explicit OpenAI Batch ID")
    extend = sub.add_parser("extend-repair")
    extend.add_argument("--batch-dir", type=Path, required=True)
    extend.add_argument("--max-retries", type=int, required=True)
    extend_fresh = sub.add_parser("extend-fresh-replacements")
    extend_fresh.add_argument("--batch-dir", type=Path, required=True)
    extend_fresh.add_argument("--max-rounds", type=int, required=True)
    reprocess = sub.add_parser("reprocess-output")
    reprocess.add_argument("--batch-dir", type=Path, required=True)
    reprocess.add_argument("--shard-index", type=int, required=True)
    reprocess.add_argument("--attempt-number", type=int, required=True)
    fresh = sub.add_parser("replace-stubborn")
    fresh.add_argument("--batch-dir", type=Path, required=True)
    edit_repair = sub.add_parser(
        "edit-repair",
        help="repair geometry-only QA failures using source images in one edit Batch",
    )
    edit_repair.add_argument("--batch-dir", type=Path, required=True)
    crop_repair = sub.add_parser(
        "crop-repair",
        help="locally crop head-too-small images using detector boxes without an API call",
    )
    crop_repair.add_argument("--batch-dir", type=Path, required=True)
    restore_edit = sub.add_parser(
        "restore-edit-source",
        help="restore an archived source image after a terminal edit rejection",
    )
    restore_edit.add_argument("--batch-dir", type=Path, required=True)
    restore_edit.add_argument("--custom-id", action="append", required=True)
    split_limit = sub.add_parser(
        "split-token-limit",
        help="replace a zero-progress token-limited shard with sequential sub-shards",
    )
    split_limit.add_argument("--batch-dir", type=Path, required=True)
    split_limit.add_argument("--max-requests", type=int, default=500)
    revise = sub.add_parser(
        "revise-held-prompts",
        help="revise only pristine unsent requests behind an early-drift gate",
    )
    revise.add_argument("--batch-dir", type=Path, required=True)
    revise.add_argument("--config", type=Path, required=True)
    revise_unsent = sub.add_parser(
        "revise-unsent-prompts",
        help="append a rolling-QA correction only to pristine unsent generations",
    )
    revise_unsent.add_argument("--batch-dir", type=Path, required=True)
    revise_unsent.add_argument("--instruction", required=True)
    rollback_unsent = sub.add_parser(
        "rollback-unsent-prompts",
        help="remove the latest harmful rolling-QA suffix from pristine unsent requests",
    )
    rollback_unsent.add_argument("--batch-dir", type=Path, required=True)
    override = sub.add_parser("set-retry-override")
    override.add_argument("--batch-dir", type=Path, required=True)
    override.add_argument("--custom-id", action="append", required=True)
    override.add_argument("--instruction", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            run_dir = create_plan(args.config, args.stage, args.batch_id, args.output_root,
                                  args.seed, args.approved_batch_dir)
            state = load_state(run_dir)
            print(json.dumps({
                "batch_dir": str(run_dir),
                "stage": state["stage"],
                "requests": len(state["items"]),
                "shards": len(state["shards"]),
            }, indent=2))
        elif args.command == "submit":
            print(json.dumps({"batch_ids": submit_pending(args.batch_dir)}, indent=2))
        elif args.command == "status":
            if args.watch:
                result = watch_until_terminal(
                    args.batch_dir, args.interval_seconds, detector=not args.skip_detector,
                    auto_repair=args.auto_repair,
                    auto_resume=args.auto_resume,
                )
            else:
                result = refresh_status(args.batch_dir)
            print(json.dumps(result, indent=2))
        elif args.command == "collect":
            result: dict[str, Any] = collect_results(
                args.batch_dir, detector=not args.skip_detector
            )
            if args.auto_repair:
                repair = prepare_quality_retry(args.batch_dir)
                repair["batch_ids"] = submit_pending(args.batch_dir) if repair["retry_requests"] else []
                result["quality_repair"] = repair
            print(json.dumps(result, indent=2))
        elif args.command == "resume":
            # Collect any partial successes before calculating the retry set.
            collect_results(args.batch_dir, prepare_review=False)
            count = prepare_resume(args.batch_dir)
            ids = submit_pending(args.batch_dir) if count else []
            print(json.dumps({"retry_requests": count, "batch_ids": ids}, indent=2))
        elif args.command == "repair":
            repair = prepare_quality_retry(args.batch_dir)
            repair["batch_ids"] = submit_pending(args.batch_dir) if repair["retry_requests"] else []
            print(json.dumps(repair, indent=2))
        elif args.command == "compact":
            print(json.dumps(compact_legacy_png_run(args.batch_dir), indent=2))
        elif args.command == "prune-outputs":
            print(json.dumps(prune_batch_outputs(args.batch_dir), indent=2))
        elif args.command == "archive-inputs":
            print(json.dumps(archive_batch_inputs(args.batch_dir), indent=2))
        elif args.command == "extend-repair":
            extension = extend_quality_retry_limit(args.batch_dir, args.max_retries)
            repair = prepare_quality_retry(args.batch_dir)
            repair["batch_ids"] = submit_pending(args.batch_dir) if repair["retry_requests"] else []
            print(json.dumps({"extension": extension, "quality_repair": repair}, indent=2))
        elif args.command == "extend-fresh-replacements":
            print(json.dumps(extend_fresh_replacement_limit(
                args.batch_dir, args.max_rounds
            ), indent=2))
        elif args.command == "reprocess-output":
            print(json.dumps(reprocess_local_output(
                args.batch_dir, args.shard_index, args.attempt_number
            ), indent=2))
        elif args.command == "replace-stubborn":
            replacement = prepare_fresh_replacement(args.batch_dir)
            replacement["batch_ids"] = (
                submit_pending(args.batch_dir) if replacement["retry_requests"] else []
            )
            print(json.dumps(replacement, indent=2))
        elif args.command == "edit-repair":
            edit = prepare_image_edit_retry(args.batch_dir)
            edit["batch_ids"] = (
                submit_pending(args.batch_dir) if edit["retry_requests"] else []
            )
            print(json.dumps(edit, indent=2))
        elif args.command == "crop-repair":
            print(json.dumps(apply_local_crop_repair(args.batch_dir), indent=2))
        elif args.command == "restore-edit-source":
            print(json.dumps(restore_failed_image_edit_sources(
                args.batch_dir, args.custom_id
            ), indent=2))
        elif args.command == "split-token-limit":
            print(json.dumps(split_token_limited_shard(
                args.batch_dir, args.max_requests
            ), indent=2))
        elif args.command == "revise-held-prompts":
            print(json.dumps(revise_held_prompts(
                args.batch_dir, args.config
            ), ensure_ascii=False, indent=2))
        elif args.command == "revise-unsent-prompts":
            print(json.dumps(revise_unsent_prompts(
                args.batch_dir, args.instruction
            ), ensure_ascii=False, indent=2))
        elif args.command == "rollback-unsent-prompts":
            print(json.dumps(rollback_latest_unsent_prompt_revision(
                args.batch_dir
            ), ensure_ascii=False, indent=2))
        elif args.command == "set-retry-override":
            print(json.dumps(set_retry_prompt_override(
                args.batch_dir, args.custom_id, args.instruction
            ), indent=2))
        elif args.command == "cancel":
            print(json.dumps({"batch_id": args.batch_id,
                              "status": cancel_batch(args.batch_dir, args.batch_id)}, indent=2))
        return 0
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
