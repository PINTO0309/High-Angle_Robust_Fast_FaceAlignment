"""Pose-collapse QA for the upward-only synthetic lookup-data run.

This is deliberately a conservative categorical audit.  6DRepNet360 is used
for frontal-collapse detection and moderate-angle checks, while DEIMv2 facial
features recorded by :mod:`gpt_head_review` back the extreme-pitch decisions.
The estimator is not treated as exact ground truth outside its calibrated
range; the output records the heuristic and its reliability for audit.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

from ..augment.gpt_head_gen import load_state, save_state, sha256_file, utc_now
from .gpt_head_review import DEFAULT_QA_POLICY, VALID_INTENT, quality_gate_reasons
from .sixdrepnet import SixDRepNet360

class LookupPoseQAError(RuntimeError):
    """Raised when the lookup pose audit cannot run safely."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise LookupPoseQAError(f"required file is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise LookupPoseQAError(f"invalid JSONL {path}:{number}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def _intent_grade(record: dict[str, Any], qa: dict[str, Any],
                  pitch_est: float, yaw_est: float) -> tuple[str, str]:
    """Return (grade, basis) without claiming exact extreme-angle regression."""
    bin_id = str(record["bin"])
    abs_pitch = abs(pitch_est)
    abs_yaw = abs(yaw_est)
    facial_feature = bool(qa.get("has_nose") or qa.get("has_mouth"))

    if bin_id == "pitch_up_20_40":
        if 12 <= pitch_est <= 55:
            return "match", "sixd_moderate_pitch"
        if 5 <= pitch_est <= 70:
            return "off-by-one-bin", "sixd_moderate_pitch"
        return "wrong", "sixd_frontal_or_inverted_collapse"

    if bin_id == "pitch_up_40_60":
        if 20 <= pitch_est <= 80:
            return "match", "sixd_strong_pitch"
        if 8 <= pitch_est <= 95:
            return "off-by-one-bin", "sixd_strong_pitch"
        return "wrong", "sixd_frontal_or_inverted_collapse"

    if bin_id == "pitch_up_60_90":
        if abs_pitch >= 20 and facial_feature:
            return "match", "sixd_extreme_collapse_plus_face_parts"
        if abs_pitch >= 10 or facial_feature:
            return "off-by-one-bin", "sixd_extreme_collapse_plus_face_parts"
        return "wrong", "extreme_pitch_collapse"

    if bin_id == "pitch_up_90_120":
        if abs_pitch >= 30 and facial_feature:
            return "match", "sixd_past_vertical_plus_face_parts"
        if abs_pitch >= 15 or facial_feature:
            return "off-by-one-bin", "sixd_past_vertical_plus_face_parts"
        return "wrong", "past_vertical_collapse"

    if bin_id == "camera_low_20_60":
        if 15 <= pitch_est <= 80:
            return "match", "sixd_worms_eye_effect"
        if 5 <= pitch_est <= 95:
            return "off-by-one-bin", "sixd_worms_eye_effect"
        return "wrong", "low_angle_effect_missing"

    if bin_id == "combined_pitch_30_60_yaw_30_90":
        if abs_pitch >= 15 and abs_yaw >= 20:
            return "match", "sixd_combined_categorical"
        if abs_pitch >= 8 or abs_yaw >= 15:
            return "off-by-one-bin", "sixd_combined_categorical"
        return "wrong", "combined_pose_collapse"

    if bin_id == "combined_pitch_60_90_yaw_30_60":
        if abs_pitch >= 20 and abs_yaw >= 18 and facial_feature:
            return "match", "sixd_combined_extreme_plus_face_parts"
        if abs_pitch >= 10 or abs_yaw >= 12 or facial_feature:
            return "off-by-one-bin", "sixd_combined_extreme_plus_face_parts"
        return "wrong", "combined_extreme_pose_collapse"

    raise LookupPoseQAError(f"unsupported lookup pose bin: {bin_id}")


def _roll_is_reliable(record: dict[str, Any]) -> bool:
    # Euler roll is unstable near profiles and past-vertical pitch.  In those
    # bins DEIMv2's back-of-head category remains the fail-closed check.
    # A medical mask can make SixDRepNet flip an otherwise upright worm's-eye
    # head by roughly 180 degrees.  Keep that estimate in the audit output, but
    # do not make it a hard roll/back rejection.
    return (
        str(record["bin"]) == "camera_low_20_60"
        and "medical mask" not in str(record.get("accessories", "")).lower()
    )


def _apply_current_human_pose_review(
    pose_row: dict[str, Any], review: dict[str, str] | None,
    image_sha256: str | None,
) -> dict[str, Any]:
    """Apply a SHA-bound human pose adjudication over a noisy teacher result."""
    if not review or not image_sha256:
        return pose_row
    if review.get("reviewed_sha256") != image_sha256:
        return pose_row
    if review.get("intent_match") not in VALID_INTENT:
        return pose_row
    if review.get("roll_no_back") not in {"pass", "fail"}:
        return pose_row

    result = dict(pose_row)
    result["teacher_intent_match"] = result.get("intent_match")
    result["teacher_basis"] = result.get("basis")
    result["teacher_roll_no_back"] = result.get("roll_no_back")
    result["intent_match"] = review["intent_match"]
    result["basis"] = "human_review_sha256_bound"
    result["intent_hard_failure"] = review["intent_match"] == "wrong"
    result["roll_no_back"] = review["roll_no_back"] == "pass"
    result["roll_hard_failure"] = review["roll_no_back"] == "fail"
    result["human_adjudicated"] = True
    return result


def _completed_pending_pose_ids(
    run_dir: Path, state: dict[str, Any], plan: dict[str, dict[str, Any]],
    auto: dict[str, dict[str, Any]], evaluated_ids: set[str],
) -> set[str]:
    """Return pending IDs whose detector/pose QA is bound to current bytes."""
    completed: set[str] = set()
    for custom_id in set(state.get("pending_qa_ids", [])) & evaluated_ids:
        row = auto.get(custom_id, {})
        record = plan.get(custom_id)
        if not record or not row.get("quality_gate_complete"):
            continue
        image_path = run_dir / "images" / record["filename"]
        if (image_path.exists() and row.get("sha256")
                and sha256_file(image_path) == row["sha256"]):
            completed.add(custom_id)
    return completed


def _intent_is_confident_failure(record: dict[str, Any], qa: dict[str, Any],
                                 grade: str, pitch_est: float,
                                 yaw_est: float) -> bool:
    """Fail only clear collapses; keep noisy extreme estimates as audit evidence."""
    if grade != "wrong":
        return False
    # A medical mask removes most nose/mouth geometry used by the pose teacher.
    # Validation established that masked upward poses remain usable when DEIMv2
    # still detects the head and facial evidence.  Keep SixD estimates for audit,
    # but do not turn this known occlusion weak region into a hard rejection.
    if "medical mask" in str(record.get("accessories", "")).lower():
        return False
    bin_id = str(record["bin"])
    facial_feature = bool(qa.get("has_nose") or qa.get("has_mouth"))
    if bin_id in {"pitch_up_20_40", "pitch_up_40_60", "camera_low_20_60"}:
        return -20 <= pitch_est < 5
    if bin_id in {"pitch_up_60_90", "pitch_up_90_120"}:
        return abs(pitch_est) < 8 and not facial_feature
    return abs(pitch_est) < 8 and abs(yaw_est) < 12 and not facial_feature


def run_lookup_pose_qa(run_dir: Path, custom_ids: set[str] | None = None) -> dict[str, Any]:
    state = load_state(run_dir)
    if state.get("stage") != "lookup":
        raise LookupPoseQAError("lookup pose QA is restricted to stage=lookup")

    plan_rows = _read_jsonl(run_dir / state["plan_path"])
    plan = {row["custom_id"]: row for row in plan_rows}
    auto_rows = _read_jsonl(run_dir / "auto_qa.jsonl")
    auto = {row["custom_id"]: row for row in auto_rows}
    if set(auto) != set(plan):
        raise LookupPoseQAError("auto_qa.jsonl must cover the complete generation plan")

    requested = set(custom_ids or ())
    targets = [
        row for row in plan_rows
        if (not requested or row["custom_id"] in requested)
        and (run_dir / "images" / row["filename"]).exists()
    ]
    if requested - set(plan):
        raise LookupPoseQAError("pose QA requested an unknown custom_id")
    if not targets:
        raise LookupPoseQAError("no collected images are available for pose QA")

    existing_path = run_dir / "pose_qa.jsonl"
    existing = {
        row["custom_id"]: row for row in _read_jsonl(existing_path)
    } if existing_path.exists() else {}
    human_reviews: dict[str, dict[str, str]] = {}
    review_path = run_dir / "human_review.csv"
    if review_path.exists():
        with review_path.open(newline="", encoding="utf-8-sig") as fh:
            human_reviews = {
                row.get("custom_id", ""): row for row in csv.DictReader(fh)
                if row.get("custom_id")
            }
    model = SixDRepNet360()
    evaluated: dict[str, dict[str, Any]] = {}
    policy = {**DEFAULT_QA_POLICY, **state.get("auto_correction", {})}

    for record in targets:
        custom_id = record["custom_id"]
        row = auto[custom_id]
        box = row.get("head_box_xyxy")
        image_path = run_dir / "images" / record["filename"]
        image = cv2.imread(str(image_path))
        if row.get("detector_status") != "ok" or box is None or image is None:
            pose_row = {
                "custom_id": custom_id,
                "filename": record["filename"],
                "bin": record["bin"],
                "intent_match": "wrong",
                "basis": "pose_unavailable",
                "roll_no_back": False,
                "image_sha256": row.get("sha256"),
            }
        else:
            yaw_est, pitch_est, roll_est = model.infer(image, box)
            grade, basis = _intent_grade(record, row, pitch_est, yaw_est)
            roll_reliable = _roll_is_reliable(record)
            roll_no_back = not bool(row.get("back_reference")) and (
                not roll_reliable or abs(roll_est) <= 25
            )
            intent_hard_failure = _intent_is_confident_failure(
                record, row, grade, pitch_est, yaw_est
            )
            roll_hard_failure = roll_reliable and abs(roll_est) > 25
            pose_row = {
                "custom_id": custom_id,
                "filename": record["filename"],
                "bin": record["bin"],
                "pitch_intent": record["pitch"],
                "yaw_intent": record["yaw"],
                "cam_intent": record["cam"],
                "sixd_pitch": round(pitch_est, 2),
                "sixd_yaw": round(yaw_est, 2),
                "sixd_roll": round(roll_est, 2),
                "has_nose": bool(row.get("has_nose")),
                "has_mouth": bool(row.get("has_mouth")),
                "intent_match": grade,
                "basis": basis,
                "intent_hard_failure": intent_hard_failure,
                "roll_reliable": roll_reliable,
                "roll_no_back": roll_no_back,
                "roll_hard_failure": roll_hard_failure,
                "image_sha256": row.get("sha256"),
            }
        pose_row = _apply_current_human_pose_review(
            pose_row, human_reviews.get(custom_id), row.get("sha256")
        )
        evaluated[custom_id] = pose_row
        existing[custom_id] = pose_row

        row["intent_match_auto"] = pose_row["intent_match"]
        row["intent_match_auto_basis"] = pose_row["basis"]
        row["intent_match_auto_hard_failure"] = pose_row.get(
            "intent_hard_failure", True
        )
        row["roll_no_back_auto"] = pose_row["roll_no_back"]
        row["roll_no_back_auto_hard_failure"] = pose_row.get(
            "roll_hard_failure", True
        )
        row["sixd_pitch"] = pose_row.get("sixd_pitch")
        row["sixd_yaw"] = pose_row.get("sixd_yaw")
        row["sixd_roll"] = pose_row.get("sixd_roll")
        reasons, complete = quality_gate_reasons(row, policy)
        # Rebuild the complete hard-gate result from the current policy.  Carrying
        # old detector reasons forward would make a reviewed policy correction
        # (for example allowing a background second person) impossible to apply.
        row["quality_gate_reasons"] = list(dict.fromkeys(reasons))
        row["quality_gate_complete"] = complete
        row["quality_gate_pass"] = complete and not row["quality_gate_reasons"]

    _write_jsonl(run_dir / "auto_qa.jsonl", [auto[row["custom_id"]] for row in plan_rows])
    pose_rows = [existing[row["custom_id"]] for row in plan_rows if row["custom_id"] in existing]
    _write_jsonl(existing_path, pose_rows)
    cleared_pending = _completed_pending_pose_ids(
        run_dir, state, plan, auto, set(evaluated)
    )
    if cleared_pending:
        state["pending_qa_ids"] = sorted(
            set(state.get("pending_qa_ids", [])) - cleared_pending
        )
        state.setdefault("qa_completion_history", []).append({
            "at": utc_now(),
            "kind": "pose_qa_sha256_bound",
            "custom_ids": sorted(cleared_pending),
        })
        save_state(run_dir, state)

    grades = Counter(row["intent_match"] for row in evaluated.values())
    by_bin: dict[str, Counter[str]] = defaultdict(Counter)
    for row in evaluated.values():
        by_bin[row["bin"]][row["intent_match"]] += 1
    summary = {
        "evaluated_at": utc_now(),
        "evaluated": len(evaluated),
        "match": grades["match"],
        "off_by_one_bin": grades["off-by-one-bin"],
        "wrong": grades["wrong"],
        "by_bin": {key: dict(value) for key, value in sorted(by_bin.items())},
        "method": "6DRepNet360 categorical collapse audit plus DEIMv2 facial features",
        "pose_qa": "pose_qa.jsonl",
    }
    summary_path = run_dir / "pose_qa_summary.json"
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(summary_path)
    return summary


def finalize_early_drift_report(run_dir: Path) -> dict[str, Any]:
    state = load_state(run_dir)
    gate = state.get("submission_gate") or {}
    if gate.get("type") != "early_drift":
        raise LookupPoseQAError("this run has no early-drift submission gate")
    early_ids = list(gate.get("custom_ids") or [])
    if len(early_ids) != int(gate.get("count", 0)):
        raise LookupPoseQAError("early-drift gate custom_ids are inconsistent")

    plan = {
        row["custom_id"]: row for row in _read_jsonl(run_dir / state["plan_path"])
    }
    auto = {row["custom_id"]: row for row in _read_jsonl(run_dir / "auto_qa.jsonl")}
    pose = {row["custom_id"]: row for row in _read_jsonl(run_dir / "pose_qa.jsonl")}
    missing = [custom_id for custom_id in early_ids if custom_id not in pose]
    if missing:
        raise LookupPoseQAError(
            f"early pose QA is incomplete ({len(missing)} missing); run pose QA first"
        )

    # Section 9 of the delivery specification explicitly calls for grading the
    # first sample.  Prefer a complete review tied to the exact current image
    # bytes; otherwise retain the automated pose audit as a conservative
    # fallback for unattended tests and interrupted review sessions.
    human_grades: dict[str, str] = {}
    review_path = run_dir / "human_review.csv"
    if review_path.exists():
        with review_path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                custom_id = row.get("custom_id", "")
                if custom_id not in early_ids or row.get("intent_match") not in VALID_INTENT:
                    continue
                record = plan[custom_id]
                image_path = run_dir / "images" / record["filename"]
                if (image_path.exists()
                        and row.get("reviewed_sha256") == sha256_file(image_path)):
                    human_grades[custom_id] = str(row["intent_match"])
    if set(human_grades) == set(early_ids):
        grades_by_id = human_grades
        intent_source = "complete human review bound to current image SHA-256"
    else:
        grades_by_id = {
            custom_id: str(pose[custom_id]["intent_match"])
            for custom_id in early_ids
        }
        intent_source = "automated pose-collapse audit fallback"

    grades = Counter(grades_by_id[custom_id] for custom_id in early_ids)
    by_bin: dict[str, Counter[str]] = defaultdict(Counter)
    for custom_id in early_ids:
        by_bin[plan[custom_id]["bin"]][grades_by_id[custom_id]] += 1
    match_rate = grades["match"] / len(early_ids)
    bin_rates = {
        bin_id: counts["match"] / sum(counts.values())
        for bin_id, counts in sorted(by_bin.items())
    }
    hard_failures = [
        custom_id for custom_id in early_ids
        if not auto.get(custom_id, {}).get("quality_gate_pass")
    ]
    minimum_match = float(gate.get("minimum_match_rate", 0.80))
    minimum_bin = float(gate.get("minimum_bin_match_rate", 0.60))
    passed = (
        match_rate >= minimum_match
        and all(rate >= minimum_bin for rate in bin_rates.values())
        and not hard_failures
    )
    report = {
        "created_at": utc_now(),
        "stage": state["stage"],
        "checked": len(early_ids),
        "grades": {
            "match": grades["match"],
            "off-by-one-bin": grades["off-by-one-bin"],
            "wrong": grades["wrong"],
        },
        "match_rate": round(match_rate, 4),
        "minimum_match_rate": minimum_match,
        "bin_match_rates": {key: round(value, 4) for key, value in bin_rates.items()},
        "minimum_bin_match_rate": minimum_bin,
        "hard_quality_failures": hard_failures,
        "passed": passed,
        "intent_source": intent_source,
        "method": f"{intent_source} plus detector/framing/duplicate hard QA",
    }
    report_path = run_dir / str(gate["report_path"])
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(report_path)
    gate["last_checked_at"] = report["created_at"]
    gate["report_sha256"] = sha256_file(report_path)
    gate["released"] = passed
    state["submission_gate"] = gate
    save_state(run_dir, state)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pose-collapse QA for the upward-only synthetic lookup-data run (GPT batch review).")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--batch-dir", type=Path, required=True)
    run.add_argument("--early-only", action="store_true")
    run.add_argument("--custom-id", action="append")
    report = sub.add_parser("early-report")
    report.add_argument("--batch-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            custom_ids = set(args.custom_id or [])
            if args.early_only:
                state = load_state(args.batch_dir)
                custom_ids.update((state.get("submission_gate") or {}).get("custom_ids") or [])
            result = run_lookup_pose_qa(args.batch_dir, custom_ids or None)
        else:
            result = finalize_early_drift_report(args.batch_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except LookupPoseQAError as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
