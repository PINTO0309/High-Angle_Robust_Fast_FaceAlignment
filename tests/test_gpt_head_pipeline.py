from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import re
import tempfile
import unittest
from copy import deepcopy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from PIL import Image

from hrffa.dataset.augment.gpt_head_gen import (
    EDIT_ENDPOINT,
    PipelineError,
    _quality_retry_request,
    apply_local_crop_repair,
    archive_batch_inputs,
    build_plan,
    collect_results,
    create_plan,
    extend_fresh_replacement_limit,
    load_config,
    load_state,
    prepare_fresh_replacement,
    prepare_image_edit_retry,
    prepare_quality_retry,
    prepare_resume,
    process_output_jsonl,
    prune_batch_outputs,
    rollback_latest_unsent_prompt_revision,
    restore_failed_image_edit_sources,
    revise_held_prompts,
    revise_unsent_prompts,
    set_retry_prompt_override,
    split_token_limited_shard,
    submit_pending,
    validate_batch_request,
    watch_until_terminal,
)
from hrffa.dataset.qa.lookup_pose_qa import (
    _apply_current_human_pose_review,
    _completed_pending_pose_ids,
    _intent_is_confident_failure,
    _roll_is_reliable,
    finalize_early_drift_report,
)
from hrffa.dataset.qa.gpt_head_review import (
    ReviewError,
    approve_review,
    prepare_review_artifacts,
    quality_gate_reasons,
    regenerate_overview_contact_sheet,
    run_auto_qa,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "head_image_generation.yaml"
LOOKUP_CONFIG = ROOT / "configs" / "lookup_traindata_generation.yaml"


def png_b64(size: str, color: tuple[int, int, int]) -> str:
    width, height = (int(value) for value in size.split("x"))
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def jpeg_b64(size: str, color: tuple[int, int, int]) -> str:
    width, height = (int(value) for value in size.split("x"))
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, "JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class DownloadResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def stream_to_file(self, path: Path):
        path.write_bytes(self.payload)


class FakeFiles:
    def __init__(self):
        self.create_calls = 0
        self.content_calls = 0
        self.payloads: dict[str, bytes] = {}
        self.with_streaming_response = self

    def create(self, *, file, purpose: str):
        self.create_calls += 1
        assert purpose == "batch"
        assert file.read(1)
        return SimpleNamespace(id=f"file-input-{self.create_calls}")

    def content(self, file_id: str):
        self.content_calls += 1
        return DownloadResponse(self.payloads[file_id])


class FakeBatches:
    def __init__(self):
        self.create_calls = 0
        self.create_kwargs: list[dict] = []
        self.objects: dict[str, SimpleNamespace] = {}
        self.listed: list[SimpleNamespace] = []

    def list(self, *, limit: int):
        assert limit == 100
        return SimpleNamespace(data=self.listed)

    def create(self, **kwargs):
        self.create_calls += 1
        self.create_kwargs.append(kwargs)
        batch_id = f"batch-{self.create_calls}"
        obj = SimpleNamespace(
            id=batch_id,
            status="validating",
            output_file_id=None,
            error_file_id=None,
            request_counts=SimpleNamespace(completed=0, failed=0, total=10),
            metadata=kwargs["metadata"],
        )
        self.objects[batch_id] = obj
        return obj

    def retrieve(self, batch_id: str):
        return self.objects[batch_id]

    def cancel(self, batch_id: str):
        self.objects[batch_id].status = "cancelling"
        return self.objects[batch_id]


class FakeClient:
    def __init__(self):
        self.files = FakeFiles()
        self.batches = FakeBatches()


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)

    def test_validation_is_fixed_and_strict(self):
        records = build_plan(self.config, "validation", seed=7)
        self.assertEqual(len(records), 10)
        self.assertEqual(records[0]["custom_id"], "pitch-108_yaw+010_cam+00_000001")
        self.assertEqual(records[-1]["custom_id"], "pitch+058_yaw-025_cam+00_000010")
        self.assertEqual(Counter(row["size"] for row in records), {
            "1024x1536": 5, "1024x1024": 3, "1536x1024": 2,
        })
        self.assertTrue(all(row["roll"] == 0 for row in records))
        self.assertTrue(all(row["filename"] == row["custom_id"] + ".jpg" for row in records))
        self.assertTrue(all("fictional adult" in row["prompt"] for row in records))
        self.assertTrue(all("25% to 50% as a hard limit" in row["prompt"] for row in records))
        self.assertTrue(all("Never let the torso or limbs fade" in row["prompt"] for row in records))

    def test_pilot_and_production_quotas_ranges_and_uniqueness(self):
        expected = {
            "pilot": [75, 50, 75, 50, 75, 50, 75, 50],
            "production": [750, 500, 750, 500, 750, 500, 750, 500],
        }
        bin_ids = [item["id"] for item in self.config["bins"]]
        for stage, counts in expected.items():
            records = build_plan(self.config, stage, seed=19)
            histogram = Counter(row["bin"] for row in records)
            self.assertEqual([histogram[bin_id] for bin_id in bin_ids], counts)
            self.assertEqual(len(records), len({row["custom_id"] for row in records}))
            self.assertEqual(Counter(row["size"] for row in records), {
                "1024x1536": len(records) // 2,
                "1024x1024": len(records) * 3 // 10,
                "1536x1024": len(records) // 5,
            })
            for row in records:
                self.assertLessEqual(abs(row["yaw"]), 100)
                self.assertEqual(row["roll"], 0)
                self.assertRegex(row["custom_id"], r"^pitch[+-]\d{3}_yaw[+-]\d{3}_cam[+-]\d{2}_\d{6}$")
                if row["bin"] == "camera_high":
                    self.assertGreater(row["cam"], 0)
                if row["bin"] == "camera_low":
                    self.assertLess(row["cam"], 0)
            for bin_id in bin_ids:
                rows = [row for row in records if row["bin"] == bin_id]
                positive_yaw = sum(row["yaw"] > 0 for row in rows)
                negative_yaw = sum(row["yaw"] < 0 for row in rows)
                self.assertLessEqual(abs(positive_yaw - negative_yaw), 1)
            combined = [row for row in records if row["bin"] == "combined_pitch_yaw"]
            self.assertLessEqual(abs(sum(row["pitch"] > 0 for row in combined) -
                                     sum(row["pitch"] < 0 for row in combined)), 1)

    def test_validation_feedback_is_repeated_in_affected_bin_anchors(self):
        records = build_plan(self.config, "pilot", seed=19)
        by_bin = {bin_id: [row for row in records if row["bin"] == bin_id]
                  for bin_id in {row["bin"] for row in records}}
        for bin_id in ["pitch_up_mid", "pitch_up_extreme", "camera_low"]:
            self.assertTrue(all("head height near 38%" in row["prompt"]
                                for row in by_bin[bin_id]))
        for bin_id in ["pitch_down_mid", "pitch_down_extreme"]:
            self.assertTrue(all("space above the crown and below the chin" in row["prompt"]
                                for row in by_bin[bin_id]))

    def test_large_head_retry_resets_conflicting_accumulated_prompt(self):
        record = build_plan(self.config, "pilot", seed=19)[0]
        old = {
            "body": {
                "prompt": record["prompt"]
                + " QUALITY CORRECTION ROUND 1: Move the camera closer."
            }
        }
        request = _quality_retry_request(
            record,
            ["head_too_large", "insufficient_margin"],
            base_request=old,
            round_number=2,
        )
        prompt = request["body"]["prompt"]
        self.assertEqual(prompt.count("QUALITY CORRECTION ROUND"), 1)
        self.assertIn("significantly farther back", prompt)
        self.assertIn("28% to 32%", prompt)
        self.assertIn("at least 20%", prompt)
        self.assertNotIn("Move the camera closer instead", prompt)

    def test_back_direction_with_visible_face_eye_or_ear_is_not_rejected(self):
        row = {
            "exists": True,
            "image_valid": True,
            "dimension_match": True,
            "duplicate_of": None,
            "detector_status": "ok",
            "head_count": 1,
            "head_height_ratio": 0.35,
            "margin_left_head_ratio": 1.0,
            "margin_right_head_ratio": 1.0,
            "margin_top_head_ratio": 1.0,
            "margin_bottom_head_ratio": 1.0,
            "body_count": 1,
            "back_reference": True,
            "has_face": True,
            "has_eye": False,
            "has_nose": False,
            "has_mouth": False,
            "has_ear": False,
        }
        reasons, complete = quality_gate_reasons(row)
        self.assertTrue(complete)
        self.assertNotIn("back_of_head", reasons)
        row["has_face"] = False
        row["has_ear"] = True
        reasons, _ = quality_gate_reasons(row)
        self.assertNotIn("back_of_head", reasons)
        row["has_ear"] = False
        reasons, _ = quality_gate_reasons(row)
        self.assertIn("back_of_head", reasons)

    def test_production_reuses_pilot_500_and_submits_only_nine_500_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root / "approved-parent"
            parent.mkdir()
            pilot_records = build_plan(self.config, "pilot", seed=23)
            with (parent / "generation_plan.jsonl").open("w", encoding="utf-8") as fh:
                for record in pilot_records:
                    fh.write(json.dumps(record) + "\n")
            images = parent / "images_jpeg_q92"
            images.mkdir()
            templates: dict[str, Path] = {}
            for record in pilot_records:
                size = record["size"]
                if size not in templates:
                    width, height = (int(value) for value in size.split("x"))
                    template = parent / f"template-{size}.jpg"
                    Image.new("RGB", (width, height), (20, 40, 60)).save(
                        template, "JPEG", quality=92
                    )
                    templates[size] = template
                (images / record["filename"]).hardlink_to(templates[size])
            review = parent / "human_review.csv"
            review.write_text("reviewed\n", encoding="utf-8")
            (parent / "approval.json").write_text(json.dumps({
                "approved": True,
                "stage": "pilot",
                "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            production = create_plan(
                CONFIG, "production", "production-shards", root, 23, parent
            )
            state = load_state(production)
            self.assertEqual(len(state["items"]), 5000)
            self.assertEqual(state["target_count"], 5000)
            self.assertEqual(state["reused_count"], 500)
            self.assertEqual(state["request_count"], 4500)
            self.assertEqual(len(state["shards"]), 9)
            self.assertTrue(all(len(shard["custom_ids"]) == 500 for shard in state["shards"]))
            self.assertEqual(sum(item["status"] == "success" for item in state["items"].values()), 500)
            final_records = [json.loads(line) for line in
                             (production / "generation_plan.jsonl").read_text().splitlines()]
            self.assertEqual(
                [record["custom_id"] for record in final_records[:500]],
                [record["custom_id"] for record in pilot_records],
            )
            self.assertTrue(all(record.get("reused_from_pilot")
                                for record in final_records[:500]))

            # Reused Pilot images are deliberately absent from the initial
            # nine request shards, but a later full-run QA failure must still
            # be eligible for a selective corrective Batch.
            reused = final_records[0]
            (production / "auto_qa.jsonl").write_text(json.dumps({
                "custom_id": reused["custom_id"],
                "quality_gate_complete": True,
                "quality_gate_pass": False,
                "quality_gate_reasons": ["insufficient_margin"],
            }) + "\n", encoding="utf-8")
            retry_result = prepare_quality_retry(production)
            self.assertEqual(retry_result["retry_requests"], 1)
            repaired_state = load_state(production)
            self.assertEqual(len(repaired_state["shards"]), 10)
            repair_shard = repaired_state["shards"][-1]
            self.assertTrue(repair_shard["repair_only"])
            self.assertEqual(repair_shard["custom_ids"], [reused["custom_id"]])
            self.assertEqual(repair_shard["attempts"][0]["kind"], "quality_retry")
            self.assertFalse((production / "images" / reused["filename"]).exists())
            self.assertTrue((production / repair_shard["attempts"][0]["archive_dir"] /
                             reused["filename"]).exists())

    def test_request_validator_rejects_all_cost_or_format_overrides(self):
        base = {
            "custom_id": "x", "method": "POST", "url": "/v1/images/generations",
            "body": {
                "model": "gpt-image-2", "prompt": "p", "n": 1,
                "size": "1024x1024", "quality": "low",
                "background": "opaque", "output_format": "png",
            },
        }
        validate_batch_request(base)
        for field, bad in [
            ("model", "gpt-image-1"), ("quality", "high"), ("n", 2),
            ("background", "transparent"), ("output_format", "jpeg"),
            ("size", "1536x1536"),
        ]:
            changed = json.loads(json.dumps(base))
            changed["body"][field] = bad
            with self.assertRaises(PipelineError, msg=field):
                validate_batch_request(changed)

        edit = json.loads(json.dumps(base))
        edit["url"] = EDIT_ENDPOINT
        edit["body"]["images"] = [{
            "image_url": "data:image/jpeg;base64," + jpeg_b64(
                "1024x1024", (1, 2, 3)
            )
        }]
        validate_batch_request(edit, expected_endpoint=EDIT_ENDPOINT)
        with self.assertRaises(PipelineError):
            validate_batch_request(edit)
        broken = json.loads(json.dumps(edit))
        broken["body"]["images"] = [{"image_url": "not-an-image"}]
        with self.assertRaises(PipelineError):
            validate_batch_request(broken, expected_endpoint=EDIT_ENDPOINT)

    def test_lookup_plan_has_exact_upward_quotas_and_jpeg_profile(self):
        config = load_config(LOOKUP_CONFIG)
        records = build_plan(config, "lookup", seed=20260823)
        bin_ids = [item["id"] for item in config["bins"]]
        expected = [600, 750, 600, 300, 300, 300, 150]
        early_expected = [20, 25, 20, 10, 10, 10, 5]
        histogram = Counter(row["bin"] for row in records)
        early_histogram = Counter(row["bin"] for row in records[:100])
        self.assertEqual(len(records), 3000)
        self.assertEqual([histogram[key] for key in bin_ids], expected)
        self.assertEqual([early_histogram[key] for key in bin_ids], early_expected)
        self.assertEqual(Counter(row["size"] for row in records), {
            "1024x1536": 1500, "1024x1024": 900, "1536x1024": 600,
        })
        self.assertEqual(len({row["custom_id"] for row in records}), 3000)
        self.assertTrue(all(row["pitch"] >= 0 for row in records))
        self.assertTrue(all(abs(row["yaw"]) <= 90 for row in records))
        self.assertTrue(all(row["roll"] == 0 for row in records))
        self.assertTrue(all(row["scenario"] and row["scenario"] == row["context"]
                            for row in records))
        self.assertTrue(all(row["filename"].endswith(".jpg") for row in records))
        self.assertTrue(all("frontal or level gaze is unusable" in row["prompt"]
                            for row in records))
        past_vertical = [row for row in records if row["pitch"] > 70]
        self.assertTrue(all(any(word in row["scenario"] for word in [
            "reclined", "lying", "headrest", "supported", "hanging back",
        ]) for row in past_vertical))

        with tempfile.TemporaryDirectory() as temp:
            run_dir = create_plan(
                LOOKUP_CONFIG, "lookup", "lookup-test", Path(temp), 20260823
            )
            state = load_state(run_dir)
            self.assertEqual([len(shard["custom_ids"]) for shard in state["shards"]],
                             [100, 2900])
            self.assertFalse(state["submission_gate"]["released"])
            first = json.loads(
                (run_dir / "batch_input_000_attempt_00.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(first["body"]["output_format"], "jpeg")
            self.assertEqual(first["body"]["output_compression"], 92)
            validate_batch_request(first, state["api_request"])
            scenarios = list(map(
                json.loads, (run_dir / "scenario_log.jsonl").read_text().splitlines()
            ))
            self.assertEqual(len(scenarios), 3000)
            self.assertEqual(len({row["custom_id"] for row in scenarios}), 3000)

    def test_lookup_submit_stops_at_early_gate_then_releases_main(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = create_plan(
                LOOKUP_CONFIG, "lookup", "lookup-gate", Path(temp), 20260823
            )
            client = FakeClient()
            self.assertEqual(submit_pending(run_dir, client), ["batch-1"])
            self.assertEqual(client.batches.create_calls, 1)
            self.assertEqual(prepare_resume(run_dir), 0)

            state = load_state(run_dir)
            early_ids = state["submission_gate"]["custom_ids"]
            plan = {
                row["custom_id"]: row for row in map(
                    json.loads, (run_dir / "generation_plan.jsonl").read_text().splitlines()
                )
            }
            auto_rows = []
            pose_rows = []
            for custom_id, record in plan.items():
                is_early = custom_id in early_ids
                auto_rows.append({
                    "custom_id": custom_id,
                    "quality_gate_pass": is_early,
                    "quality_gate_complete": is_early,
                })
                if is_early:
                    pose_rows.append({
                        "custom_id": custom_id,
                        "bin": record["bin"],
                        "intent_match": "wrong",
                    })
            (run_dir / "auto_qa.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in auto_rows), encoding="utf-8"
            )
            (run_dir / "pose_qa.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in pose_rows), encoding="utf-8"
            )
            review_rows = []
            for custom_id in early_ids:
                image_path = run_dir / "images" / plan[custom_id]["filename"]
                image_path.write_bytes(custom_id.encode("utf-8"))
                review_rows.append({
                    "custom_id": custom_id,
                    "intent_match": "match",
                    "reviewed_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                })
            with (run_dir / "human_review.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["custom_id", "intent_match", "reviewed_sha256"]
                )
                writer.writeheader()
                writer.writerows(review_rows)
            report = finalize_early_drift_report(run_dir)
            self.assertTrue(report["passed"])
            self.assertIn("complete human review", report["intent_source"])
            ids = submit_pending(run_dir, client)
            self.assertEqual(ids, ["batch-1", "batch-2"])
            self.assertEqual(client.batches.create_calls, 2)

    def test_lookup_revises_only_pristine_held_prompts_with_audit_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_config = deepcopy(load_config(LOOKUP_CONFIG))
            old_config["prompt"]["framing"] = "OLD HELD FRAMING."
            old_config["bins"][3]["anchor"] = "OLD PAST-VERTICAL ANCHOR."
            old_config["prompt"]["postures"]["pitch_up_90_120"] = [{
                "scenario": "old merely reclined scenario",
                "backgrounds": ["an old test room"],
            }]
            old_config["auto_correction"]["require_single_head"] = True
            config_path = root / "lookup.yaml"
            config_path.write_text(
                yaml.safe_dump(old_config, sort_keys=False), encoding="utf-8"
            )
            run_dir = create_plan(
                config_path, "lookup", "lookup-revise", root, 20260823
            )
            before_state = load_state(run_dir)
            before_rows = [
                json.loads(line) for line in
                (run_dir / "generation_plan.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            early_input = (run_dir / "batch_input_000_attempt_00.jsonl").read_bytes()
            main_sha = before_state["shards"][1]["attempts"][0]["input_sha256"]

            new_config = deepcopy(load_config(LOOKUP_CONFIG))
            config_path.write_text(
                yaml.safe_dump(new_config, sort_keys=False), encoding="utf-8"
            )
            result = revise_held_prompts(run_dir, config_path)
            self.assertEqual(result["revised_requests"], 2900)
            self.assertEqual(result["immutable_early_requests"], 100)

            after_state = load_state(run_dir)
            after_rows = [
                json.loads(line) for line in
                (run_dir / "generation_plan.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(before_rows[:100], after_rows[:100])
            self.assertEqual(
                [(row["custom_id"], row["pitch"], row["yaw"], row["cam"], row["size"])
                 for row in before_rows],
                [(row["custom_id"], row["pitch"], row["yaw"], row["cam"], row["size"])
                 for row in after_rows],
            )
            self.assertTrue(all(
                "targeting 36% to 42%" in row["prompt"] for row in after_rows[100:]
            ))
            held_past_vertical = [
                row for row in after_rows[100:] if row["bin"] == "pitch_up_90_120"
            ]
            self.assertTrue(all(
                "head hangs backward beyond" in row["anchor"]
                for row in held_past_vertical
            ))
            self.assertTrue(all(
                row["scenario"] != "old merely reclined scenario"
                for row in held_past_vertical
            ))
            self.assertEqual(
                early_input, (run_dir / "batch_input_000_attempt_00.jsonl").read_bytes()
            )
            self.assertNotEqual(
                main_sha, after_state["shards"][1]["attempts"][0]["input_sha256"]
            )
            self.assertFalse(after_state["auto_correction"]["require_single_head"])
            self.assertEqual(
                after_state["prompt_revision_history"][-1]["held_requests"], 2900
            )
            self.assertEqual(submit_pending(run_dir, FakeClient()), ["batch-1"])
            repeat = revise_held_prompts(run_dir, config_path)
            self.assertTrue(repeat["idempotent"])
            self.assertEqual(repeat["revised_requests"], 0)

    def test_lookup_revises_only_pristine_unsent_prompts_after_rolling_qa(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = create_plan(
                LOOKUP_CONFIG, "lookup", "lookup-rolling-revise", Path(temp), 20260823
            )
            state = load_state(run_dir)
            early_attempt = state["shards"][0]["attempts"][0]
            early_attempt.update({
                "batch_id": "batch-early-completed",
                "input_file_id": "file-early",
                "status": "completed",
            })
            (run_dir / "batch_state.json").write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            early_before = (run_dir / early_attempt["input_path"]).read_bytes()
            main_before = (
                run_dir / state["shards"][1]["attempts"][0]["input_path"]
            ).read_bytes()
            plan_before = [
                json.loads(line) for line in
                (run_dir / "generation_plan.jsonl").read_text().splitlines()
            ]
            instruction = (
                "place the entire head at exact canvas center and keep it inside "
                "the central 60% with half-head empty margin on every side"
            )
            result = revise_unsent_prompts(run_dir, instruction)
            self.assertEqual(result["revised_requests"], 2900)
            self.assertEqual(result["inputs"], 1)
            self.assertEqual(
                early_before, (run_dir / early_attempt["input_path"]).read_bytes()
            )
            self.assertNotEqual(
                main_before,
                (run_dir / state["shards"][1]["attempts"][0]["input_path"]).read_bytes(),
            )
            plan_after = [
                json.loads(line) for line in
                (run_dir / "generation_plan.jsonl").read_text().splitlines()
            ]
            self.assertEqual(plan_before[:100], plan_after[:100])
            self.assertTrue(all(instruction in row["prompt"] for row in plan_after[100:]))
            self.assertTrue(revise_unsent_prompts(run_dir, instruction)["idempotent"])
            rollback = rollback_latest_unsent_prompt_revision(run_dir)
            self.assertEqual(rollback["rolled_back_requests"], 2900)
            self.assertEqual(
                main_before,
                (run_dir / state["shards"][1]["attempts"][0]["input_path"]).read_bytes(),
            )
            plan_rolled_back = [
                json.loads(line) for line in
                (run_dir / "generation_plan.jsonl").read_text().splitlines()
            ]
            self.assertEqual(plan_before, plan_rolled_back)

    def test_lookup_direct_jpeg_response_is_saved_without_png_conversion(self):
        config = load_config(LOOKUP_CONFIG)
        record = build_plan(config, "lookup", seed=5)[0]
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            (run_dir / "images").mkdir()
            output = run_dir / "output.jsonl"
            output.write_text(json.dumps({
                "custom_id": record["custom_id"],
                "response": {
                    "status_code": 200,
                    "body": {"data": [{"b64_json": jpeg_b64(record["size"], (9, 8, 7))}]},
                },
            }) + "\n", encoding="utf-8")
            state = {
                "api_request": config["api"],
                "items": {record["custom_id"]: {
                    "status": "planned", "filename": record["filename"],
                }},
            }
            changed = process_output_jsonl(
                output, run_dir, state, {record["custom_id"]: record}
            )
            self.assertEqual(changed, {record["custom_id"]})
            with Image.open(run_dir / "images" / record["filename"]) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(f"{image.width}x{image.height}", record["size"])


class BatchStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = create_plan(CONFIG, "validation", "validation-test", self.root, 3)

    def tearDown(self):
        self.temp.cleanup()

    def test_submit_is_idempotent(self):
        client = FakeClient()
        first = submit_pending(self.run_dir, client)
        second = submit_pending(self.run_dir, client)
        self.assertEqual(first, ["batch-1"])
        self.assertEqual(second, ["batch-1"])
        self.assertEqual(client.files.create_calls, 1)
        self.assertEqual(client.batches.create_calls, 1)
        state = load_state(self.run_dir)
        attempt = state["shards"][0]["attempts"][0]
        self.assertEqual(attempt["input_file_id"], "file-input-1")
        self.assertEqual(attempt["batch_id"], "batch-1")

    def test_archived_inputs_remain_submit_compatible_and_idempotent(self):
        state = load_state(self.run_dir)
        attempt = state["shards"][0]["attempts"][0]
        input_path = self.run_dir / attempt["input_path"]
        original_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()

        result = archive_batch_inputs(self.run_dir)
        self.assertEqual(result["files_archived"], 1)
        self.assertFalse(input_path.exists())
        self.assertTrue((self.run_dir / "batch_inputs.zip").exists())
        refreshed = load_state(self.run_dir)
        self.assertEqual(refreshed["input_archiving"]["status"], "completed")
        self.assertEqual(
            refreshed["shards"][0]["attempts"][0]["input_sha256"], original_sha
        )
        self.assertTrue(archive_batch_inputs(self.run_dir)["already_archived"])

        client = FakeClient()
        self.assertEqual(submit_pending(self.run_dir, client), ["batch-1"])
        self.assertEqual(client.files.create_calls, 1)

    def test_archived_inputs_validate_image_edit_endpoint(self):
        state = load_state(self.run_dir)
        attempt = state["shards"][0]["attempts"][0]
        input_path = self.run_dir / attempt["input_path"]
        requests = [
            json.loads(line) for line in input_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        source = "data:image/jpeg;base64," + jpeg_b64(
            "1024x1024", (10, 20, 30)
        )
        for request in requests:
            request["url"] = EDIT_ENDPOINT
            request["body"]["images"] = [{"image_url": source}]
        payload = "".join(json.dumps(request) + "\n" for request in requests)
        input_path.write_text(payload, encoding="utf-8")
        attempt["endpoint"] = EDIT_ENDPOINT
        attempt["input_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        result = archive_batch_inputs(self.run_dir)
        self.assertEqual(result["files_archived"], 1)
        self.assertFalse(input_path.exists())
        self.assertTrue((self.run_dir / "batch_inputs.zip").exists())

    def test_archived_inputs_support_resume_and_later_rearchive(self):
        archive_batch_inputs(self.run_dir)
        state = load_state(self.run_dir)
        original = state["shards"][0]["attempts"][0]
        original.update({"status": "expired", "batch_id": "batch-expired"})
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        self.assertEqual(prepare_resume(self.run_dir), 10)
        refreshed = load_state(self.run_dir)
        retry = refreshed["shards"][0]["attempts"][1]
        retry_path = self.run_dir / retry["input_path"]
        self.assertTrue(retry_path.exists())
        self.assertEqual(len(retry_path.read_text(encoding="utf-8").splitlines()), 10)

        result = archive_batch_inputs(self.run_dir)
        self.assertEqual(result["files_archived"], 2)
        self.assertFalse(retry_path.exists())
        self.assertEqual(
            load_state(self.run_dir)["shards"][0]["attempts"][1]["input_archive_member"],
            retry["input_path"],
        )

    def test_fresh_replacement_limit_can_be_audited_to_round_seven(self):
        result = extend_fresh_replacement_limit(self.run_dir, 7)
        self.assertEqual(result, {"previous_max": 3, "new_max": 7})
        state = load_state(self.run_dir)
        self.assertEqual(state["fresh_replacement_max_rounds"], 7)
        self.assertEqual(state["fresh_replacement_limit_history"][-1]["new_max"], 7)

    def test_watch_auto_resume_requeues_missing_before_final_qa(self):
        failed = {"stage": "production", "batches": [
            {"status": "in_progress"}, {"status": "failed"},
        ]}
        completed = {"stage": "production", "batches": [{"status": "completed"}]}
        with (
            patch("hrffa.dataset.augment.gpt_head_gen.refresh_status",
                  side_effect=[failed, completed]),
            patch("hrffa.dataset.augment.gpt_head_gen.collect_results",
                  side_effect=[
                      {"success": 8, "total": 10, "missing": 2},
                      {"success": 10, "total": 10, "missing": 0},
                      {"success": 10, "total": 10, "missing": 0, "quality_pass": 10},
                  ]) as collect,
            patch("hrffa.dataset.augment.gpt_head_gen.prepare_resume", return_value=2),
            patch("hrffa.dataset.augment.gpt_head_gen.submit_pending",
                  return_value=["batch-retry"]),
        ):
            result = watch_until_terminal(
                self.run_dir, interval_seconds=5, auto_resume=True, client=object()
            )
        self.assertEqual(result["collection"]["quality_pass"], 10)
        self.assertEqual(collect.call_count, 3)
        self.assertFalse(collect.call_args_list[0].kwargs["prepare_review"])

    def test_watch_auto_repair_switches_to_bounded_fresh_replacement(self):
        completed = {"stage": "production", "batches": [{"status": "completed"}]}
        with (
            patch("hrffa.dataset.augment.gpt_head_gen.refresh_status",
                  side_effect=[completed, completed]),
            patch("hrffa.dataset.augment.gpt_head_gen.collect_results",
                  return_value={
                      "success": 10, "total": 10, "missing": 0,
                      "quality_pass": 9, "quality_failed": 1,
                      "quality_incomplete": 0,
                  }),
            patch("hrffa.dataset.augment.gpt_head_gen.prepare_quality_retry",
                  side_effect=[{
                      "retry_requests": 0, "exhausted": ["stubborn"],
                      "incomplete_machine_qa": [], "reasons": {},
                  }, {
                      "retry_requests": 0, "exhausted": [],
                      "incomplete_machine_qa": [], "reasons": {},
                  }]),
            patch("hrffa.dataset.augment.gpt_head_gen.prepare_fresh_replacement",
                  return_value={
                      "retry_requests": 1, "fresh_round": 1,
                      "reasons": {"stubborn": ["insufficient_margin"]},
                  }) as fresh,
            patch("hrffa.dataset.augment.gpt_head_gen.submit_pending",
                  return_value=["batch-fresh"]),
        ):
            result = watch_until_terminal(
                self.run_dir, interval_seconds=5, auto_repair=True, client=object()
            )
        fresh.assert_called_once_with(self.run_dir)
        self.assertEqual(result["quality_repair"]["retry_requests"], 0)

    def test_remote_sha_reconciliation_prevents_duplicate_submit(self):
        client = FakeClient()
        state = load_state(self.run_dir)
        attempt = state["shards"][0]["attempts"][0]
        remote = SimpleNamespace(
            id="batch-existing", status="in_progress", output_file_id=None,
            error_file_id=None, request_counts=None,
            metadata={
                "local_batch_id": "validation-test", "stage": "validation",
                "shard": "0", "attempt": "0", "input_sha256": attempt["input_sha256"],
            },
        )
        client.batches.listed.append(remote)
        client.batches.objects[remote.id] = remote
        self.assertEqual(submit_pending(self.run_dir, client), ["batch-existing"])
        self.assertEqual(client.files.create_calls, 0)
        self.assertEqual(client.batches.create_calls, 0)

    def test_unordered_partial_expired_results_retry_only_missing(self):
        client = FakeClient()
        submit_pending(self.run_dir, client)
        state = load_state(self.run_dir)
        plan = {}
        with (self.run_dir / state["plan_path"]).open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                plan[row["custom_id"]] = row
        ids = list(plan)
        output_rows = []
        for custom_id, color in [(ids[1], (2, 3, 4)), (ids[0], (1, 2, 3))]:
            output_rows.append({
                "custom_id": custom_id,
                "response": {"status_code": 200, "body": {"data": [{
                    "b64_json": png_b64(plan[custom_id]["size"], color)
                }]}},
            })
        client.files.payloads["file-output"] = ("\n".join(json.dumps(row) for row in output_rows) + "\n").encode()
        client.files.payloads["file-error"] = (json.dumps({
            "custom_id": ids[2], "error": {"code": "image_generation_failed", "message": "failed"}
        }) + "\n").encode()
        batch = client.batches.objects["batch-1"]
        batch.status = "expired"
        batch.output_file_id = "file-output"
        batch.error_file_id = "file-error"
        batch.request_counts = SimpleNamespace(completed=2, failed=1, total=10)

        counts = collect_results(self.run_dir, client, prepare_review=False)
        self.assertEqual(counts, {"success": 2, "total": 10, "missing": 8})
        self.assertEqual(
            set(load_state(self.run_dir)["pending_qa_ids"]), set(ids[:2])
        )
        self.assertTrue((self.run_dir / "images" / f"{ids[0]}.jpg").exists())
        self.assertTrue((self.run_dir / "images" / f"{ids[1]}.jpg").exists())
        self.assertEqual(prepare_resume(self.run_dir), 8)
        retry = load_state(self.run_dir)["shards"][0]["attempts"][1]
        self.assertEqual(set(retry["custom_ids"]), set(ids[2:]))
        retry_lines = (self.run_dir / retry["input_path"]).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(retry_lines), 8)
        self.assertNotIn(ids[0], {json.loads(line)["custom_id"] for line in retry_lines})
        # A crash after writing retry JSONL but before submit must not create another attempt.
        self.assertEqual(prepare_resume(self.run_dir), 8)
        self.assertEqual(len(load_state(self.run_dir)["shards"][0]["attempts"]), 2)

    def test_failed_full_retry_is_distinct_from_crash_reconciliation(self):
        client = FakeClient()
        submit_pending(self.run_dir, client)
        original = client.batches.objects["batch-1"]
        original.status = "failed"
        client.batches.listed.append(original)
        # Persist the terminal status, then prepare the same ten lines as attempt 1.
        from hrffa.dataset.augment.gpt_head_gen import refresh_status
        refresh_status(self.run_dir, client)
        self.assertEqual(prepare_resume(self.run_dir), 10)
        ids = submit_pending(self.run_dir, client)
        self.assertEqual(ids, ["batch-1", "batch-2"])
        self.assertEqual(client.batches.create_calls, 2)

    def test_token_limited_shard_is_split_and_submitted_sequentially(self):
        state = load_state(self.run_dir)
        source = state["shards"][0]
        source_attempt = source["attempts"][0]
        source_attempt.update({
            "status": "failed",
            "batch_id": "batch-token-limited",
            "request_counts": {"completed": 0, "failed": 0, "total": 0},
            "batch_errors": {"data": [{
                "code": "token_limit_exceeded",
                "message": "enqueued token limit reached",
            }]},
        })
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(PipelineError, "split-token-limit"):
            prepare_resume(self.run_dir)
        split = split_token_limited_shard(self.run_dir, max_requests=4)
        self.assertEqual(split["replacement_counts"], [4, 4, 2])
        refreshed = load_state(self.run_dir)
        self.assertTrue(refreshed["shards"][0]["retired"])
        self.assertTrue(all(
            attempt["superseded"]
            for attempt in refreshed["shards"][0]["attempts"]
        ))
        replacements = [
            shard for shard in refreshed["shards"] if shard.get("token_limit_reshard")
        ]
        self.assertEqual([len(shard["custom_ids"]) for shard in replacements], [4, 4, 2])
        self.assertEqual(len({shard["sequential_group"] for shard in replacements}), 1)

        client = FakeClient()
        self.assertEqual(submit_pending(self.run_dir, client), ["batch-1"])
        self.assertEqual(client.batches.create_calls, 1)
        after_submit = load_state(self.run_dir)
        replacement_attempts = [
            shard["attempts"][0]
            for shard in after_submit["shards"] if shard.get("token_limit_reshard")
        ]
        self.assertIsNotNone(replacement_attempts[0]["batch_id"])
        self.assertIsNone(replacement_attempts[1]["batch_id"])
        self.assertIsNone(replacement_attempts[2]["batch_id"])

    def test_stale_output_cannot_restore_id_owned_by_newer_attempt(self):
        state = load_state(self.run_dir)
        records = [
            json.loads(line) for line in
            (self.run_dir / state["plan_path"]).read_text(encoding="utf-8").splitlines()
        ]
        output = self.run_dir / "stale-output.jsonl"
        output.write_text("".join(
            json.dumps({
                "custom_id": record["custom_id"],
                "response": {
                    "status_code": 200,
                    "body": {"data": [{"b64_json": png_b64(record["size"], (4, 5, 6))}]},
                },
            }) + "\n"
            for record in records[:2]
        ), encoding="utf-8")

        changed = process_output_jsonl(
            output, self.run_dir, state,
            {record["custom_id"]: record for record in records},
            replace_existing=True,
            eligible_ids={records[1]["custom_id"]},
        )
        self.assertEqual(changed, {records[1]["custom_id"]})
        self.assertFalse((self.run_dir / "images" / records[0]["filename"]).exists())
        self.assertTrue((self.run_dir / "images" / records[1]["filename"]).exists())

    def test_quality_retry_uses_reason_specific_prompt_and_is_idempotent(self):
        state = load_state(self.run_dir)
        records = []
        with (self.run_dir / state["plan_path"]).open(encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh]
        for index, record in enumerate(records):
            width, height = (int(value) for value in record["size"].split("x"))
            Image.new("RGB", (width, height), (index, index + 1, index + 2)).save(
                self.run_dir / "images" / record["filename"], "JPEG", quality=92
            )
            state["items"][record["custom_id"]]["status"] = "success"
        state["shards"][0]["attempts"][0]["status"] = "completed"
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        prepare_review_artifacts(self.run_dir, use_detector=False)

        qa_rows = []
        for index, record in enumerate(records):
            qa_rows.append({
                "custom_id": record["custom_id"],
                "quality_gate_complete": True,
                "quality_gate_pass": index != 0,
                "quality_gate_reasons": ["head_too_small"] if index == 0 else [],
            })
        (self.run_dir / "auto_qa.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in qa_rows), encoding="utf-8"
        )
        review_path = self.run_dir / "human_review.csv"
        with review_path.open(newline="", encoding="utf-8-sig") as fh:
            review_rows = list(csv.DictReader(fh))
            fields = list(review_rows[0])
        review_rows[1]["body_integrity"] = "fail"
        with review_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(review_rows)

        state = load_state(self.run_dir)
        state["auto_correction"]["enabled_stages"] = ["validation", "pilot", "production"]
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        override = "move camera slightly farther back; keep head height strictly 30% to 35%, not 50%"
        set_retry_prompt_override(self.run_dir, [records[0]["custom_id"]], override)
        result = prepare_quality_retry(self.run_dir)
        expected_ids = {records[0]["custom_id"], records[1]["custom_id"]}
        self.assertEqual(result["retry_requests"], 2)
        self.assertEqual(set(result["reasons"]), expected_ids)
        retry = load_state(self.run_dir)["shards"][0]["attempts"][-1]
        self.assertEqual(retry["kind"], "quality_retry")
        lines = [json.loads(line) for line in (self.run_dir / retry["input_path"]).read_text().splitlines()]
        prompts = {row["custom_id"]: row["body"]["prompt"] for row in lines}
        self.assertIn("Never fall below the 25%", prompts[records[0]["custom_id"]])
        self.assertNotIn(
            "Move the camera closer instead of using a distant wide composition",
            prompts[records[0]["custom_id"]],
        )
        self.assertEqual(
            prompts[records[0]["custom_id"]].count("QUALITY CORRECTION ROUND"), 1
        )
        self.assertIn(override, prompts[records[0]["custom_id"]])
        self.assertNotIn(override, prompts[records[1]["custom_id"]])
        self.assertIn("Never dissolve or merge the torso", prompts[records[1]["custom_id"]])
        self.assertTrue(all(not (self.run_dir / "images" / f"{custom_id}.jpg").exists()
                            for custom_id in expected_ids))
        self.assertEqual(prepare_quality_retry(self.run_dir)["retry_requests"], 2)
        self.assertEqual(len(load_state(self.run_dir)["shards"][0]["attempts"]), 2)

        # If the corrective Batch expires, API resume must retain its corrective
        # prompts instead of falling back to the immutable original plan.
        state = load_state(self.run_dir)
        state["shards"][0]["attempts"][-1].update({
            "status": "expired", "batch_id": "batch-quality-expired",
        })
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self.assertEqual(prepare_resume(self.run_dir), 2)
        api_retry = load_state(self.run_dir)["shards"][0]["attempts"][-1]
        retry_lines = [
            json.loads(line) for line in
            (self.run_dir / api_retry["input_path"]).read_text().splitlines()
        ]
        retry_prompts = {row["custom_id"]: row["body"]["prompt"] for row in retry_lines}
        self.assertIn("Never fall below the 25%", retry_prompts[records[0]["custom_id"]])
        self.assertIn("Never dissolve or merge the torso", retry_prompts[records[1]["custom_id"]])

    def test_geometry_edit_retry_embeds_source_and_resumes_same_edit(self):
        state = load_state(self.run_dir)
        records = [
            json.loads(line) for line in
            (self.run_dir / state["plan_path"]).read_text(encoding="utf-8").splitlines()
        ]
        source_bytes: dict[str, bytes] = {}
        for index, record in enumerate(records):
            width, height = (int(value) for value in record["size"].split("x"))
            path = self.run_dir / "images" / record["filename"]
            Image.new("RGB", (width, height), (index, 60, 90)).save(
                path, "JPEG", quality=92
            )
            source_bytes[record["custom_id"]] = path.read_bytes()
            state["items"][record["custom_id"]]["status"] = "success"
        state["shards"][0]["attempts"][0]["status"] = "completed"
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        geometry_id = records[0]["custom_id"]
        non_geometry_id = records[1]["custom_id"]
        (self.run_dir / "auto_qa.jsonl").write_text("".join(
            json.dumps({
                "custom_id": record["custom_id"],
                "quality_gate_complete": True,
                "quality_gate_pass": record["custom_id"] not in {
                    geometry_id, non_geometry_id,
                },
                "quality_gate_reasons": (
                    ["insufficient_margin"] if record["custom_id"] == geometry_id
                    else ["body_not_detected"] if record["custom_id"] == non_geometry_id
                    else []
                ),
            }) + "\n"
            for record in records
        ), encoding="utf-8")

        result = prepare_image_edit_retry(self.run_dir)
        self.assertEqual(result["retry_requests"], 1)
        self.assertEqual(result["skipped_non_geometry"], [non_geometry_id])
        refreshed = load_state(self.run_dir)
        edit_shard = refreshed["shards"][-1]
        edit_attempt = edit_shard["attempts"][0]
        self.assertEqual(edit_attempt["kind"], "image_edit")
        self.assertEqual(edit_attempt["endpoint"], EDIT_ENDPOINT)
        request = json.loads(
            (self.run_dir / edit_attempt["input_path"]).read_text().splitlines()[0]
        )
        validate_batch_request(request, refreshed["api_request"], EDIT_ENDPOINT)
        encoded = request["body"]["images"][0]["image_url"].partition(",")[2]
        self.assertEqual(base64.b64decode(encoded), source_bytes[geometry_id])
        self.assertIn("EDIT THE SUPPLIED SOURCE PHOTOGRAPH", request["body"]["prompt"])
        self.assertFalse((self.run_dir / "images" / records[0]["filename"]).exists())
        self.assertTrue(
            (self.run_dir / edit_attempt["archive_dir"] / records[0]["filename"]).exists()
        )
        restored = restore_failed_image_edit_sources(self.run_dir, [geometry_id])
        self.assertEqual(restored["restored"], 1)
        restored_path = self.run_dir / "images" / records[0]["filename"]
        self.assertEqual(restored_path.read_bytes(), source_bytes[geometry_id])
        restored_path.unlink()

        client = FakeClient()
        submit_pending(self.run_dir, client)
        self.assertEqual(client.batches.create_kwargs[-1]["endpoint"], EDIT_ENDPOINT)

        refreshed = load_state(self.run_dir)
        edit_attempt = refreshed["shards"][-1]["attempts"][0]
        edit_attempt.update({"status": "expired", "batch_id": "batch-edit-expired"})
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(refreshed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self.assertEqual(prepare_resume(self.run_dir), 1)
        resumed = load_state(self.run_dir)["shards"][-1]["attempts"][-1]
        self.assertEqual(resumed["endpoint"], EDIT_ENDPOINT)
        resumed_request = json.loads(
            (self.run_dir / resumed["input_path"]).read_text().splitlines()[0]
        )
        self.assertEqual(
            resumed_request["body"]["images"], request["body"]["images"]
        )

    def test_geometry_edit_limit_is_scoped_to_source_sha(self):
        state = load_state(self.run_dir)
        records = [
            json.loads(line) for line in
            (self.run_dir / state["plan_path"]).read_text(encoding="utf-8").splitlines()
        ]
        geometry_id = records[0]["custom_id"]
        geometry_record = records[0]
        for index, record in enumerate(records):
            width, height = (int(value) for value in record["size"].split("x"))
            Image.new("RGB", (width, height), (index, 60, 90)).save(
                self.run_dir / "images" / record["filename"], "JPEG", quality=92
            )
            state["items"][record["custom_id"]]["status"] = "success"
        state["shards"][0]["attempts"][0]["status"] = "completed"
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (self.run_dir / "auto_qa.jsonl").write_text("".join(
            json.dumps({
                "custom_id": record["custom_id"],
                "quality_gate_complete": True,
                "quality_gate_pass": record["custom_id"] != geometry_id,
                "quality_gate_reasons": (
                    ["insufficient_margin"] if record["custom_id"] == geometry_id else []
                ),
            }) + "\n"
            for record in records
        ), encoding="utf-8")

        self.assertEqual(prepare_image_edit_retry(self.run_dir)["retry_requests"], 1)
        state = load_state(self.run_dir)
        first_attempt = state["shards"][-1]["attempts"][0]
        first_attempt["status"] = "completed"
        first_attempt["batch_id"] = "batch-edit-completed"
        source_archive = self.run_dir / first_attempt["archive_dir"] / geometry_record["filename"]
        current_source = self.run_dir / "images" / geometry_record["filename"]
        current_source.write_bytes(source_archive.read_bytes())
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        same_source = prepare_image_edit_retry(self.run_dir)
        self.assertEqual(same_source["retry_requests"], 0)
        self.assertEqual(same_source["exhausted"], [geometry_id])

        width, height = (int(value) for value in geometry_record["size"].split("x"))
        Image.new("RGB", (width, height), (220, 120, 30)).save(
            current_source, "JPEG", quality=92
        )
        new_source = prepare_image_edit_retry(self.run_dir)
        self.assertEqual(new_source["retry_requests"], 1)
        second_attempt = load_state(self.run_dir)["shards"][-1]["attempts"][0]
        self.assertNotEqual(
            second_attempt["source_sha256"][geometry_id],
            first_attempt["source_sha256"][geometry_id],
        )

    def test_local_crop_repairs_only_head_too_small_and_archives_source(self):
        state = load_state(self.run_dir)
        records = [
            json.loads(line) for line in
            (self.run_dir / state["plan_path"]).read_text(encoding="utf-8").splitlines()
        ]
        target = records[0]
        for index, record in enumerate(records):
            width, height = (int(value) for value in record["size"].split("x"))
            Image.new("RGB", (width, height), (40 + index, 80, 120)).save(
                self.run_dir / "images" / record["filename"], "JPEG", quality=92
            )
            state["items"][record["custom_id"]]["status"] = "success"
        state["shards"][0]["attempts"][0]["status"] = "completed"
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        source = self.run_dir / "images" / target["filename"]
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        (self.run_dir / "auto_qa.jsonl").write_text("".join(
            json.dumps({
                "custom_id": record["custom_id"],
                "quality_gate_complete": True,
                "quality_gate_pass": record["custom_id"] != target["custom_id"],
                "quality_gate_reasons": (
                    ["head_too_small"] if record["custom_id"] == target["custom_id"] else []
                ),
                "head_box_xyxy": (
                    [300.0, 400.0, 600.0, 700.0]
                    if record["custom_id"] == target["custom_id"] else None
                ),
            }) + "\n"
            for record in records
        ), encoding="utf-8")

        result = apply_local_crop_repair(self.run_dir)
        self.assertEqual(result["repaired"], 1)
        self.assertEqual(result["custom_ids"], [target["custom_id"]])
        with Image.open(source) as image:
            self.assertEqual(f"{image.width}x{image.height}", target["size"])
            self.assertEqual(image.format, "JPEG")
        self.assertNotEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_sha)
        event = load_state(self.run_dir)["local_crop_history"][0]
        self.assertEqual(event["entries"][0]["source_sha256"], source_sha)
        self.assertTrue((self.run_dir / event["entries"][0]["source_path"]).exists())
        manifest = {
            row["custom_id"]: row for row in (
                json.loads(line) for line in
                (self.run_dir / "image_sha256.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            )
        }
        self.assertEqual(
            manifest[target["custom_id"]]["sha256"],
            hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        self.assertEqual(apply_local_crop_repair(self.run_dir)["repaired"], 0)
        width, height = (int(value) for value in target["size"].split("x"))
        Image.new("RGB", (width, height), (220, 130, 40)).save(
            source, "JPEG", quality=92
        )
        self.assertEqual(apply_local_crop_repair(self.run_dir)["repaired"], 1)

    def test_masked_pose_teacher_wrong_is_audit_only(self):
        record = {"bin": "pitch_up_40_60", "accessories": "a medical mask"}
        qa = {"has_nose": True, "has_mouth": True}
        self.assertFalse(
            _intent_is_confident_failure(record, qa, "wrong", 0.0, 0.0)
        )
        record["accessories"] = "none"
        self.assertTrue(
            _intent_is_confident_failure(record, qa, "wrong", 0.0, 0.0)
        )

        low_camera = {
            "bin": "camera_low_20_60", "accessories": "a medical mask"
        }
        self.assertFalse(_roll_is_reliable(low_camera))
        low_camera["accessories"] = "none"
        self.assertTrue(_roll_is_reliable(low_camera))

    def test_human_pose_adjudication_is_bound_to_current_image_sha(self):
        teacher = {
            "intent_match": "wrong",
            "basis": "sixd_frontal_or_inverted_collapse",
            "intent_hard_failure": True,
            "roll_no_back": False,
            "roll_hard_failure": True,
        }
        review = {
            "intent_match": "match",
            "roll_no_back": "pass",
            "reviewed_sha256": "current-sha",
        }
        adjudicated = _apply_current_human_pose_review(
            teacher, review, "current-sha"
        )
        self.assertEqual(adjudicated["intent_match"], "match")
        self.assertEqual(adjudicated["basis"], "human_review_sha256_bound")
        self.assertFalse(adjudicated["intent_hard_failure"])
        self.assertTrue(adjudicated["roll_no_back"])
        self.assertFalse(adjudicated["roll_hard_failure"])
        self.assertEqual(adjudicated["teacher_intent_match"], "wrong")
        self.assertEqual(
            _apply_current_human_pose_review(teacher, review, "new-sha"), teacher
        )

    def test_pending_pose_qa_completion_requires_current_image_sha(self):
        state = load_state(self.run_dir)
        records = [
            json.loads(line) for line in
            (self.run_dir / state["plan_path"]).read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        record = records[0]
        image_path = self.run_dir / "images" / record["filename"]
        width, height = (int(value) for value in record["size"].split("x"))
        Image.new("RGB", (width, height), (20, 30, 40)).save(
            image_path, "JPEG", quality=92
        )
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        state["pending_qa_ids"] = [record["custom_id"]]
        plan = {record["custom_id"]: record}
        auto = {record["custom_id"]: {
            "quality_gate_complete": True, "sha256": digest,
        }}
        self.assertEqual(
            _completed_pending_pose_ids(
                self.run_dir, state, plan, auto, {record["custom_id"]}
            ),
            {record["custom_id"]},
        )
        auto[record["custom_id"]]["sha256"] = "stale"
        self.assertEqual(
            _completed_pending_pose_ids(
                self.run_dir, state, plan, auto, {record["custom_id"]}
            ),
            set(),
        )

    def test_fresh_replacement_aggregates_failures_into_one_batch(self):
        state = load_state(self.run_dir)
        records = [
            json.loads(line) for line in
            (self.run_dir / state["plan_path"]).read_text(encoding="utf-8").splitlines()
        ]
        for index, record in enumerate(records):
            width, height = (int(value) for value in record["size"].split("x"))
            Image.new("RGB", (width, height), (index, 30, 60)).save(
                self.run_dir / "images" / record["filename"], "JPEG", quality=92
            )
            state["items"][record["custom_id"]]["status"] = "success"
        state["shards"][0]["attempts"][0]["status"] = "completed"
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        failed_ids = [records[1]["custom_id"], records[8]["custom_id"]]
        (self.run_dir / "auto_qa.jsonl").write_text("".join(
            json.dumps({
                "custom_id": record["custom_id"],
                "quality_gate_complete": True,
                "quality_gate_pass": record["custom_id"] not in failed_ids,
                "quality_gate_reasons": (
                    ["insufficient_margin"] if record["custom_id"] in failed_ids else []
                ),
            }) + "\n"
            for record in records
        ), encoding="utf-8")

        human_failed = records[4]
        human_path = self.run_dir / "images" / human_failed["filename"]
        with (self.run_dir / "human_review.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "custom_id", "filename", "body_integrity", "reviewed_sha256"
            ])
            writer.writeheader()
            writer.writerow({
                "custom_id": human_failed["custom_id"],
                "filename": human_failed["filename"],
                "body_integrity": "fail",
                "reviewed_sha256": hashlib.sha256(human_path.read_bytes()).hexdigest(),
            })

        result = prepare_fresh_replacement(self.run_dir)
        self.assertEqual(result["retry_requests"], 3)
        refreshed = load_state(self.run_dir)
        aggregate = [
            shard for shard in refreshed["shards"] if shard.get("aggregate_correction")
        ]
        self.assertEqual(len(aggregate), 1)
        expected_ids = [
            records[1]["custom_id"], records[4]["custom_id"], records[8]["custom_id"]
        ]
        self.assertEqual(aggregate[0]["custom_ids"], expected_ids)
        self.assertEqual(len(aggregate[0]["attempts"]), 1)
        lines = (
            self.run_dir / aggregate[0]["attempts"][0]["input_path"]
        ).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)

    def test_resume_does_not_duplicate_an_active_aggregate_replacement(self):
        state = load_state(self.run_dir)
        records = [
            json.loads(line) for line in
            (self.run_dir / state["plan_path"]).read_text(encoding="utf-8").splitlines()
        ]
        for index, record in enumerate(records):
            width, height = (int(value) for value in record["size"].split("x"))
            Image.new("RGB", (width, height), (index, 40, 70)).save(
                self.run_dir / "images" / record["filename"], "JPEG", quality=92
            )
            state["items"][record["custom_id"]]["status"] = "success"
        state["shards"][0]["attempts"][0]["status"] = "completed"
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        failed_id = records[2]["custom_id"]
        (self.run_dir / "auto_qa.jsonl").write_text("".join(
            json.dumps({
                "custom_id": record["custom_id"],
                "quality_gate_complete": True,
                "quality_gate_pass": record["custom_id"] != failed_id,
                "quality_gate_reasons": (
                    ["insufficient_margin"] if record["custom_id"] == failed_id else []
                ),
            }) + "\n"
            for record in records
        ), encoding="utf-8")

        prepare_fresh_replacement(self.run_dir)
        state = load_state(self.run_dir)
        aggregate = state["shards"][-1]
        aggregate["attempts"][0].update({
            "status": "in_progress", "batch_id": "batch-active-fresh",
        })
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        self.assertEqual(prepare_resume(self.run_dir), 0)
        refreshed = load_state(self.run_dir)
        self.assertEqual(len(refreshed["shards"][0]["attempts"]), 1)
        self.assertEqual(len(refreshed["shards"][-1]["attempts"]), 1)

    def test_prune_outputs_preserves_audit_manifest_and_is_idempotent(self):
        state = load_state(self.run_dir)
        records = [
            json.loads(line) for line in
            (self.run_dir / state["plan_path"]).read_text(encoding="utf-8").splitlines()
        ]
        image_rows = []
        qa_rows = []
        for index, record in enumerate(records):
            width, height = (int(value) for value in record["size"].split("x"))
            path = self.run_dir / "images" / record["filename"]
            Image.new("RGB", (width, height), (index, 50, 80)).save(
                path, "JPEG", quality=92
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            state["items"][record["custom_id"]].update({
                "status": "success", "sha256": digest,
            })
            image_rows.append({
                "custom_id": record["custom_id"], "filename": record["filename"],
                "sha256": digest, "duplicate_of": None,
            })
            qa_rows.append({
                "custom_id": record["custom_id"],
                "quality_gate_complete": True,
                "quality_gate_pass": True,
                "quality_gate_reasons": [],
            })
        (self.run_dir / "image_sha256.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in image_rows), encoding="utf-8"
        )
        (self.run_dir / "auto_qa.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in qa_rows), encoding="utf-8"
        )
        output_name = "shard_000_attempt_00_output.jsonl.gz"
        output_path = self.run_dir / output_name
        response = {
            "custom_id": records[0]["custom_id"],
            "response": {
                "status_code": 200,
                "body": {"usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "input_tokens_details": {"text_tokens": 10, "image_tokens": 0},
                    "output_tokens_details": {"text_tokens": 0, "image_tokens": 20},
                }},
            },
        }
        with gzip.open(output_path, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps(response) + "\n")
        attempt = state["shards"][0]["attempts"][0]
        attempt.update({
            "status": "completed",
            "batch_id": "batch-prune",
            "output_file_id": "file-prune",
            "local_output_path": output_name,
        })
        (self.run_dir / "batch_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        result = prune_batch_outputs(self.run_dir)
        self.assertEqual(result["files_deleted"], 1)
        self.assertEqual(result["rows_preserved_in_manifest"], 1)
        self.assertFalse(output_path.exists())
        manifest = json.loads(
            (self.run_dir / "batch_output_prune_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["aggregate"]["usage"]["output_image_tokens"], 20)
        refreshed = load_state(self.run_dir)
        self.assertTrue(refreshed["shards"][0]["attempts"][0]["local_output_pruned"])
        self.assertEqual(refreshed["output_pruning"]["status"], "completed")
        self.assertTrue(prune_batch_outputs(self.run_dir)["already_pruned"])

        client = FakeClient()
        client.batches.objects["batch-prune"] = SimpleNamespace(
            id="batch-prune", status="completed", output_file_id="file-prune",
            error_file_id=None,
            request_counts=SimpleNamespace(completed=1, failed=0, total=1),
            errors=None,
        )
        collected = collect_results(self.run_dir, client, prepare_review=False)
        self.assertEqual(collected, {"success": 10, "total": 10, "missing": 0})
        self.assertEqual(client.files.content_calls, 0)


class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = create_plan(CONFIG, "validation", "validation-review", self.root, 5)
        state = load_state(self.run_dir)
        self.records = []
        with (self.run_dir / state["plan_path"]).open(encoding="utf-8") as fh:
            self.records = [json.loads(line) for line in fh]
        # Two identical files exercise duplicate detection; missing files remain explicit in QA.
        for record in [self.records[0], self.records[3]]:
            width, height = (int(value) for value in record["size"].split("x"))
            Image.new("RGB", (width, height), (10, 20, 30)).save(
                self.run_dir / "images" / record["filename"], "JPEG", quality=92
            )

    def tearDown(self):
        self.temp.cleanup()

    def _fill_review(self, matches: int):
        path = self.run_dir / "human_review.csv"
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
            fields = list(rows[0])
        for index, row in enumerate(rows):
            row.update({
                "photorealism": "pass", "intent_match": "match" if index < matches else "wrong",
                "framing": "pass", "roll_no_back": "pass", "body_integrity": "pass",
                "notes": "",
            })
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_artifacts_and_explicit_approval_gate(self):
        artifacts = prepare_review_artifacts(self.run_dir, use_detector=False)
        self.assertTrue((self.run_dir / "contact_sheet.jpg").exists())
        self.assertTrue((self.run_dir / "human_review.csv").exists())
        self.assertEqual(artifacts["valid_images"], 2)
        self.assertEqual(artifacts["duplicates"], 1)

        self._fill_review(matches=6)
        # Detector execution is intentionally skipped in this unit test; provide a
        # complete mocked hard-QA result before exercising the approval gate.
        qa_path = self.run_dir / "auto_qa.jsonl"
        qa_rows = [json.loads(line) for line in qa_path.read_text().splitlines()]
        for row in qa_rows:
            row["quality_gate_complete"] = True
            row["quality_gate_pass"] = True
            row["quality_gate_reasons"] = []
        qa_path.write_text("".join(json.dumps(row) + "\n" for row in qa_rows))
        approval = approve_review(self.run_dir, "human-reviewer")
        self.assertEqual(approval["intent_match"], 6)
        pilot = create_plan(CONFIG, "pilot", "pilot-approved", self.root, 9, self.run_dir)
        self.assertEqual(len(load_state(pilot)["items"]), 500)
        self.assertEqual(len(load_state(pilot)["shards"]), 1)

        # Any post-approval CSV edit invalidates the stage gate.
        with (self.run_dir / "human_review.csv").open("a", encoding="utf-8") as fh:
            fh.write("\n")
        with self.assertRaises(PipelineError):
            submit_pending(pilot, FakeClient())
        with self.assertRaises(PipelineError):
            create_plan(CONFIG, "pilot", "pilot-tampered", self.root, 9, self.run_dir)

    def test_incremental_qa_runs_detector_only_for_replaced_ids(self):
        initial = run_auto_qa(self.run_dir, use_detector=False)
        self.assertEqual(initial[3]["duplicate_of"], self.records[0]["custom_id"])

        target = self.records[0]
        width, height = (int(value) for value in target["size"].split("x"))
        Image.new("RGB", (width, height), (90, 80, 70)).save(
            self.run_dir / "images" / target["filename"], "JPEG", quality=92
        )
        evaluated: list[str] = []

        def fake_detector(_run_dir, records, _model_path, _score_threshold):
            evaluated.extend(record["custom_id"] for record in records)
            return {
                target["custom_id"]: {
                    "detector_status": "no_head", "head_count": 0,
                    "body_count": 0, "direction": None,
                }
            }

        with patch(
            "hrffa.dataset.qa.gpt_head_review._detector_annotations",
            side_effect=fake_detector,
        ):
            refreshed = run_auto_qa(
                self.run_dir, use_detector=True, custom_ids={target["custom_id"]}
            )

        self.assertEqual(evaluated, [target["custom_id"]])
        by_id = {row["custom_id"]: row for row in refreshed}
        self.assertEqual(by_id[target["custom_id"]]["detector_status"], "no_head")
        self.assertIsNone(by_id[self.records[3]["custom_id"]]["duplicate_of"])

    def test_overview_is_nine_square_sources_in_a_square_grid(self):
        run_dir = self.root / "overview-nine"
        images_dir = run_dir / "images"
        images_dir.mkdir(parents=True)
        bins = [
            "pitch_down_mid", "pitch_down_extreme", "pitch_up_mid",
            "pitch_up_extreme", "camera_high", "camera_low",
            "combined_pitch_yaw", "yaw_extreme",
        ]
        records = []
        for index in range(80):
            bin_id = bins[index // 10]
            custom_id = f"overview-{index:03d}"
            record = {
                "serial": index + 1,
                "custom_id": custom_id,
                "filename": custom_id + ".jpg",
                "bin": bin_id,
                "pitch": index - 40,
                "yaw": 0,
                "cam": 0,
                "size": "1024x1024",
            }
            records.append(record)
            Image.new("RGB", (32, 32), (index, 40, 60)).save(
                images_dir / record["filename"], "JPEG"
            )
        (run_dir / "generation_plan.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        (run_dir / "batch_state.json").write_text(json.dumps({
            "plan_path": "generation_plan.jsonl",
        }) + "\n", encoding="utf-8")

        result = regenerate_overview_contact_sheet(run_dir)
        self.assertEqual(result["images"], 9)
        self.assertEqual(result["dimensions"], "900x900")
        self.assertEqual(set(result["bins"]), set(bins))
        self.assertEqual(set(result["source_sizes"]), {"1024x1024"})

    def test_fewer_than_six_matches_forbids_approval_and_pilot(self):
        prepare_review_artifacts(self.run_dir, use_detector=False)
        self._fill_review(matches=5)
        with self.assertRaises(ReviewError):
            approve_review(self.run_dir, "human-reviewer")
        with self.assertRaises(PipelineError):
            create_plan(CONFIG, "pilot", "pilot-forbidden", self.root, 9, self.run_dir)


if __name__ == "__main__":
    unittest.main()
