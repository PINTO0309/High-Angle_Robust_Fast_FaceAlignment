#!/usr/bin/env bash
# Phase A(history/026 §3)を実行計画順に自動連続実行する(96GB 端末想定)。
#
# 使い方:
#   nohup bash scripts/run_phase_a.sh > runs/phase_a_runner.log 2>&1 &
#   bash scripts/run_phase_a.sh --dry-run                       # コマンド列を表示するだけ
#   ARMS="abl_s256_gtonly_96gb abl_s256_noin_96gb" bash scripts/run_phase_a.sh   # 一部のみ
#
# 挙動:
#   - 既定順序: A1 → A1 の 2 周目 r2(026 §3 のアーム A0 は中断、A2〜A6 は棄却。ARMS=... で個別実行は可能)
#   - runs/<preset>/PHASE_A_DONE があるアームはスキップ(冪等。中断後は再実行で続きから)
#   - runs/<preset>/<preset>_last.pt があれば --resume で再開
#   - 学習成功後に best ckpt を 4 計器(official / stratify-real / pose-stress / style-shift)で
#     評価し、ONNX(256、dynamic)を書き出す。評価・export の失敗は記録するがチェーンは止めない
#   - 学習が失敗したアームは記録して次へ進む(再実行すれば resume される)。末尾に要約を出す
set -u
cd "$(dirname "$0")/.."
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
# 2026-08-27 ユーザー決定(043 §7-§8): A0 は中断(A1 より明らかに遅い → KD 採用確定)、A2〜A6 は棄却。
# 既定列は A1(WSD、ゼロから)→ r2(旧 A1 best init の WSD)。045 の派生アームは ARMS=... で明示(2026-08-28)
DEFAULT_ARMS="student_s256_96gb student_s256_96gb_r2"
ARMS="${ARMS:-$DEFAULT_ARMS}"
STRESS_N="${STRESS_N:-300}"
STYLE_N="${STYLE_N:-300}"
failed=""

ev() {  # ev <preset> <best> <tag> <evaluate flags...>
  local p=$1 best=$2 tag=$3; shift 3
  local out="runs/$p"
  echo "+ uv run python -m hrffa.train.evaluate --ckpt $best --preset $p --use-ema $* > $out/eval_best_$tag.log"
  [ $DRY = 1 ] && return 0
  uv run python -m hrffa.train.evaluate --ckpt "$best" --preset "$p" --use-ema "$@" \
    > "$out/eval_best_$tag.log" 2> "$out/eval_best_$tag.err" || echo "!! $p: eval $tag failed (rc=$?)"
}

for p in $ARMS; do
  out="runs/$p"
  if [ -f "$out/PHASE_A_DONE" ]; then
    echo "## $p: PHASE_A_DONE exists -> skip"
    continue
  fi
  echo "## $p: training $(date '+%F %T')"
  resume_arg=""
  [ -f "$out/${p}_last.pt" ] && resume_arg="--resume $out/${p}_last.pt"
  echo "+ uv run python -m hrffa.train.distill_student --preset $p $resume_arg >> $out/train_$p.out"
  if [ $DRY = 0 ]; then
    mkdir -p "$out"
    uv run python -m hrffa.train.distill_student --preset "$p" $resume_arg >> "$out/train_$p.out" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "!! $p: training failed rc=$rc (re-run to resume)"
      failed="$failed $p"
      continue
    fi
  fi
  best=$(ls -1 "$out"/${p}_best_*.pt 2>/dev/null | sort | tail -n 1)
  if [ -z "$best" ]; then
    if [ $DRY = 1 ]; then best="$out/${p}_best_<epoch>_<nme>.pt"; else
      echo "!! $p: no best checkpoint"; failed="$failed $p"; continue; fi
  fi
  echo "## $p: evaluation $(date '+%F %T') best=$best"
  ev "$p" "$best" official  --official
  ev "$p" "$best" stratreal --stratify-real
  ev "$p" "$best" stress    --pose-stress --stress-n "$STRESS_N"
  ev "$p" "$best" style     --style-shift --style-n "$STYLE_N"
  echo "+ uv run python -m hrffa.export.export_onnx --ckpt $best --preset $p --dynamic --output $out/$p.onnx > $out/export.log"
  if [ $DRY = 0 ]; then
    uv run python -m hrffa.export.export_onnx --ckpt "$best" --preset "$p" --dynamic \
      --output "$out/$p.onnx" > "$out/export.log" 2>&1 || echo "!! $p: export failed (rc=$?; measure size/latency manually)"
    date '+%F %T' > "$out/PHASE_A_DONE"
  fi
  echo "## $p: done $(date '+%F %T')"
done
echo "=== Phase A runner finished $(date '+%F %T') failed:${failed:- none} ==="
