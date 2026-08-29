"""教師モデル学習 CLI。

使い方:
    # 開発 PC(8GB)での動作確認のみ
    PYTHONPATH=src python3 -m hrffa.train.train_teacher --preset smoke_8gb

    # 本学習(96GB 端末)
    PYTHONPATH=src python3 -m hrffa.train.train_teacher --preset teacher_vitl_96gb

チェックポイントは ckpts/teacher/(gitignore)に保存。resume は --resume <path>。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import random

from ..data.dataset import SchemeBatchSampler, SourceDataset, SourceSpec, collate
from ..dataset.augment.geometric import GeometricPolicy
from ..model.losses import (
    coord_loss, geodesic_loss, roll_biternion_loss, roll_from_matrix,
    visibility_loss, von_mises_yaw_loss, yaw_from_matrix,
)
from ..model.teacher import TeacherModel
from .config import get_config


# 実写ソースの全 split(train_all_splits 用)
REAL_ALL_SPLITS = {
    "wflw": ("train", "test"),
    "300w": ("train", "valid_common", "valid_challenge"),
    "cofw": ("train", "test"),
}


def build_loaders(cfg):
    policy = GeometricPolicy(out_size=cfg.out_size, roll_mode=cfg.roll_mode,
                             pad=cfg.crop_pad, scale_range=tuple(cfg.scale_range),
                             translate=cfg.translate)
    dsets, weights = [], []
    for sw in cfg.sources:
        # 300wlp 系(合成含む)は約 2% を検証ホールドアウトとして学習から除外
        holdout = ("train" if sw.name.startswith(("300wlp", "synth_dwarp"))
                   else None)
        splits = ("train",)
        if cfg.train_all_splits and sw.name in REAL_ALL_SPLITS:
            splits = REAL_ALL_SPLITS[sw.name]   # テスト split も学習に投入
        spec = SourceSpec(sw.name, sw.weight, splits, holdout=holdout,
                          yaw_max=cfg.source_yaw_max.get(sw.name))
        ds = SourceDataset(cfg.unified, spec,
                           cfg.out_size, train=True, policy=policy,
                           erase_prob=cfg.erase_prob,
                           motion_blur_prob=cfg.motion_blur_prob)
        dsets.append(ds)
        weights.append(sw.weight)
        print(f"  source {sw.name}: {len(ds):,} recs (scheme={ds.scheme}, w={sw.weight})")
    concat = ConcatDataset(dsets)
    sampler = SchemeBatchSampler([len(d) for d in dsets], weights,
                                 cfg.batch_size, cfg.steps_per_epoch, cfg.seed)
    loader = DataLoader(concat, batch_sampler=sampler, num_workers=cfg.num_workers,
                        collate_fn=collate, pin_memory=True,
                        persistent_workers=cfg.num_workers > 0)
    return loader, sampler


def build_val_sets(cfg):
    """学習中バリデーション用の決定的データセット群。

    - 300wlp_val: 300wlp のホールドアウト(≒2%)。姿勢 GT を持つ唯一の検証データ
    - 300w valid_common / wflw test / cofw test: 標準ベンチ相当
    """
    policy = GeometricPolicy(out_size=cfg.out_size, pad=cfg.crop_pad)
    defs = [
        ("300wlp_val", SourceSpec("300wlp", 1.0, holdout="val",
                                  yaw_max=cfg.source_yaw_max.get("300wlp"))),
        ("300w_vc", SourceSpec("300w", 1.0, ("valid_common",))),
        ("wflw_test", SourceSpec("wflw", 1.0, ("test",))),
        ("cofw_test", SourceSpec("cofw", 1.0, ("test",))),
    ]
    out = []
    for name, spec in defs:
        ds = SourceDataset(cfg.unified, spec, cfg.out_size, train=False, policy=policy,
                           input_norm=cfg.input_norm)
        print(f"  val {name}: {len(ds):,} recs")
        out.append((name, ds))
    return out


def compute_losses(out, batch, cfg):
    losses = {
        "coord": coord_loss(out["points"], batch["points"], batch["vis"]) * cfg.w_coord,
        "vis": visibility_loss(out["vis_logits"].flatten(0, 1),
                               batch["vis"].flatten()) * cfg.w_vis,
    }
    if "dist_pred" in out and cfg.w_dist > 0:                  # D8 POPoS(history/045)
        from ..model.popos import distance_loss
        h, w = out["grid_hw"]
        losses["dist"] = distance_loss(out["dist_pred"].float(), batch["points"], batch["vis"],
                                       h, w, radius=cfg.popos_radius) * cfg.w_dist
    # 重み 0 の損失は計算自体をスキップ(アブレーション時に勾配経路を完全遮断)
    has_rot = batch["has_rot"]
    if has_rot.any() and cfg.w_rot > 0:
        R_gt = batch["rot"][has_rot]
        losses["rot"] = geodesic_loss(out["rot"][has_rot], R_gt).mean() * cfg.w_rot
        if cfg.w_roll > 0:
            losses["roll"] = roll_biternion_loss(
                out["roll_bit"][has_rot], roll_from_matrix(R_gt)).mean() * cfg.w_roll
    has_d8 = batch["has_dir8"]
    if has_d8.any() and cfg.w_yaw_weak > 0:
        yaw_pr = yaw_from_matrix(out["rot"][has_d8])
        losses["yaw_weak"] = von_mises_yaw_loss(
            yaw_pr, batch["yaw_weak"][has_d8], cfg.vm_kappa).mean() * cfg.w_yaw_weak
    return losses


def rng_states() -> dict:
    import numpy as np
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
            "python": random.getstate()}


def restore_rng(states: dict) -> None:
    import numpy as np
    torch.set_rng_state(states["torch"])
    if torch.cuda.is_available() and states["cuda"]:
        torch.cuda.set_rng_state_all(states["cuda"])
    np.random.set_state(states["numpy"])
    random.setstate(states["python"])


def save_ckpt(path, model, ema, opt, sched, epoch, step_in_epoch, gstep, cfg,
              best_nme=float("inf"), scaler=None):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch, "step_in_epoch": step_in_epoch, "gstep": gstep,
                "best_nme": best_nme,
                "rng": rng_states(),
                "cfg": vars(cfg) | {
                    "sources": [(s.name, s.weight) for s in cfg.sources],
                    "unified": str(cfg.unified), "ckpt_out": str(cfg.ckpt_out)}},
               tmp)
    tmp.replace(path)  # 書き込み途中クラッシュで壊れた ckpt を残さない


def lr_lambda_factory(cfg, total_steps):
    """LR 係数(optimizer step 単位。教師・学生で共有)。

    - cosine(既定): warmup → cosine。total_steps は呼び出し側の単位のまま
      (sched_full_anneal=False の教師は batch 単位 = half-cosine、030 §2 の互換)
    - wsd(history/044): warmup → 一定 1.0 → 末尾 decay_epochs を cosine で 0 へ。長さは cfg から
      optimizer step 単位で再計算し(epochs × steps_per_epoch // grad_accum)、sched_full_anneal に
      依存しない。一定区間の LR は epochs に依存しないため、resume 時に epochs を書き換えて
      延長/短縮できる(epochs = 現 epoch + decay_epochs で即 decay に入る)
    """
    if getattr(cfg, "lr_schedule", "cosine") == "wsd":
        ga = max(cfg.grad_accum, 1)
        opt_total = cfg.epochs * cfg.steps_per_epoch // ga
        decay_steps = cfg.decay_epochs * cfg.steps_per_epoch // ga
        decay_start = max(opt_total - decay_steps, cfg.warmup_steps)

        def fn_wsd(step):
            if step < cfg.warmup_steps:
                return step / max(cfg.warmup_steps, 1)
            if decay_steps <= 0 or step < decay_start:
                return 1.0
            t = (step - decay_start) / max(opt_total - decay_start, 1)
            return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        return fn_wsd

    def fn(step):
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        t = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return fn


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the teacher model (DINOv3 backbone + point-query decoder).")
    ap.add_argument("--preset", default="smoke_8gb")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = get_config(args.preset)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[cfg.amp_dtype]

    print(f"[{cfg.preset}] backbone={cfg.backbone} batch={cfg.batch_size}"
          f"x{cfg.grad_accum} device={device}")
    loader, sampler = build_loaders(cfg)
    val_sets = build_val_sets(cfg)
    model = TeacherModel.from_config(cfg).to(device)
    if cfg.init_from and not args.resume:
        import glob as _glob
        matches = sorted(_glob.glob(str(cfg.init_from)))
        if not matches:
            raise FileNotFoundError(f"no file matches init_from: {cfg.init_from}")
        ck_init = torch.load(matches[-1], map_location="cpu", weights_only=False)
        sd = ck_init.get("ema") or ck_init.get("model") or ck_init
        try:
            model.load_state_dict(sd)
        except RuntimeError:
            # 新設モジュール(patch_in 等)は初期値のまま許容する
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"init_from strict=False: missing={list(missing)} "
                  f"unexpected={list(unexpected)}")
        print(f"initialized from {matches[-1]}")
    model.freeze_backbone(True)

    opt = torch.optim.AdamW(model.param_groups(cfg.lr_backbone, cfg.lr_head,
                                               cfg.weight_decay))
    # fp16 選択時のみ損失スケーリングを有効化(bf16 は不要)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16
                                                   and device == "cuda"))
    total_steps = cfg.epochs * cfg.steps_per_epoch
    if cfg.sched_full_anneal:
        # sched.step() は optimizer step ごとのため、全長も同じ単位に揃えて下げ切る
        total_steps //= max(cfg.grad_accum, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(cfg, total_steps))
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad = False

    start_epoch, skip_steps, gstep = 0, 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gstep = ck["gstep"]
        step_in_epoch = ck.get("step_in_epoch", cfg.steps_per_epoch)
        if step_in_epoch < cfg.steps_per_epoch:  # エポック途中で中断していた
            start_epoch, skip_steps = ck["epoch"], step_in_epoch
        else:
            start_epoch = ck["epoch"] + 1
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        if "rng" in ck:
            restore_rng(ck["rng"])
        resumed_best_nme = ck.get("best_nme", float("inf"))
        print(f"resumed from {args.resume} "
              f"(epoch {start_epoch}, skip {skip_steps} steps, gstep {gstep}, "
              f"best_nme {resumed_best_nme:.5f})")

    cfg.ckpt_out.mkdir(parents=True, exist_ok=True)
    log_path = cfg.ckpt_out / f"train_log_{cfg.preset}.jsonl"
    log_f = open(log_path, "a", encoding="utf-8")
    best_nme = resumed_best_nme if args.resume else float("inf")
    # 検証履歴(プロット用)。resume 時は既存ログから復元する
    val_history: list[dict] = []
    if args.resume and log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "val" in r:
                    val_history.append(r)

    for epoch in range(start_epoch, cfg.epochs):
        if epoch >= cfg.freeze_backbone_epochs and cfg.preset != "smoke_8gb":
            model.freeze_backbone(False)
        sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        steps_done = 0
        pbar = tqdm(loader, total=cfg.steps_per_epoch, dynamic_ncols=True,
                    desc=f"epoch {epoch}/{cfg.epochs - 1}", unit="step")
        for i, batch in enumerate(pbar):
            steps_done = i + 1
            if epoch == start_epoch and i < skip_steps:
                continue  # 中断位置まで読み飛ばし(サンプラは seed+epoch で決定的)
            batch = {k: (v.to(device, non_blocking=True)
                         if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=device == "cuda"):
                out = model(batch["image"], batch["scheme"])
                losses = compute_losses(out, batch, cfg)
                loss = sum(losses.values()) / cfg.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                gstep += 1
                with torch.no_grad():
                    d = cfg.ema_decay
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
                if gstep % max(cfg.save_every_steps, 1) == 0:
                    save_ckpt(cfg.ckpt_out / f"{cfg.preset}_last.pt",
                              model, ema, opt, sched, epoch, i + 1, gstep, cfg,
                              best_nme, scaler)
            if (i + 1) % cfg.log_every == 0:
                loss_vals = {k: round(float(v.detach()), 4) for k, v in losses.items()}
                row = {"epoch": epoch, "step": i + 1, "gstep": gstep,
                       "scheme": batch["scheme"],
                       "lr": sched.get_last_lr()[-1],
                       "sec_per_step": round((time.time() - t0) / (i + 1), 3),
                       **loss_vals}
                pbar.set_postfix(loss_vals, refresh=False)
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
            if cfg.max_steps and gstep >= cfg.max_steps:
                break
        train_sec = time.time() - t0
        if (epoch + 1) % cfg.eval_every_epochs == 0:
            from .evaluate import eval_nme
            t_eval = time.time()
            ema.eval()
            nmes = []
            row = {"epoch": epoch, "gstep": gstep, "val": {}}
            for name, ds in val_sets:
                res = eval_nme(ema, ds, device, cfg.eval_n)
                row["val"][name] = {k: round(v, 5) for k, v in res.items()}
                nmes.append(res["head_nme"])
            row["val_mean_nme"] = round(sum(nmes) / len(nmes), 5)
            row["epoch_train_sec"] = round(train_sec, 1)
            row["eval_sec"] = round(time.time() - t_eval, 1)
            val_history.append(row)
            tqdm.write(json.dumps(row))
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            from .plots import plot_val_metrics
            for old in cfg.ckpt_out.glob(f"{cfg.preset}_val_metrics*.png"):
                old.unlink()  # 最新のみ保持
            plot_val_metrics(val_history,
                             cfg.ckpt_out / f"{cfg.preset}_val_metrics_e{epoch:04d}.png")
            if row["val_mean_nme"] < best_nme:
                best_nme = row["val_mean_nme"]
                for old in cfg.ckpt_out.glob(f"{cfg.preset}_best_*.pt"):
                    old.unlink()  # 最新 best のみ保持
                torch.save({"ema": ema.state_dict(), "epoch": epoch,
                            "gstep": gstep, "val": row["val"],
                            "val_mean_nme": best_nme},
                           cfg.ckpt_out /
                           f"{cfg.preset}_best_e{epoch:04d}_{best_nme:.6f}.pt")
                if cfg.val_render_n > 0:
                    from .evaluate import render_val_samples
                    render_val_samples(ema, val_sets, device,
                                       cfg.ckpt_out / f"{cfg.preset}_best_vis",
                                       cfg.val_render_n,
                                       draw_axes=cfg.w_rot > 0)
                tqdm.write(f"\033[92mnew best: val_mean_nme={best_nme:.5f} "
                           f"(renders -> {cfg.preset}_best_vis/)\033[0m")

        # eval 後に保存することで best_nme 込みの状態が last に残る
        save_ckpt(cfg.ckpt_out / f"{cfg.preset}_last.pt",
                  model, ema, opt, sched, epoch, steps_done, gstep, cfg, best_nme,
                  scaler)
        if cfg.snapshot_every_epochs and (epoch + 1) % cfg.snapshot_every_epochs == 0:
            # best と同一の EMA のみ形式(~1.2GB/本)。評価・init_from に使える。
            # 学習再開には使えない(必要時は決定性による再現で代替: history/026)
            snap = cfg.ckpt_out / f"{cfg.preset}_snap_e{epoch:04d}.pt"
            torch.save({"ema": ema.state_dict(), "epoch": epoch, "gstep": gstep},
                       snap)
            tqdm.write(f"snapshot saved: {snap.name}")
        if cfg.max_steps and gstep >= cfg.max_steps:
            break
    log_f.close()
    print("done")


if __name__ == "__main__":
    main()
