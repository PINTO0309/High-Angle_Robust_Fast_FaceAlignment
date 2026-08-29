"""S1 学生蒸留 CLI(history/026)。

オンライン蒸留: 同一の拡張済みクロップを teacher_size(既定 320)でレンダリングし、
教師(凍結・bf16・no_grad)はそのまま、学生は out_size(既定 256)へ縮小して入力する。
正規化座標は解像度非依存のため GT・KD 目標は共有できる。

使い方:
    # 開発 PC(8GB)での動作確認
    uv run python -m hrffa.train.distill_student --preset smoke_s_8gb
    # 本学習(96GB 端末)
    uv run python -m hrffa.train.distill_student --preset student_s256_96gb

resume・EMA・best 命名・ログ・val プロットは train_teacher と同一の体系。
"""

from __future__ import annotations

import argparse
import copy
import glob as _glob
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from ..data.dataset import SchemeBatchSampler, SourceDataset, SourceSpec, collate
from ..dataset.augment.geometric import GeometricPolicy
from ..model.teacher import TeacherModel
from .config import get_config
from .train_teacher import (
    REAL_ALL_SPLITS,
    build_val_sets, compute_losses, lr_lambda_factory, restore_rng, rng_states,
)


def build_distill_loaders(cfg):
    """teacher_size でレンダリングする学習ローダ(構成は build_loaders と同一)。"""
    policy = GeometricPolicy(out_size=cfg.teacher_size, roll_mode=cfg.roll_mode,
                             pad=cfg.crop_pad, scale_range=tuple(cfg.scale_range),
                             translate=cfg.translate)
    dsets, weights = [], []
    for sw in cfg.sources:
        holdout = ("train" if sw.name.startswith(("300wlp", "synth_dwarp"))
                   else None)
        splits = ("train",)
        if cfg.train_all_splits and sw.name in REAL_ALL_SPLITS:
            splits = REAL_ALL_SPLITS[sw.name]   # 教師側(train_teacher)と同じ意味
        ds = SourceDataset(cfg.unified, SourceSpec(sw.name, sw.weight, splits, holdout=holdout,
                                                   yaw_max=cfg.source_yaw_max.get(sw.name)),
                           cfg.teacher_size, train=True, policy=policy,
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


def load_teacher(cfg, device):
    t_cfg = get_config(cfg.teacher_preset)
    teacher = TeacherModel.from_config(t_cfg)
    matches = sorted(_glob.glob(str(cfg.teacher_ckpt)))
    if not matches:
        raise FileNotFoundError(f"no file matches teacher_ckpt: {cfg.teacher_ckpt}")
    ck = torch.load(matches[-1], map_location="cpu", weights_only=False)
    teacher.load_state_dict(ck.get("ema") or ck["model"])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"teacher: {cfg.teacher_preset} <- {matches[-1]} (d_model={t_cfg.d_model})")
    return teacher, t_cfg


def kd_losses(s_out, t_out, adapter, cfg):
    """KD 損失 3 項。教師出力は float32 化して目標にする。"""
    out = {}
    if cfg.w_kd_coord > 0:
        out["kd_coord"] = F.smooth_l1_loss(
            s_out["points"], t_out["points"].float(), beta=0.01) * cfg.w_kd_coord
    if cfg.w_kd_vis > 0:
        T = cfg.kd_temp
        s = F.log_softmax(s_out["vis_logits"].flatten(0, 1) / T, dim=-1)
        t = F.softmax(t_out["vis_logits"].float().flatten(0, 1) / T, dim=-1)
        out["kd_vis"] = (F.kl_div(s, t, reduction="batchmean")
                         * T * T) * cfg.w_kd_vis
    if cfg.w_kd_tok > 0:
        out["kd_tok"] = F.smooth_l1_loss(
            adapter(s_out["dec_tokens"]), t_out["dec_tokens"].float(),
            beta=1.0) * cfg.w_kd_tok
    return out


def save_ckpt(path, model, adapter, ema, opt, sched, epoch, step_in_epoch, gstep,
              cfg, best_nme=float("inf"), scaler=None):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "adapter": adapter.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch, "step_in_epoch": step_in_epoch, "gstep": gstep,
                "best_nme": best_nme, "rng": rng_states(),
                "cfg": vars(cfg) | {
                    "sources": [(s.name, s.weight) for s in cfg.sources],
                    "unified": str(cfg.unified), "ckpt_out": str(cfg.ckpt_out)}},
               tmp)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Online distillation of a student from the frozen teacher (same augmented crops, teacher at teacher_size, student at out_size).")
    ap.add_argument("--preset", required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = get_config(args.preset)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    assert cfg.teacher_preset and cfg.teacher_ckpt, "teacher_preset/teacher_ckpt are required"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[cfg.amp_dtype]

    print(f"[{cfg.preset}] student={cfg.backbone}@{cfg.out_size} "
          f"(input_norm={cfg.input_norm}) teacher@{cfg.teacher_size} "
          f"batch={cfg.batch_size}x{cfg.grad_accum} "
          f"kd=(coord {cfg.w_kd_coord}, vis {cfg.w_kd_vis}, tok {cfg.w_kd_tok}) "
          f"device={device}")
    # 学習ローダは教師入力用に ImageNet 正規化でレンダリングし、学生入力は
    # ステップ内で学生の正規化仕様へ厳密変換する。val は学生仕様で直接レンダリング
    loader, sampler = build_distill_loaders(cfg)
    val_sets = build_val_sets(cfg)          # 学生解像度・学生 input_norm で評価
    teacher, t_cfg = load_teacher(cfg, device)
    model = TeacherModel.from_config(cfg).to(device)
    adapter = (nn.Identity() if cfg.d_model == t_cfg.d_model
               else nn.Linear(cfg.d_model, t_cfg.d_model)).to(device)
    if cfg.init_from and not args.resume:
        matches = sorted(_glob.glob(str(cfg.init_from)))
        if not matches:
            raise FileNotFoundError(f"no file matches init_from: {cfg.init_from}")
        ck_init = torch.load(matches[-1], map_location="cpu", weights_only=False)
        sd = ck_init.get("ema") or ck_init.get("model") or ck_init
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"initialized from {matches[-1]} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
    model.freeze_backbone(True)

    groups = model.param_groups(cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if not isinstance(adapter, nn.Identity):
        groups.append({"params": adapter.parameters(), "lr": cfg.lr_head,
                       "weight_decay": cfg.weight_decay})
    opt = torch.optim.AdamW(groups)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16
                                                   and device == "cuda"))
    total_steps = cfg.epochs * cfg.steps_per_epoch
    if cfg.sched_full_anneal:
        # 教師側と同一の扱い(grad_accum=1 の学生では無効果だが単位を揃えておく)
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
        adapter.load_state_dict(ck["adapter"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gstep = ck["gstep"]
        step_in_epoch = ck.get("step_in_epoch", cfg.steps_per_epoch)
        if step_in_epoch < cfg.steps_per_epoch:
            start_epoch, skip_steps = ck["epoch"], step_in_epoch
        else:
            start_epoch = ck["epoch"] + 1
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        if "rng" in ck:
            restore_rng(ck["rng"])
        resumed_best_nme = ck.get("best_nme", float("inf"))
        print(f"resumed from {args.resume} (epoch {start_epoch}, "
              f"skip {skip_steps}, gstep {gstep}, best {resumed_best_nme:.5f})")

    cfg.ckpt_out.mkdir(parents=True, exist_ok=True)
    log_path = cfg.ckpt_out / f"train_log_{cfg.preset}.jsonl"
    log_f = open(log_path, "a", encoding="utf-8")
    best_nme = resumed_best_nme if args.resume else float("inf")
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

    need_resize = cfg.out_size != cfg.teacher_size
    # 学生 input_norm への厳密変換(z_new = z_im · s_im/s_new + (m_im−m_new)/s_new)
    from ..model.backbone import IMAGE_MEAN, IMAGE_STD, norm_constants
    m_new, s_new = norm_constants(cfg.input_norm)
    need_renorm = cfg.input_norm != "imagenet"
    ren_a = torch.tensor([IMAGE_STD[c] / s_new[c] for c in range(3)],
                         device=device).view(1, 3, 1, 1)
    ren_b = torch.tensor([(IMAGE_MEAN[c] - m_new[c]) / s_new[c] for c in range(3)],
                         device=device).view(1, 3, 1, 1)
    for epoch in range(start_epoch, cfg.epochs):
        if epoch >= cfg.freeze_backbone_epochs:
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
                continue
            batch = {k: (v.to(device, non_blocking=True)
                         if torch.is_tensor(v) else v) for k, v in batch.items()}
            img_t = batch["image"]
            img_s = (F.interpolate(img_t, size=(cfg.out_size, cfg.out_size),
                                   mode="bilinear", align_corners=False,
                                   antialias=True)
                     if need_resize else img_t)
            if need_renorm:
                img_s = img_s * ren_a + ren_b
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=device == "cuda"):
                with torch.no_grad():
                    t_out = teacher(img_t, batch["scheme"])
                s_out = model(img_s, batch["scheme"])
                losses = compute_losses(s_out, batch, cfg)
                losses |= kd_losses(s_out, t_out, adapter, cfg)
                loss = sum(losses.values()) / cfg.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(adapter.parameters()),
                    cfg.grad_clip)
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
                              model, adapter, ema, opt, sched, epoch, i + 1,
                              gstep, cfg, best_nme, scaler)
            if (i + 1) % cfg.log_every == 0:
                loss_vals = {k: round(float(v.detach()), 4) for k, v in losses.items()}
                row = {"epoch": epoch, "step": i + 1, "gstep": gstep,
                       "scheme": batch["scheme"], "lr": sched.get_last_lr()[-1],
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
                old.unlink()
            plot_val_metrics(val_history,
                             cfg.ckpt_out / f"{cfg.preset}_val_metrics_e{epoch:04d}.png")
            if row["val_mean_nme"] < best_nme:
                best_nme = row["val_mean_nme"]
                for old in cfg.ckpt_out.glob(f"{cfg.preset}_best_*.pt"):
                    old.unlink()
                torch.save({"ema": ema.state_dict(), "epoch": epoch,
                            "gstep": gstep, "val": row["val"],
                            "val_mean_nme": best_nme},
                           cfg.ckpt_out /
                           f"{cfg.preset}_best_e{epoch:04d}_{best_nme:.6f}.pt")
                if cfg.val_render_n > 0:
                    from .evaluate import render_val_samples
                    render_val_samples(ema, val_sets, device,
                                       cfg.ckpt_out / f"{cfg.preset}_best_vis",
                                       cfg.val_render_n, draw_axes=False)
                tqdm.write(f"\033[92mnew best: val_mean_nme={best_nme:.5f} "
                           f"(renders -> {cfg.preset}_best_vis/)\033[0m")

        save_ckpt(cfg.ckpt_out / f"{cfg.preset}_last.pt",
                  model, adapter, ema, opt, sched, epoch, steps_done, gstep,
                  cfg, best_nme, scaler)
        if cfg.snapshot_every_epochs and (epoch + 1) % cfg.snapshot_every_epochs == 0:
            # 教師側と同じ EMA のみ形式(best と同形式・評価/init_from 用)
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
