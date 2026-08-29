"""教師学習の設定(YAML プリセット方式)。

- `TrainConfig` dataclass の既定値が共通ベース(= teacher_vitl_96gb 相当)
- `configs/<preset>.yaml` に**差分のみ**を記述する。`_base_: <name|path>` で
  他 YAML を継承可能(チェーン、後勝ちマージ)
- `get_config()` にはプリセット名(configs/ から解決)または YAML パスを渡せる
- 未知のキーはエラー(タイポ検出)。設定全体は checkpoint にも保存される

例(configs/abl_pose_off_96gb.yaml):
    _base_: abl_pose_base
    w_rot: 0.0
    w_roll: 0.0
    w_yaw_weak: 0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class SourceWeight:
    name: str
    weight: float


@dataclass
class TrainConfig:
    preset: str = "teacher_vitl_96gb"
    # data
    unified: Path = Path("datasets/unified")
    out_size: int = 256
    sources: list[SourceWeight] = field(default_factory=lambda: [
        SourceWeight("300wlp", 0.60),
        SourceWeight("synth_dwarp", 0.15),
        SourceWeight("wflw", 0.15),
        SourceWeight("300w", 0.06),
        SourceWeight("cofw", 0.04),
    ])
    roll_mode: str = "full360"
    erase_prob: float = 0.0      # random erasing(遮蔽拡張)の適用確率
    motion_blur_prob: float = 0.0  # 線形モーションブラー拡張の適用確率(0 = 無効)
    # クロップ幾何(history/030 pad アブレーション用に公開。既定は従来値)
    # 注意: crop_pad と scale_range/translate はエンベロープ収容条件
    # (0.5·scale_max/(1+2·pad) + translate ≤ 0.5)を満たすようセットで設定する
    crop_pad: float = 0.15
    scale_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    translate: float = 0.05
    num_workers: int = 16
    # model
    backbone: str = "vitl16"
    patch_instance_norm: bool = False  # patch embed 直後の token 軸 IN(history/017 F)
    # 入力正規化仕様: imagenet(教師系)| center05 = ((x/255)-0.5)/0.5(学生系。
    # history/026: vitt 事前学習重みへは conv 折り込みで無劣化変換される)
    input_norm: str = "imagenet"
    d_model: int = 256
    dec_layers: int = 4
    dec_heads: int = 8              # デコーダの注意ヘッド数(history/049 で軽量化用に公開)
    dec_ffn: int = 1024             # デコーダ FFN 幅
    # --- 軽量 CNN 学生(history/049)。backbone が hgnetv2_* のときのみ有効 ---
    feat_stride: int = 16           # メモリ格子の stride(CNN: 8 = stage1 と FPN 融合した 32×32、16 = stage2)
    cnn_feat_ch: int = 128          # CNN ネック出力チャネル(= backbone.embed_dim)
    # --- 学生アーキテクチャの派生アーム(history/045 D5〜D8。既定は全て無効 = 従来と同一)---
    feat_layers: list = field(default_factory=list)  # D5: 結合するブロック添字(例 [5, 8, 11])。空 = 最終層のみ
    local_conv: bool = False        # D7: 学生ブロックに zero-init の局所畳み込み分岐(LESA 由来)
    dec_local_iters: int = 0        # D6: デコーダ局所項による座標の反復精密化回数
    head: str = "regress"           # D8: "popos" = 距離マップ + multilateration デコード
    popos_topk: int = 6             # D8: multilateration のアンカー数
    w_dist: float = 1.0             # D8: 距離マップ損失の重み
    popos_radius: float = 6.0       # D8: L1 監督の半径(セル単位。外側は hinge)
    # optim
    batch_size: int = 96
    grad_accum: int = 2
    steps_per_epoch: int = 1200
    epochs: int = 30
    freeze_backbone_epochs: int = 1
    lr_backbone: float = 2e-5
    lr_head: float = 2e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    # cosine 全長を optimizer step 数(バッチ数/grad_accum)に合わせて終端 LR≈0 まで
    # 下げ切る。False は従来挙動 = grad_accum=2 の教師系は half-cosine で終端 0.5×ピーク
    # (030 §2。v1〜v8 の比較互換のため既定は False)
    sched_full_anneal: bool = False
    # LR スケジュール種別(history/044)。cosine = 従来(warmup → cosine)。
    # wsd = warmup → 一定 LR(ピーク)→ 末尾 decay_epochs で cosine 減衰 → 0。一定区間の LR は
    # epochs に依存しないため resume で epochs を書き換えて延長/短縮できる
    # (epochs = 現 epoch + decay_epochs にすると即 decay へ)。wsd は sched_full_anneal に依存せず
    # optimizer step 単位(epochs × steps_per_epoch // grad_accum)で長さを決める
    lr_schedule: str = "cosine"
    decay_epochs: int = 0
    # 実写ソース(wflw/300w/cofw)の全 split(テスト含む)を学習に投入する。
    # 評価側(テスト split 名・val セット)は無変更 = 公式数値は学習済みデータへの適合度になる
    # (ユーザー決定 2026-08-26: ベンチ比較は目的外、汎化はデータ量で取る)
    train_all_splits: bool = False
    # ソース別 |yaw| 上限(history/039)。例 {"300wlp": 20, "synth_dwarp": 20}。
    # 学習ローダと 300wlp_val の両方に適用(データファイルは無変更)
    source_yaw_max: dict = field(default_factory=dict)
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    amp_dtype: str = "bf16"
    # loss weights
    w_coord: float = 10.0
    w_vis: float = 1.0
    w_rot: float = 1.0
    w_roll: float = 0.5
    w_yaw_weak: float = 0.2
    vm_kappa: float = 2.0
    # io(ckpt_out 未指定時は get_config が runs/<preset> を設定する)
    ckpt_out: Path = Path("runs/teacher")
    save_every_steps: int = 300
    # N epoch ごとに EMA のみ(best と同形式)を非上書きで {preset}_snap_eXXXX.pt に
    # 保存(0=無効)。長期学習で「best と別基準の最良 epoch」を評価するための保険
    # (1 本 ~1.2GB。学習再開用の全状態は含まない)
    snapshot_every_epochs: int = 0
    log_every: int = 50
    eval_every_epochs: int = 1
    eval_n: int = 256
    val_render_n: int = 20
    seed: int = 0
    max_steps: int | None = None
    # 初期重み(パスまたは glob。glob は辞書順最後=最新 best を採用)。
    # --resume とは異なり optimizer/epoch は引き継がず新規学習として開始する
    init_from: str | None = None
    # --- S1 蒸留(distill_student.py 専用。history/026)---
    teacher_preset: str | None = None   # 教師のアーキ構築に使う preset 名
    teacher_ckpt: str | None = None     # 教師重み(パスまたは glob、末尾=最新)
    teacher_size: int = 320             # 蒸留時のレンダリング/教師入力解像度
    w_kd_coord: float = 0.0             # 教師予測座標への KD
    w_kd_vis: float = 0.0               # 可視性 logits の KL 蒸留
    w_kd_tok: float = 0.0               # dec_tokens 特徴蒸留
    kd_temp: float = 2.0                # 可視性 KD の温度


def _config_dir() -> Path:
    """configs/ の探索: カレント優先、次にリポジトリルート(editable 実行時)。"""
    for cand in (Path("configs"),
                 Path(__file__).resolve().parents[3] / "configs"):
        if cand.is_dir():
            return cand
    raise FileNotFoundError("configs/ directory not found")


def _load_chain(ref: str | Path, cfg_dir: Path, seen: set[str]) -> dict:
    path = Path(ref)
    if path.suffix != ".yaml":
        path = cfg_dir / f"{ref}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"preset yaml not found: {path}")
    key = str(path.resolve())
    if key in seen:
        raise ValueError(f"circular _base_ chain: {path}")
    seen.add(key)
    data = yaml.safe_load(path.read_text()) or {}
    base_ref = data.pop("_base_", None)
    merged = _load_chain(base_ref, cfg_dir, seen) if base_ref else {}
    merged.update(data)
    return merged


def get_config(preset: str) -> TrainConfig:
    """プリセット名(configs/<name>.yaml)または YAML パスから設定を構築する。"""
    cfg_dir = _config_dir()
    data = _load_chain(preset, cfg_dir, set())

    cfg = TrainConfig(preset=Path(preset).stem)
    valid = {f.name for f in fields(TrainConfig)}
    for k, v in data.items():
        if k not in valid:
            raise ValueError(f"unknown config key '{k}' (preset: {preset})")
        if k in ("unified", "ckpt_out"):
            v = Path(v)
        elif k == "sources":
            v = [SourceWeight(**d) for d in v]
        setattr(cfg, k, v)
    if cfg.feat_stride not in (8, 16) or (cfg.feat_stride == 8 and not cfg.backbone.startswith("hgnetv2")):
        raise ValueError(f"feat_stride must be 8 (CNN only) or 16 (preset: {preset}, {cfg.feat_stride}, backbone {cfg.backbone})")
    if cfg.head not in ("regress", "popos"):
        raise ValueError(f"head must be regress | popos (preset: {preset}, got {cfg.head!r})")
    if cfg.head == "popos" and cfg.dec_local_iters > 0:
        raise ValueError(f"head=popos cannot be combined with dec_local_iters>0 (preset: {preset})")
    if cfg.dec_local_iters < 0 or cfg.popos_topk < 3:
        raise ValueError(f"dec_local_iters must be >= 0 and popos_topk >= 3 (preset: {preset})")
    if cfg.lr_schedule not in ("cosine", "wsd"):
        raise ValueError(f"lr_schedule must be cosine | wsd (preset: {preset}, got {cfg.lr_schedule!r})")
    if cfg.lr_schedule == "wsd" and not (0 <= cfg.decay_epochs <= cfg.epochs):
        raise ValueError(f"decay_epochs must be within 0..epochs (preset: {preset}, {cfg.decay_epochs}/{cfg.epochs})")
    # 出力先はプリセットごとに分離(YAML で ckpt_out を明示した場合のみそちらを優先)
    if "ckpt_out" not in data:
        cfg.ckpt_out = Path("runs") / cfg.preset
    return cfg
