#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# --------------------- モデル構築（ImageNet pretrained → FCを330に差し替え） ---------------------
def build_model_sushi330() -> Tuple[nn.Module, EfficientNet_B0_Weights]:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 330)
    nn.init.normal_(model.classifier[1].weight, 0, 0.01)
    nn.init.zeros_(model.classifier[1].bias)
    model.eval()
    return model, weights

# --------------------- 画像前処理 ---------------------
def preprocess_image(path: Path, weights) -> torch.Tensor:
    tfm = weights.transforms()
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)

# --------------------- 中間出力のフック ---------------------
def register_captures(model: nn.Module, captures: Dict):
    stage_c_to_name = {16:"Stage1",24:"Stage2",40:"Stage3",80:"Stage4",112:"Stage5",192:"Stage6",320:"Stage7"}
    def stem_hook(_, __, out): captures["Stem"] = out.detach()
    model.features[0].register_forward_hook(stem_hook)
    def stage_hook(_, __, out):
        c = int(out.shape[1])
        if c in stage_c_to_name:
            captures[stage_c_to_name[c]] = out.detach()
    for m in model.features[1:]:
        m.register_forward_hook(stage_hook)
    def head_hook(_, __, out): captures["HeadConv1x1"] = out.detach()
    model.features[-1].register_forward_hook(head_hook)
    def gap_hook(_, __, out): captures["GAP_2D"] = out.detach()
    model.avgpool.register_forward_hook(gap_hook)

def summarize(name: str, t: torch.Tensor):
    t = t.detach().cpu().float()
    return dict(name=name, shape=list(t.shape),
                min=float(t.min()), max=float(t.max()),
                mean=float(t.mean()), std=float(t.std(unbiased=False)))

# --------------------- チェックポイント選択ロジック ---------------------
def _parse_metrics_from_name(p: Path) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    s = p.stem  # 例: epoch=0100-acc=0.876-loss=0.45.ckpt
    e = re.search(r"(?:epoch[=_-])(\d+)", s)
    a = re.search(r"(?:acc|accuracy)[=_-]([0-9.]+)", s, re.I)
    l = re.search(r"(?:loss)[=_-]([0-9.]+)", s, re.I)
    epoch = int(e.group(1)) if e else None
    acc   = float(a.group(1)) if a else None
    loss  = float(l.group(1)) if l else None
    return epoch, acc, loss

def pick_checkpoint(ckpt_dir: Path, select: str, epoch: Optional[int]) -> Optional[Path]:
    files = [p for p in ckpt_dir.glob("**/*") if p.suffix.lower() in [".pt",".pth",".ckpt"]]
    if not files:
        return None
    # latest（更新時刻で最大）
    if select == "latest":
        return max(files, key=lambda p: p.stat().st_mtime)
    # epoch 指定
    if select == "epoch" and epoch is not None:
        # まずファイル名から epoch を抽出して一致を探す
        candidates = []
        for p in files:
            e, _, _ = _parse_metrics_from_name(p)
            if e == epoch:
                candidates.append(p)
        if candidates:
            return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
        # ダメなら “*{epoch}*” を含むものの最新を返す
        fuzzy = [p for p in files if str(epoch) in p.stem]
        if fuzzy:
            return sorted(fuzzy, key=lambda p: p.stat().st_mtime)[-1]
        return None
    # best_acc / best_loss
    scored: List[Tuple[Path,float]] = []
    if select in ["best_acc","best_loss"]:
        for p in files:
            _, acc, loss = _parse_metrics_from_name(p)
            if select == "best_acc" and acc is not None:
                scored.append((p, acc))          # 大きいほど良い
            if select == "best_loss" and loss is not None:
                scored.append((p, -loss))        # 小さいほど良い → マイナスで最大化
        if scored:
            scored.sort(key=lambda x: x[1])  # 一応安定ソート
            return scored[-1][0]
        # メタ情報がない時は latest
        return max(files, key=lambda p: p.stat().st_mtime)
    # デフォルトは latest
    return max(files, key=lambda p: p.stat().st_mtime)

# --------------------- state_dict 読み込み（多形式対応） ---------------------
def load_state_into_model(model: nn.Module, ckpt_path: Path, device: torch.device):
    sd = torch.load(ckpt_path, map_location=device)
    # 代表的なラッパーを吸収
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
        sd = sd["model_state_dict"]
    # EMA があれば優先
    if isinstance(sd, dict) and "ema_state_dict" in sd and isinstance(sd["ema_state_dict"], dict):
        sd = sd["ema_state_dict"]
    # DataParallel/Lightning の prefix 剥がし
    new_sd = {}
    for k, v in sd.items():
        k2 = k
        for prefix in ["module.", "model.", "net."]:
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]
        # 例: classifier.fc → classifier.1 などのズレを適宜補正したい場合はここにマッピングを書く
        new_sd[k2] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    return missing, unexpected

# --------------------- メイン ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True, help="入力画像（寿司写真）")
    ap.add_argument("--outdir", type=Path, default=Path("./trace_out"))
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    # チェックポイント指定
    ap.add_argument("--ckpt", type=Path, default=None, help="直接パス指定 (.pt/.pth/.ckpt)")
    ap.add_argument("--ckpt_dir", type=Path, default=None, help="ディレクトリから自動選択")
    ap.add_argument("--select", type=str, default="latest",
                    choices=["latest","epoch","best_acc","best_loss"],
                    help="ckpt_dir 使用時の選択基準")
    ap.add_argument("--epoch", type=int, default=None, help="--select epoch の場合のターゲットエポック")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # モデル & 画像
    model, weights = build_model_sushi330()
    model.to(device)
    x = preprocess_image(args.image, weights).to(device)

    # チェックポイント決定
    ckpt_to_use = None
    if args.ckpt and args.ckpt.exists():
        ckpt_to_use = args.ckpt
    elif args.ckpt_dir and args.ckpt_dir.exists():
        ckpt_to_use = pick_checkpoint(args.ckpt_dir, args.select, args.epoch)

    if ckpt_to_use is not None:
        print(f"[*] loading checkpoint: {ckpt_to_use}")
        missing, unexpected = load_state_into_model(model, ckpt_to_use, device)
        print("    missing:", missing[:6], " ...", f"({len(missing)} keys)") if missing else print("    missing: []")
        print("    unexpected:", unexpected[:6], " ...", f"({len(unexpected)} keys)") if unexpected else print("    unexpected: []")
    else:
        print("[!] チェックポイント未指定/未発見：final FC(330) は未学習のままです。")

    # キャプチャ用フック
    captures: Dict[str, torch.Tensor] = {}
    register_captures(model, captures)

    # 推論
    with torch.no_grad():
        logits = model(x)
    captures["Logits"] = logits.detach()
    if "GAP_2D" in captures:
        captures["GAP"] = captures["GAP_2D"].flatten(1)
        del captures["GAP_2D"]

    # 保存
    order = ["Stem","Stage1","Stage2","Stage3","Stage4","Stage5","Stage6","Stage7","HeadConv1x1","GAP","Logits"]
    rows = [summarize(k, captures[k]) for k in order if k in captures]
    pd.DataFrame(rows).to_csv(args.outdir/"intermediate_shapes_stats.csv", index=False)

    if "HeadConv1x1" in captures:
        head_mean = captures["HeadConv1x1"].mean(dim=(2,3)).squeeze(0).cpu().numpy()
        pd.DataFrame({"channel_mean_after_head": head_mean}).to_csv(args.outdir/"head_conv_channel_means.csv", index=False)
    if "GAP" in captures:
        pd.DataFrame({"gap": captures["GAP"].squeeze(0).cpu().numpy()}).to_csv(args.outdir/"gap_vector_1280.csv", index=False)
    pd.DataFrame({"logits": captures["Logits"].squeeze(0).cpu().numpy()}).to_csv(args.outdir/"logits_330.csv", index=False)

    # プレビュー
    preview = {
        "ckpt": str(ckpt_to_use) if ckpt_to_use else None,
        "shapes": {k: list(v.shape) for k, v in captures.items()},
        "gap_first16": captures["GAP"].squeeze(0).cpu().numpy()[:16].round(6).tolist() if "GAP" in captures else None,
        "logits_first16": captures["Logits"].squeeze(0).cpu().numpy()[:16].round(6).tolist(),
    }
    (args.outdir/"preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2))
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(f"Saved CSVs under: {args.outdir.resolve()}")

if __name__ == "__main__":
    main()
