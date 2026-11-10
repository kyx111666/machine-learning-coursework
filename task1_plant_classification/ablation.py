"""Fixed-epoch five-fold ablations using competition training images.

Run --list to inspect the completed experiment recipes. The held-out folds
are reused across experiments, so their scores are exploratory evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"]
FOLD_SEED = 2026
N_FOLDS = 5
N_VIEWS = 8  # 4 个 90° 旋转 × 是否水平翻转

BASE = dict(
    model="convnext_tiny.fb_in22k_ft_in1k",
    pretrained=True,
    strategy="full",          # full / head / partial / scratch
    img_size=320,
    aug=("flip", "rot90", "color", "crop"),
    epochs=12,
    warmup_epochs=1,
    batch_size=16,
    lr=1e-4,                  # 主干学习率；分类头 = lr × head_lr_mult
    head_lr_mult=10.0,
    weight_decay=0.05,
    label_smoothing=0.05,
    seed=2026,
    model_kwargs={},
    partial_prefixes=("stages.3", "head", "norm_pre"),  # strategy=partial 时解冻的参数前缀（ConvNeXt）
    amp=True,                 # CUDA 上用 FP16 混合精度；DINOv2 用 FP32（避免 FP16 溢出）
)

# ViT 部分微调：只解冻最后 4 个 block + 最终 norm
VIT_LAST4 = ("blocks.8.", "blocks.9.", "blocks.10.", "blocks.11.", "norm.")

# 每个实验 = (组别, 相对 BASE 的改动)。组别用于报告分表。
EXPERIMENTS: dict[str, tuple[str, dict]] = {
    "base": ("基线", {}),
    # A. 数据增强（逐项累加，最后一项 = base）
    "aug_none": ("增强", dict(aug=())),
    "aug_flip": ("增强", dict(aug=("flip",))),
    "aug_flip_rot": ("增强", dict(aug=("flip", "rot90"))),
    "aug_flip_rot_color": ("增强", dict(aug=("flip", "rot90", "color"))),
    # B. 分辨率
    "res_224": ("分辨率", dict(img_size=224)),
    "res_448": ("分辨率", dict(img_size=448, batch_size=8)),
    # C. 训练方式与预训练
    "strat_scratch": ("训练方式", dict(strategy="scratch", pretrained=False, lr=1e-3, head_lr_mult=1.0)),
    "strat_head": ("训练方式", dict(strategy="head")),
    "strat_partial": ("训练方式", dict(strategy="partial")),
    "pretrain_in1k": ("训练方式", dict(model="convnext_tiny.fb_in1k")),
    # D. 超参数
    "lr_3e-5": ("学习率", dict(lr=3e-5)),
    "lr_3e-4": ("学习率", dict(lr=3e-4)),
    "ep_6": ("轮数", dict(epochs=6)),
    "ep_24": ("轮数", dict(epochs=24)),
    "ls_0": ("标签平滑", dict(label_smoothing=0.0)),
    "ls_0.1": ("标签平滑", dict(label_smoothing=0.1)),
    # E. 模型对比（统一 224，与 res_224 的 ConvNeXt-T 对照）
    "m_resnet50": ("模型", dict(model="resnet50.a1_in1k", img_size=224)),
    "m_effnet_b2": ("模型", dict(model="efficientnet_b2.ra_in1k", img_size=224)),
    "m_convnext_s": ("模型", dict(model="convnext_small.fb_in22k_ft_in1k", img_size=224)),
    "m_swin_t": ("模型", dict(model="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", img_size=224)),
    "m_dinov2_s": ("模型", dict(model="vit_small_patch14_dinov2.lvd142m", img_size=224,
                               model_kwargs={"img_size": 224})),
    # G. 随机种子方差（与 base 合计 3 个种子）
    "seed_3407": ("种子", dict(seed=3407)),
    "seed_9187": ("种子", dict(seed=9187)),
    # F. 各模型按自己的常用配方调优（每个模型只跑一次，配方事先定死，不在 OOF 上搜索）。
    #    E 组统一用 ConvNeXt 的配方（lr 1e-4），对 ResNet/EfficientNet 太小、对 ViT 太大。
    "t_resnet50": ("调优模型", dict(model="resnet50.a1_in1k", img_size=224, lr=1e-3, head_lr_mult=1.0)),
    "t_effnet_b2": ("调优模型", dict(model="efficientnet_b2.ra_in1k", img_size=224, lr=1e-3, head_lr_mult=1.0)),
    "t_dinov2_s": ("调优模型", dict(model="vit_small_patch14_dinov2.lvd142m", img_size=224,
                                    model_kwargs={"img_size": 224}, lr=2e-5, head_lr_mult=15.0,
                                    weight_decay=0.01, amp=False)),
    # Supplementary timm DINO configuration; differs from the final training entry point.
    "dinov2_base_partial": ("调优模型", dict(model="vit_base_patch14_reg4_dinov2.lvd142m", img_size=392,
                                         model_kwargs={"img_size": 392}, strategy="partial",
                                         partial_prefixes=VIT_LAST4, lr=1e-5, head_lr_mult=30.0,
                                         weight_decay=0.01, batch_size=8, amp=False)),
}


def config_for(name: str) -> dict:
    cfg = dict(BASE)
    cfg.update(EXPERIMENTS[name][1])
    cfg["name"], cfg["group"] = name, EXPERIMENTS[name][0]
    return cfg


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 数据


def list_data(root: Path):
    """Sort training images by class/name and test images by filename."""
    from data import find_data
    files, test_files, template = find_data(root)
    return files, np.repeat(np.arange(5), 100), test_files, template


def fold_ids(labels: np.ndarray) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold
    fid = np.zeros(len(labels), dtype=int)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        fid[va] = k
    return fid


_CACHE: dict = {}


def load_images(files: list[Path], size: int):
    """补成正方形后缩放到 size，缓存在内存（原图最大 2840px，逐轮解码太慢）。"""
    from PIL import Image
    key = (tuple(str(f) for f in files), size)
    if key not in _CACHE:
        out = []
        for f in files:
            with Image.open(f) as im:
                im = im.convert("RGB")
                w, h = im.size
                side = max(w, h)
                canvas = Image.new("RGB", (side, side))
                canvas.paste(im, ((side - w) // 2, (side - h) // 2))
                out.append(canvas.resize((size, size), Image.BICUBIC))
        _CACHE[key] = out
    return _CACHE[key]


def norm_stats(cfg: dict):
    """按预训练权重取归一化参数（EVA02 用 CLIP 的均值方差，其余为 ImageNet）。"""
    import timm
    c = timm.models.get_pretrained_cfg(cfg["model"])
    return tuple(c.mean), tuple(c.std)


def make_transforms(cfg: dict):
    import torch
    from torchvision import transforms as T
    mean, std = norm_stats(cfg)
    aug, size = set(cfg["aug"]), cfg["img_size"]
    unknown = aug - {"flip", "rot90", "color", "crop"}
    if unknown:
        raise ValueError(f"未知增强：{unknown}")
    steps = []
    if "crop" in aug:
        steps.append(T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25),
                                         interpolation=T.InterpolationMode.BICUBIC))
    if "flip" in aug:
        steps += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    if "color" in aug:
        steps.append(T.ColorJitter(0.2, 0.2, 0.2, 0.02))
    steps += [T.ToTensor(), T.Normalize(mean, std)]
    if "rot90" in aug:
        # 俯拍幼苗对 90° 旋转不变；整数倍旋转不引入插值和黑边。
        steps.append(T.Lambda(lambda x: torch.rot90(x, random.randint(0, 3), dims=(1, 2))))
    train_tf = T.Compose(steps)
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return train_tf, eval_tf


def make_dataset(images, labels, tf):
    from torch.utils.data import Dataset

    class DS(Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, i):
            x = tf(images[i])
            return (x, int(labels[i])) if labels is not None else (x, -1)

    return DS()


# ---------------------------------------------------------------- 模型
def build_model(cfg: dict):
    import timm
    m = timm.create_model(cfg["model"], pretrained=cfg["pretrained"], num_classes=len(CLASSES),
                          **cfg["model_kwargs"])
    head_ids = {id(p) for p in m.get_classifier().parameters()}
    strategy = cfg["strategy"]
    for n, p in m.named_parameters():
        if id(p) in head_ids or strategy in ("full", "scratch"):
            p.requires_grad = True
        elif strategy == "head":
            p.requires_grad = False
        elif strategy == "partial":
            # 只解冻最后几层（ConvNeXt 默认 stages.3；ViT 见 VIT_LAST4）。
            p.requires_grad = n.startswith(tuple(cfg["partial_prefixes"]))
        else:
            raise ValueError(strategy)
    head = [p for p in m.parameters() if id(p) in head_ids]
    body = [p for p in m.parameters() if id(p) not in head_ids and p.requires_grad]
    return m, body, head


def tta_views(x):
    """8 个视图：rot90^k（k=0..3）× 是否水平翻转。第 0 个视图为原图。"""
    import torch
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        yield r
        yield torch.flip(r, dims=(3,))


def predict(model, loader, device, amp, views: int):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            vs = []
            for v, xv in enumerate(tta_views(x)):
                if v >= views:
                    break
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    vs.append(torch.softmax(model(xv).float(), 1))
            out.append(torch.stack(vs, 1).cpu().numpy())
    return np.concatenate(out)  # (N, views, C)


# ---------------------------------------------------------------- 训练
def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_fold(cfg, k, files, labels, fid, test_files, device, out_dir, with_test, workers):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    set_seed(cfg["seed"] * 10 + k)
    size = cfg["img_size"]
    imgs = load_images(files, size)
    tr, va = np.where(fid != k)[0], np.where(fid == k)[0]
    train_tf, eval_tf = make_transforms(cfg)
    g = torch.Generator().manual_seed(cfg["seed"] * 10 + k)
    tl = DataLoader(make_dataset([imgs[i] for i in tr], labels[tr], train_tf),
                    batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                    num_workers=workers, generator=g, persistent_workers=workers > 0)
    vl = DataLoader(make_dataset([imgs[i] for i in va], labels[va], eval_tf),
                    batch_size=cfg["batch_size"] * 2, num_workers=0)

    model, body, head = build_model(cfg)
    model.to(device)
    lr = cfg["lr"]
    groups = [{"params": head, "lr": lr * cfg["head_lr_mult"]}]
    if body:
        groups.append({"params": body, "lr": lr})
    for gr in groups:
        gr["base_lr"] = gr["lr"]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    amp = device.type == "cuda" and cfg["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    steps_per_epoch = len(tl)
    total, warm = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch

    hist, step, t0 = [], 0, time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in tl:
            # 线性预热 + 余弦退火，按步更新。
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for gr in opt.param_groups:
                gr["lr"] = gr["base_lr"] * f
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * len(y)
            n += len(y)
            step += 1
        p = predict(model, vl, device, amp, views=1)[:, 0]
        vloss = float(-np.log(np.clip(p[np.arange(len(va)), labels[va]], 1e-8, 1)).mean())
        vacc = float((p.argmax(1) == labels[va]).mean())
        hist.append(dict(epoch=ep + 1, train_loss=loss_sum / n, val_loss=vloss, val_acc=vacc))
        log(f"  {cfg['name']} fold{k} ep{ep + 1:02d} train_loss={loss_sum / n:.4f} "
            f"val_loss={vloss:.4f} val_acc={vacc:.3f}")
    train_time = time.time() - t0

    val_views = predict(model, vl, device, amp, views=N_VIEWS)
    res = dict(val_idx=va, val_views=val_views.astype(np.float32),
               history=json.dumps(hist), train_time=train_time)
    if with_test:
        timgs = load_images(test_files, size)
        tl2 = DataLoader(make_dataset(timgs, None, eval_tf), batch_size=cfg["batch_size"] * 2)
        res["test_views"] = predict(model, tl2, device, amp, views=N_VIEWS).astype(np.float32)
    np.savez_compressed(out_dir / f"fold{k}.npz", **res)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    del model, opt
    return n_train, n_all


def run_experiment(name, files, labels, fid, test_files, device, out_root, with_test, workers):
    cfg = config_for(name)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    serialized = json.dumps(cfg, indent=2, ensure_ascii=False)
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(serialized):
        raise ValueError("Experiment configuration changed; choose a new --out directory")
    config_path.write_text(serialized)
    params = None
    for k in range(N_FOLDS):
        if (out / f"fold{k}.npz").exists():
            log(f"{name} fold{k} 已完成，跳过")
            continue
        log(f"=== {name} fold{k}")
        params = run_fold(cfg, k, files, labels, fid, test_files, device, out, with_test, workers)
    summarize(name, labels, out, params)


def summarize(name, labels, out, params=None):
    oof = np.zeros((len(labels), N_VIEWS, len(CLASSES)), dtype=np.float32)
    times, test = [], []
    for k in range(N_FOLDS):
        z = np.load(out / f"fold{k}.npz")
        oof[z["val_idx"]] = z["val_views"]
        times.append(float(z["train_time"]))
        if "test_views" in z.files:
            test.append(z["test_views"])
    acc1 = float((oof[:, 0].argmax(1) == labels).mean())
    acc8 = float((oof.mean(1).argmax(1) == labels).mean())
    np.savez_compressed(out / "oof.npz", oof_views=oof, labels=labels,
                        **({"test_views": np.stack(test)} if len(test) == N_FOLDS else {}))
    s = dict(name=name, oof_acc_single=acc1, oof_acc_tta8=acc8,
             errors_tta8=int((oof.mean(1).argmax(1) != labels).sum()),
             train_time_s=sum(times))
    if params:
        s["trainable_params"], s["total_params"] = params
    elif (out / "summary.json").exists():
        old = json.loads((out / "summary.json").read_text())
        s.update({k: old[k] for k in ("trainable_params", "total_params") if k in old})
    (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
    log(f"*** {name}: OOF 单视图 {acc1:.3f}  8 视图 TTA {acc8:.3f}  用时 {sum(times) / 60:.1f} 分钟")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--exp", nargs="*", default=None, help="实验名；缺省 = 全部")
    ap.add_argument("--no-test", action="store_true", help="不预测测试集")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：1 轮、五折、小图")
    a = ap.parse_args(argv)
    if a.list:
        for n, (g, d) in EXPERIMENTS.items():
            print(f"{n:22s} {g:6s} {d}")
        return 0
    import torch
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    log(f"device = {device}")
    files, labels, test_files, _ = list_data(a.data)
    fid = fold_ids(labels)
    names = a.exp or list(EXPERIMENTS)
    for n in names:
        if n not in EXPERIMENTS:
            raise SystemExit(f"未知实验：{n}")
    if a.smoke:
        BASE.update(epochs=1, warmup_epochs=1, img_size=64, batch_size=8)
        for n in names:
            EXPERIMENTS[n][1].update(img_size=64, batch_size=8, epochs=1)
            EXPERIMENTS[n][1].get("model_kwargs", {}).pop("img_size", None)
            if "dinov2" in EXPERIMENTS[n][1].get("model", ""):
                EXPERIMENTS[n][1]["model_kwargs"] = {"img_size": 70}
                EXPERIMENTS[n][1]["img_size"] = 70
            if "swin" in EXPERIMENTS[n][1].get("model", ""):
                EXPERIMENTS[n][1]["model_kwargs"] = {"img_size": 64, "window_size": 2}
    for n in names:
        run_experiment(n, files, labels, fid, test_files, device, a.out, not a.no_test, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
