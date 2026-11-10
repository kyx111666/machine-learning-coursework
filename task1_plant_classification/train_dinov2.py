"""Five-fold DINOv2-Base partial finetuning and fixed ConvNeXt probability blending."""
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from data import (
    CLASSES, CROP_CONFIG, competition_score, foreground_box,
    load_inputs, validate_probabilities, write_submission, parse_paths, find_artifact,
)

CONFIG = {
    "revision": "dinov2-supervised-fp32",
    "precision": "float32",
    "hub_repo": "facebookresearch/dinov2:7764ea0",
    "model": "dinov2_vitb14_reg",
    "seed": 2026, "image_size": 392, "unfreeze_blocks": 4,
    "epochs": 12, "head_only_epochs": 2,
    "batch_size": 2, "accumulation": 8, "workers": 2,
    "backbone_lr": 1e-5, "head_lr": 3e-4, "weight_decay": 0.01,
    "label_smoothing": 0.05, "ema_decay": 0.95,
    "new_weight": 0.5,
    "tta": ["full", "full_flip", "crop", "crop_flip"],
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_torch():
    # Keep CPU input audits usable without installing PyTorch.
    global torch, nn, transforms, DataLoader, checkpoint
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.checkpoint import checkpoint
    from torchvision import transforms


def make_model(backbone):
    class PlantDINO(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head_only = False
            if getattr(backbone, "chunked_blocks", False):
                raise ValueError("Expected unchunked DINO blocks; use block_chunks=0")
            self.cut = len(backbone.blocks) - CONFIG["unfreeze_blocks"]
            if self.cut < 0:
                raise ValueError("Not enough transformer blocks")
            for param in backbone.parameters():
                param.requires_grad_(False)
            for block in backbone.blocks[self.cut:]:
                block.requires_grad_(True)
            backbone.norm.requires_grad_(True)
            self.head = nn.Sequential(nn.LayerNorm(backbone.embed_dim * 2),
                                      nn.Linear(backbone.embed_dim * 2, 5))

        def train(self, mode=True):
            super().train(mode)
            self.backbone.eval()
            for block in self.backbone.blocks[self.cut:]:
                block.train(mode)
            return self

        def forward(self, images):
            # The frozen prefix cannot accumulate gradients or optimizer state.
            with torch.no_grad():
                tokens = self.backbone.prepare_tokens_with_masks(images)
                for block in self.backbone.blocks[:self.cut]:
                    tokens = block(tokens)
            # During head warmup do not backpropagate through the backbone at all.
            with torch.set_grad_enabled(torch.is_grad_enabled() and not self.head_only):
                for block in self.backbone.blocks[self.cut:]:
                    if self.training and not self.head_only:
                        tokens = checkpoint(block, tokens, use_reentrant=False)
                    else:
                        tokens = block(tokens)
                tokens = self.backbone.norm(tokens)
                patches = tokens[:, 1 + self.backbone.num_register_tokens:]
                features = torch.cat([tokens[:, 0], patches.mean(dim=1)], dim=1)
            return self.head(features)

    return PlantDINO()


class PlantImages:
    def __init__(self, files, labels=None, training=False, view="full", size=None):
        self.images, self.boxes = [], []
        self.labels, self.training, self.view = labels, training, view
        size = CONFIG["image_size"] if size is None else size
        for path in files:
            with Image.open(path) as im:
                image = im.convert("RGB").copy()
            self.images.append(image)
            self.boxes.append(foreground_box(image)[0])
        self.color = transforms.ColorJitter(0.10, 0.10, 0.10, 0.02)
        self.tensor = transforms.Compose([
            transforms.Resize((size, size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        crop = random.random() < 0.5 if self.training else self.view.startswith("crop")
        if crop:
            image = image.crop(self.boxes[index])
        side = max(image.size)
        left, top = (side-image.width)//2, (side-image.height)//2
        image = ImageOps.expand(image, (left, top, side-image.width-left,
                                       side-image.height-top), fill=0)
        if self.training:
            image = image.rotate(90 * random.randrange(4))
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            image = self.color(image)
        elif self.view.endswith("flip"):
            image = ImageOps.mirror(image)
        tensor = self.tensor(image)
        return (tensor, int(self.labels[index])) if self.labels is not None else tensor


def trainable_state(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()
            if p.requires_grad}


def restore_parameters(model, values):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in values.items():
            params[name].copy_(value)


def train_fold(model, dataset, device, state_path, fingerprint, fold):
    """Receives training labels only; no holdout labels or early stopping."""
    backbone = [p for name, p in model.named_parameters()
                if p.requires_grad and name.startswith("backbone.")]
    head = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": CONFIG["backbone_lr"]},
        {"params": head, "lr": CONFIG["head_lr"]},
    ], weight_decay=CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    ema = trainable_state(model)
    start, logs = 0, []
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=True)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint protocol mismatch; use a new output directory")
        restore_parameters(model, state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
        ema, start, logs = state["ema"], state["epoch"], state["logs"]
        print(f"Fold {fold+1}: restored epoch {start}", flush=True)
    for epoch in range(start, CONFIG["epochs"]):
        # Reset the epoch seed so an epoch-boundary resume has the same shuffle.
        seed_everything(CONFIG["seed"] + fold*1000 + epoch)
        loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                            num_workers=CONFIG["workers"], pin_memory=device.type == "cuda")
        phase = max(0, epoch - CONFIG["head_only_epochs"])
        factor = 0.1 + 0.9 * (1 + math.cos(math.pi * phase /
                 max(1, CONFIG["epochs"]-CONFIG["head_only_epochs"]-1))) / 2
        optimizer.param_groups[0]["lr"] = (0 if epoch < CONFIG["head_only_epochs"]
                                           else CONFIG["backbone_lr"] * factor)
        optimizer.param_groups[1]["lr"] = CONFIG["head_lr"] * factor
        model.train()
        model.head_only = epoch < CONFIG["head_only_epochs"]
        print(f"Fold {fold+1} epoch {epoch+1}: FP32, "
              f"{'head warmup' if model.head_only else 'last-block finetuning'}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        begun, total_loss, total = time.monotonic(), 0.0, 0
        accum = CONFIG["accumulation"]
        for batch, (images, labels) in enumerate(loader):
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
            group_start = (batch // accum) * accum
            group_size = min(accum, len(loader)-group_start)
            # T4 FP16 overflow must not interrupt training before GradScaler can
            # recover. This stability revision uses FP32 without loss scaling.
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"FP32 loss is non-finite: fold={fold+1}, epoch={epoch+1}, batch={batch+1}; "
                    "no submission produced")
            (loss / group_size).backward()
            if (batch+1) % accum == 0 or batch+1 == len(loader):
                norm = nn.utils.clip_grad_norm_(backbone + head, 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError(
                        f"FP32 gradient is non-finite: fold={fold+1}, epoch={epoch+1}, batch={batch+1}; "
                        "optimizer not updated")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in ema:
                            ema[name].lerp_(p.detach(), 1-CONFIG["ema_decay"])
            total_loss += float(loss.detach()) * len(labels)
            total += len(labels)
            if (batch+1) % 25 == 0:
                print(f"Fold {fold+1} epoch {epoch+1}: batch {batch+1}/{len(loader)}", flush=True)
        entry = {"fold": fold, "epoch": epoch+1, "precision": "float32", "train_loss": total_loss/total,
                 "seconds": time.monotonic()-begun}
        logs.append(entry)
        print(json.dumps(entry), flush=True)
        state = {"fingerprint": fingerprint, "epoch": epoch+1,
                 "parameters": trainable_state(model), "ema": ema,
                 "optimizer": optimizer.state_dict(),
                 "logs": logs}
        temporary = state_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(state_path)
    restore_parameters(model, ema)
    return logs


def predict(model, files, device):
    model.eval()
    # Reuse decoded images across views; never use labels here.
    dataset = PlantImages(files)
    averaged = np.zeros((len(files), 5), dtype=np.float64)
    for view in CONFIG["tta"]:
        dataset.view = view
        loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=CONFIG["workers"], pin_memory=device.type == "cuda")
        cursor = 0
        with torch.inference_mode():
            for images in loader:
                probabilities = model(images.to(device, dtype=torch.float32)).softmax(1).cpu().numpy()
                averaged[cursor:cursor+len(images)] += probabilities / len(CONFIG["tta"])
                cursor += len(images)
        print(f"Inference {view}: {cursor}/{len(files)}", flush=True)
    validate_probabilities(averaged, len(files), "DINOv2 inference")
    return averaged.astype(np.float32)


def write_results(bundle, template, oof, test, output):
    labels, folds = bundle["labels"], bundle["fold_id"]
    validate_probabilities(oof, len(labels), "DINOv2 OOF")
    validate_probabilities(test, 378, "DINOv2 test")
    weight = CONFIG["new_weight"]
    candidate_oof = (1-weight)*bundle["convnext_oof"] + weight*oof
    candidate_test = (1-weight)*bundle["convnext_test"] + weight*test
    metrics = {"convnext_accuracy": competition_score(labels, bundle["convnext_oof"]),
               "dinov2_accuracy": competition_score(labels, oof),
               "blend_accuracy": competition_score(labels, candidate_oof),
               "blend_fold_accuracy": [competition_score(labels[folds == i], candidate_oof[folds == i])
                                       for i in range(5)],
               "validation_note": "Repeated-use folds provide exploratory validation."}
    (output / "cv_metrics.json").write_text(json.dumps(metrics, indent=2))
    write_submission(template, bundle["test_names"], candidate_test,
                     output / "submission_cv_blend.csv")
    np.savez_compressed(output / "dinov2_cv_probabilities.npz",
                        ids=bundle["test_names"], train_names=bundle["train_names"],
                        labels=labels, fold_id=folds, dinov2_oof=oof, dinov2_test=test,
                        candidate_oof=candidate_oof, candidate_test=candidate_test)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main(argv=None):
    args = parse_paths(__doc__, argv)
    root, output = args.data, args.out
    bundle_path = find_artifact("convnext_probabilities.npz", output, root)
    bundle, train_files, test_files, template = load_inputs(bundle_path, root)
    print("878 张图片哈希、类别、ID、原五折边界与缓存校验通过。", flush=True)
    print(f"运行修订版：{CONFIG['revision']}；训练和推理均使用 FP32。", flush=True)
    (output / "dinov2_cv_config.json").write_text(json.dumps(CONFIG, indent=2))
    if os.environ.get("TASK1_AUDIT_ONLY") == "1":
        print("仅数据审计；没有运行训练，也没有生成候选提交。")
        return
    load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("请开启 Kaggle GPU 和 Internet，再运行全部单元")
    # PyTorch's native attention avoids dependence on a compatible xFormers wheel.
    os.environ["XFORMERS_DISABLED"] = "1"
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    fingerprint = hashlib.sha256(bundle_path.read_bytes() + json.dumps(
        {"config": CONFIG, "crop": CROP_CONFIG, "torch": str(torch.__version__)},
        sort_keys=True).encode()).hexdigest()
    state_dir = output / "dinov2_cv_state"
    state_dir.mkdir(exist_ok=True)
    y, folds = bundle["labels"], bundle["fold_id"]
    oof = np.full((len(y), 5), np.nan, dtype=np.float32)
    test_folds = []
    for fold in range(5):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        cache = state_dir / f"fold{fold+1}_predictions.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved["fingerprint"]) != fingerprint or not np.array_equal(saved["valid"], valid):
                    raise ValueError("Fold cache does not match this experiment")
                vp, tp = saved["valid_prob"].copy(), saved["test_prob"].copy()
            validate_probabilities(vp, len(valid), "cached validation")
            validate_probabilities(tp, len(test_files), "cached test")
            print(f"Fold {fold+1}: reuse completed predictions", flush=True)
        else:
            seed_everything(CONFIG["seed"] + fold)
            backbone = torch.hub.load(CONFIG["hub_repo"], CONFIG["model"],
                                      pretrained=True, trust_repo=True, block_chunks=0)
            model = make_model(backbone).to(device=device, dtype=torch.float32)
            dataset = PlantImages([train_files[i] for i in train], y[train], training=True)
            logs = train_fold(model, dataset, device, state_dir / f"fold{fold+1}.pt", fingerprint, fold)
            # Holdout labels are first accessed after fixed-duration training has ended.
            vp = predict(model, [train_files[i] for i in valid], device)
            tp = predict(model, test_files, device)
            temp = cache.with_suffix(".tmp")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, fingerprint=fingerprint, valid=valid,
                                    valid_prob=vp, test_prob=tp)
            temp.replace(cache)
            pd.DataFrame(logs).to_csv(state_dir / f"fold{fold+1}_training.csv", index=False)
            del model, backbone, dataset
            gc.collect()
            torch.cuda.empty_cache()
        oof[valid] = vp
        test_folds.append(tp)
        print(f"Fold {fold+1} completed: Micro-F1={competition_score(y[valid], vp):.4f}", flush=True)
    write_results(bundle, template, oof, np.mean(test_folds, axis=0), output)
    (output / "dinov2_cv_provenance.json").write_text(json.dumps({
        "fingerprint": fingerprint, "torch": str(torch.__version__),
        "pretrained_source": CONFIG["hub_repo"], "test_labels_used": False,
        "fixed_training_epochs": CONFIG["epochs"], "validation_selected_epochs": False,
    }, indent=2))


if __name__ == "__main__":
    main()
