"""Three-seed full-data DINOv2 refit and fixed final probability blend."""
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

import train_dinov2 as dino
from data import parse_paths, find_artifact


CONFIG = {
    "revision": "dinov2-full-data-three-seed-fp32",
    "seeds": [2026, 3407, 9187],
    "full_data_weight": 0.50,
    "epochs": 12,
    "head_only_epochs": 2,
    "precision": "float32",
}


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    dino.torch.manual_seed(seed)
    if dino.torch.cuda.is_available():
        dino.torch.cuda.manual_seed_all(seed)


def _train_full(model, dataset, device, state_path, fingerprint, seed):
    """Fixed-duration training on all 500 labels; no validation feedback."""
    torch, nn, DataLoader = dino.torch, dino.nn, dino.DataLoader
    backbone = [p for name, p in model.named_parameters()
                if p.requires_grad and name.startswith("backbone.")]
    head = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": dino.CONFIG["backbone_lr"]},
        {"params": head, "lr": dino.CONFIG["head_lr"]},
    ], weight_decay=dino.CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=dino.CONFIG["label_smoothing"])
    ema = dino.trainable_state(model)
    start, logs = 0, []
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=True)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint protocol mismatch; use a new output directory")
        dino.restore_parameters(model, state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
        ema, start, logs = state["ema"], state["epoch"], state["logs"]
        print(f"Seed {seed}: restored epoch {start}", flush=True)
    for epoch in range(start, CONFIG["epochs"]):
        _seed(seed + epoch)
        loader = DataLoader(dataset, batch_size=dino.CONFIG["batch_size"], shuffle=True,
                            num_workers=dino.CONFIG["workers"],
                            pin_memory=device.type == "cuda")
        phase = max(0, epoch - CONFIG["head_only_epochs"])
        factor = 0.1 + 0.9 * (1 + math.cos(math.pi * phase /
                 max(1, CONFIG["epochs"] - CONFIG["head_only_epochs"] - 1))) / 2
        optimizer.param_groups[0]["lr"] = (0 if epoch < CONFIG["head_only_epochs"]
                                           else dino.CONFIG["backbone_lr"] * factor)
        optimizer.param_groups[1]["lr"] = dino.CONFIG["head_lr"] * factor
        model.train()
        model.head_only = epoch < CONFIG["head_only_epochs"]
        phase_name = "head warmup" if model.head_only else "last-block finetuning"
        print(f"Seed {seed} epoch {epoch+1}: FP32, {phase_name}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        begun, total_loss, total = time.monotonic(), 0.0, 0
        accum = dino.CONFIG["accumulation"]
        for batch, (images, labels) in enumerate(loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)
            group_start = (batch // accum) * accum
            group_size = min(accum, len(loader) - group_start)
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"FP32 loss is non-finite: seed={seed}, epoch={epoch+1}, "
                    f"batch={batch+1}; no submission produced")
            (loss / group_size).backward()
            if (batch + 1) % accum == 0 or batch + 1 == len(loader):
                norm = nn.utils.clip_grad_norm_(backbone + head, 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError(
                        f"FP32 gradient is non-finite: seed={seed}, epoch={epoch+1}, "
                        f"batch={batch+1}; optimizer not updated")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name in ema:
                            ema[name].lerp_(parameter.detach(), 1-dino.CONFIG["ema_decay"])
            total_loss += float(loss.detach()) * len(labels)
            total += len(labels)
            if (batch + 1) % 25 == 0:
                print(f"Seed {seed} epoch {epoch+1}: batch {batch+1}/{len(loader)}", flush=True)
        entry = {"seed": seed, "epoch": epoch+1, "precision": "float32",
                 "train_loss": total_loss/total, "seconds": time.monotonic()-begun}
        logs.append(entry)
        print(json.dumps(entry), flush=True)
        state = {"fingerprint": fingerprint, "epoch": epoch+1,
                 "parameters": dino.trainable_state(model), "ema": ema,
                 "optimizer": optimizer.state_dict(), "logs": logs}
        temporary = state_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(state_path)
    dino.restore_parameters(model, ema)
    return logs


def load_cv_probabilities(path, bundle):
    with np.load(path, allow_pickle=False) as saved:
        required = {"ids", "train_names", "labels", "fold_id", "candidate_test"}
        if not required.issubset(saved.files):
            raise ValueError(f"dinov2_cv_probabilities.npz missing {sorted(required-set(saved.files))}")
        ids = saved["ids"].copy()
        train_names = saved["train_names"].copy()
        labels = saved["labels"].copy()
        folds = saved["fold_id"].copy()
        candidate = saved["candidate_test"].copy()
    if (not np.array_equal(ids, bundle["test_names"])
            or not np.array_equal(train_names, bundle["train_names"])
            or not np.array_equal(labels, bundle["labels"])
            or not np.array_equal(folds, bundle["fold_id"])):
        raise ValueError("CV blend probability IDs or labels do not match the audited competition bundle")
    dino.validate_probabilities(candidate, 378, "CV blend candidate")
    return candidate


def write_results(bundle, template, seed_probabilities, cv_test, output):
    seed_probabilities = np.asarray(seed_probabilities, dtype=np.float32)
    if seed_probabilities.shape != (len(CONFIG["seeds"]), 378, 5):
        raise ValueError("Expected one 378x5 probability matrix per configured seed")
    dino.validate_probabilities(cv_test, 378, "CV blend")
    for seed, probabilities in zip(CONFIG["seeds"], seed_probabilities):
        dino.validate_probabilities(probabilities, 378, f"seed {seed}")
    full_mean = seed_probabilities.mean(0)
    full_mean /= full_mean.sum(1, keepdims=True)
    weight = CONFIG["full_data_weight"]
    candidate = (1-weight)*cv_test + weight*full_mean
    candidate /= candidate.sum(1, keepdims=True)
    dino.write_submission(template, bundle["test_names"], candidate, output / "submission.csv")
    np.savez_compressed(output / "ensemble_probabilities.npz",
                        ids=bundle["test_names"], seeds=np.asarray(CONFIG["seeds"]),
                        seed_test=seed_probabilities, full_mean_test=full_mean,
                        cv_test=cv_test, candidate_test=candidate)
    print(f"Saved {output / 'submission.csv'}", flush=True)
    return candidate


def main(argv=None):
    args = parse_paths(__doc__, argv)
    root, output = args.data, args.out
    baseline_path = find_artifact("convnext_probabilities.npz", output, root)
    cv_path = find_artifact("dinov2_cv_probabilities.npz", output, root)
    bundle, train_files, test_files, template = dino.load_inputs(baseline_path, root)
    cv_test = load_cv_probabilities(cv_path, bundle)
    print("Competition inputs and CV predictions validated.", flush=True)
    (output / "ensemble_config.json").write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2))
    if os.environ.get("TASK1_AUDIT_ONLY") == "1":
        print("仅输入审计；没有训练或生成候选提交。")
        return
    dino.load_torch()
    if not dino.torch.cuda.is_available():
        raise RuntimeError("请开启 Kaggle GPU 和 Internet，再运行全部单元")
    os.environ["XFORMERS_DISABLED"] = "1"
    dino.torch.backends.cudnn.benchmark = False
    device = dino.torch.device("cuda")
    fingerprint_base = hashlib.sha256(
        baseline_path.read_bytes() + cv_path.read_bytes()
        + json.dumps({"ensemble": CONFIG, "dinov2_train": dino.CONFIG}, sort_keys=True).encode()
    ).hexdigest()
    state_dir = output / "ensemble_state"
    state_dir.mkdir(exist_ok=True)
    dataset = dino.PlantImages(train_files, bundle["labels"], training=True)
    seed_probabilities = []
    for seed in CONFIG["seeds"]:
        fingerprint = hashlib.sha256(f"{fingerprint_base}:{seed}".encode()).hexdigest()
        cache = state_dir / f"seed_{seed}_test.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved["fingerprint"]) != fingerprint:
                    raise ValueError("Seed cache does not match this experiment")
                probabilities = saved["test_prob"].copy()
            dino.validate_probabilities(probabilities, 378, f"cached seed {seed}")
            print(f"Seed {seed}: reuse completed predictions", flush=True)
        else:
            _seed(seed)
            backbone = dino.torch.hub.load(dino.CONFIG["hub_repo"], dino.CONFIG["model"],
                                          pretrained=True, trust_repo=True,
                                          block_chunks=0)
            model = dino.make_model(backbone).to(device=device, dtype=dino.torch.float32)
            logs = _train_full(model, dataset, device,
                               state_dir / f"seed_{seed}.pt", fingerprint, seed)
            probabilities = dino.predict(model, test_files, device)
            temporary = cache.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, fingerprint=fingerprint,
                                    test_prob=probabilities)
            temporary.replace(cache)
            pd.DataFrame(logs).to_csv(state_dir / f"seed_{seed}_training.csv", index=False)
            del model, backbone
            gc.collect()
            dino.torch.cuda.empty_cache()
        seed_probabilities.append(probabilities)
    write_results(bundle, template, seed_probabilities, cv_test, output)
    (output / "ensemble_provenance.json").write_text(json.dumps({
        "experiment_fingerprint": fingerprint_base,
        "pretrained_source": dino.CONFIG["hub_repo"],
        "seeds": CONFIG["seeds"], "test_labels_used": False,
        "fixed_training_epochs": CONFIG["epochs"],
        "validation_selected_epochs": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
