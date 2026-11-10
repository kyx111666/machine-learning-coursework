"""Independent-seed training with explicit successful-update accounting and completed-run checks."""
import copy
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import runtime as rt

TRAIN_STEPS = 4000
FORMAL_BATCH = 12

def completed_training(out, signature, steps):
    """Verify actual successful optimizer updates, not a budget multiplied by model count."""
    out = Path(out)
    meta = json.loads((out/'completed.json').read_text())
    history = pd.read_csv(out/'history.csv')
    ck = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
    assert meta['signature'] == signature == ck['signature'], '训练签名不匹配。'
    assert signature == hashlib.sha256(json.dumps(ck['spec'], sort_keys=True).encode()).hexdigest()
    assert meta['optimizer_updates'] == meta['requested_updates'] == steps == ck['spec']['steps'], \
        '实际更新不足；禁止把短训练算作正式预算。'
    assert len(history) >= 2 and history.step.iloc[0] == 0 and history.step.iloc[-1] == steps
    assert history.step.is_monotonic_increasing and not history.step.duplicated().any()
    assert history.attempted_batches.iloc[-1] == meta['attempted_batches']
    assert meta['attempted_batches'] == steps+meta['skipped_updates']
    assert np.isfinite(history[['loss','proxy_dice','lr']]).all().all()
    assert meta['best_step'] == ck['step'] and ck['step'] in history.step.values
    assert ck['recipe'] == meta['recipe'] == ck['spec']['recipe'] and ck['fold'] == meta['fold']
    assert abs(meta['best_proxy_dice']-history.proxy_dice.max()) < 1e-10
    assert abs(meta['best_proxy_dice']-float(history.set_index('step').loc[ck['step'], 'proxy_dice'])) < 1e-10
    return out/'best.pt', meta

def train_seed_fold(data, recipe, fold, out, code_hash,
                    steps=TRAIN_STEPS, batch=FORMAL_BATCH, workers=2, eval_every=250):
    """Scratch-training recipe with verified successful-update accounting."""
    assert steps > 0 and batch > 0 and eval_every > 0
    out = Path(out)
    _, spec = rt.signature(data, recipe, fold, steps, batch, code_hash, eval_every)
    spec.update(initialization='independent_random_seed', warm_start=False,
        objective='original_bce_dice', bn_statistics='training', checkpoint_metric='no_tta_proxy_dice',
        python=rt.platform.python_version(), numpy=np.__version__, opencv=rt.cv2.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name() if rt.DEVICE.type == 'cuda' else None)
    sig = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'completed.json').exists():
        return completed_training(out, sig, steps)
    if (out/'spec.json').exists():
        assert json.loads((out/'spec.json').read_text())['signature'] == sig, '配置变化，请换OUTPUT_NAME。'
    rt.write_json(out/'spec.json', dict(signature=sig, spec=spec))
    tr, va = np.flatnonzero(data.folds != fold), np.flatnonzero(data.folds == fold)
    assert len(tr) == 16 and len(va) == 4 and not set(tr) & set(va)
    rt.seed_all(recipe.seed)
    model = rt.VesselUNet(recipe).to(rt.DEVICE)
    initial_sha = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
        for p in model.state_dict().values())).hexdigest()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    max_skips = 50
    dataset = rt.PatchDataset(data.prepared(recipe.mode), data.labels, tr, recipe,
        (steps+max_skips)*batch, recipe.seed+fold*1000003)
    loader = rt.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers,
        pin_memory=rt.DEVICE.type == 'cuda', persistent_workers=workers > 0)
    optim = torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.wd)
    scaler = torch.amp.GradScaler('cuda', enabled=rt.DEVICE.type == 'cuda')
    warmup = min(100, max(1, steps//10))
    start, updates, attempts, skipped = time.time(), 0, 0, 0
    native, best = rt.validate(ema, data, va, recipe.mode)
    best_step = 0
    history = [dict(step=0, attempted_batches=0, loss=0., lr=0., native_dice=native,
        proxy_dice=best, elapsed_seconds=0.)]
    path = out/'best.pt'

    def save_best(chosen, step):
        torch.save(dict(signature=sig, spec=spec, recipe=asdict(recipe), fold=fold, step=step,
            model={k:v.detach().cpu().clone() for k,v in chosen.state_dict().items()}), path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)

    save_best(ema, 0)
    pd.DataFrame(history).to_csv(out/'history.csv', index=False)
    ema_state = ema.state_dict()
    print(f'Independent RGB seed={recipe.seed}, fold={fold}: {steps} successful updates, batch={batch}', flush=True)
    for x, target in loader:
        attempts += 1
        step = updates+1
        model.train()
        fraction = max(0., (step-warmup)/max(1, steps-warmup))
        lr = recipe.lr*(step/warmup if step <= warmup else .05+.95*.5*(1+math.cos(math.pi*fraction)))
        for group in optim.param_groups:
            group['lr'] = lr
        x, target = x.to(rt.DEVICE), target.to(rt.DEVICE)
        optim.zero_grad(set_to_none=True)
        with rt.autocast():
            loss = rt.loss_fn(model(x), target, recipe.dice_weight)
        assert torch.isfinite(loss), '非有限loss，停止并保留日志。'
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        norm = nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if rt.DEVICE.type != 'cuda':
            assert torch.isfinite(norm), '非有限梯度。'
        old_scale = scaler.get_scale()
        scaler.step(optim)
        scaler.update()
        if scaler.get_scale() < old_scale:
            skipped += 1
            assert skipped < max_skips, 'AMP跳过过多更新；未写入完成标记。'
            continue
        updates += 1
        decay = min(.995, (updates-1)/(updates+9))
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if value.is_floating_point() and 'running_' not in key:
                    ema_state[key].mul_(decay).add_(value, alpha=1-decay)
                else:
                    ema_state[key].copy_(value)
        if updates % eval_every == 0 or updates == steps:
            native, score = rt.validate(ema if recipe.ema else model, data, va, recipe.mode)
            history.append(dict(step=updates, attempted_batches=attempts, loss=float(loss.detach().cpu()),
                lr=lr, native_dice=native, proxy_dice=score, elapsed_seconds=time.time()-start))
            pd.DataFrame(history).to_csv(out/'history.csv', index=False)
            if score > best:
                best, best_step = score, updates
                save_best(ema if recipe.ema else model, updates)
            print(f'  actual={updates}/{steps} attempted={attempts} loss={history[-1]["loss"]:.4f} proxy={score:.6f}', flush=True)
        if updates == steps:
            break
    assert updates == steps
    meta = dict(signature=sig, recipe=asdict(recipe), fold=int(fold), requested_updates=steps,
        optimizer_updates=updates, attempted_batches=attempts, skipped_updates=skipped,
        best_step=best_step, best_proxy_dice=best, initial_state_sha256=initial_sha,
        warm_start=False, seconds=time.time()-start)
    rt.write_json(out/'completed.json', meta)
    del loader, model, ema, optim
    if rt.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return completed_training(out, sig, steps)
