"""Training loop (parameter groups, warmup-cosine, EMA weights) and affine-TTA prediction."""
import copy
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from data import render, apply, hflip
from models import HeatNet, heat_loss, soft_argmax, DECODERS

def default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def batch_stream(make, items, batch, steps, seed, workers=6):
    """Shuffled passes over the training images; samples are built by threads one batch ahead."""
    rng = np.random.default_rng(seed)
    reps = math.ceil(steps * batch / len(items)) + 1
    order = np.concatenate([rng.permutation(len(items)) for _ in range(reps)])[:steps * batch]
    seeds = rng.integers(0, 2 ** 31 - 1, len(order))
    with ThreadPoolExecutor(workers) as pool:
        futures = {}

        def submit(i):
            if i < len(order):
                futures[i] = pool.submit(make, items[order[i]], np.random.default_rng(seeds[i]))

        for i in range(3 * batch):
            submit(i)
        for s in range(steps):
            out = [futures.pop(s * batch + j).result() for j in range(batch)]
            for i in range((s + 3) * batch, (s + 4) * batch):
                submit(i)
            x = torch.from_numpy(np.ascontiguousarray(np.stack([o[0] for o in out]).transpose(0, 3, 1, 2)))
            y = torch.from_numpy(np.stack([o[1] for o in out]).astype(np.float32))
            yield x, y

def build_model(cfg, pretrained=True):
    return HeatNet(cfg['encoder'], pretrained, drop=cfg.get('drop', 0.))

def train_model(cfg, items, make, device, seed, log=print, val_fn=None, val_every=100):
    """val_fn(ema_model) -> dict is only logged (learning curves); it never selects weights or steps."""
    seed_all(seed)
    model = build_model(cfg, cfg.get('pretrained', True)).to(device)
    cuda = device.type == 'cuda'
    if cuda:
        model = model.to(memory_format=torch.channels_last)
    enc = [p for n, p in model.named_parameters() if n.startswith('enc.')]
    dec = [p for n, p in model.named_parameters() if not n.startswith('enc.')]
    opt = torch.optim.AdamW([{'params': enc, 'lr': cfg['lr_enc']}, {'params': dec, 'lr': cfg['lr_dec']}],
                            weight_decay=cfg.get('wd', 1e-2))
    steps = cfg['steps']
    warm = max(1, int(.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              .02 + .98 * .5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)
    ema = copy.deepcopy(model)
    ema_f = [v for v in ema.state_dict().values() if v.dtype.is_floating_point]
    cur_f = [v for v in model.state_dict().values() if v.dtype.is_floating_point]
    model.train()
    if cfg.get('freeze_bn'):

        for m in model.enc.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
    t0, hist = time.time(), []
    for step, (x, y) in enumerate(batch_stream(make, items, cfg['batch'], steps, seed, cfg.get('workers', 6))):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
            logits = model(x)
        loss, ce, l1 = heat_loss(logits.float(), y, cfg['sigma'], cfg.get('coord_w', 1.))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(opt)
        scaler.update()
        sched.step()
        decay = min(cfg.get('ema', .995), (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_mul_(ema_f, decay)
            torch._foreach_add_(ema_f, cur_f, alpha=1 - decay)
        if (step + 1) % val_every == 0 or step == 0:
            rec = dict(step=step + 1, loss=float(loss.detach()), ce=float(ce), l1_cells=float(l1),
                       lr_enc=opt.param_groups[0]['lr'], seconds=time.time() - t0)
            if val_fn is not None and (step + 1) % val_every == 0:
                _sync_buffers(model, ema)
                rec.update(val_fn(ema.eval()))
                ema.train()
            hist.append(rec)
            log(rec)
    _sync_buffers(model, ema)
    return ema.eval(), hist

def _sync_buffers(model, ema):
    for (k, v) in model.state_dict().items():
        if not v.dtype.is_floating_point:
            ema.state_dict()[k].copy_(v)

@torch.no_grad()
def predict(model, items, base_fn, size, device, flips=(False, True), batch=8, decoders=None):
    """Original-pixel predictions for every decoder, without ('plain') and with ('flip') mirror TTA.

    base_fn(fd) returns the 3x3 map from original pixels to network input; TTA views are inverted exactly."""
    model.eval()
    cuda = device.type == 'cuda'
    decoders = decoders or DECODERS
    out = {(d, f): [] for d in decoders for f in flips}
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        for flip in flips:
            Ms = [hflip(size) @ base_fn(fd) if flip else base_fn(fd) for fd in chunk]
            x = np.stack([render(fd, M, size) for fd, M in zip(chunk, Ms)]).astype(np.float32) / 255
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)
            if cuda:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
                logits = model(x).float()
            for d, kw in decoders.items():
                p = soft_argmax(logits, model.stride, **kw).cpu().numpy()
                out[(d, flip)].append(np.stack([apply(np.linalg.inv(M), q) for M, q in zip(Ms, p)]))
    res = {}
    for d in decoders:
        plain = np.concatenate(out[(d, False)])
        res[(d, 'plain')] = plain
        if True in flips:
            res[(d, 'flip')] = (plain + np.concatenate(out[(d, True)])) / 2
    return res
