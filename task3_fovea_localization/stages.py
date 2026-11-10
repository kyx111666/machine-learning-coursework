"""Grouped five-fold coarse and fine stages, OOF summaries and submission export."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import load_dataset, find_data_dir, coarse_sample, fine_sample, coarse_base, square, stem_key
from training import train_model, predict, default_device
from submission import write_submission


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))

def metrics(target, pred, folds):
    e = np.linalg.norm(target - pred, axis=1)
    row = dict(rmse=rmse(target, pred), mean_ed=float(e.mean()), median_ed=float(np.median(e)),
               p90_ed=float(np.quantile(e, .9)), max_ed=float(e.max()))
    for k in range(5):
        row[f'fold{k}'] = rmse(target[folds == k], pred[folds == k])
    return row

def summarize(name, train, out):
    idx = {fd.name: i for i, fd in enumerate(train)}
    target = np.array([fd.xy for fd in train])
    folds = np.array([fd.fold for fd in train])
    parts = [np.load(out / f'fold{k}.npz') for k in range(5)]
    keys = sorted({k[4:] for k in parts[0].files if k.startswith('val|')})
    oof, test, rows = {}, {}, []
    for key in keys:
        pred = np.full((len(train), 2), np.nan)
        for z in parts:
            for n, p in zip(z['val_names'], z['val|' + key]):
                pred[idx[str(n)]] = p
        assert np.isfinite(pred).all()
        oof[key], test[key] = pred, np.mean([z['test|' + key] for z in parts], 0)
        rows.append(dict(model=name, variant=key, **metrics(target, pred, folds)))
    np.savez(out / 'oof.npz', **oof)
    np.savez(out / 'test.npz', **test)
    table = pd.DataFrame(rows).sort_values('rmse')
    table.to_csv(out / 'summary.csv', index=False)
    return table

def _fold_loop(name, cfg, train, test, out, device, folds, make, infer):
    out.mkdir(parents=True, exist_ok=True)
    identity = {'config': cfg,
                'train': [(fd.name, fd.fold, fd.group, fd.xy.tolist(), fd.w, fd.h) for fd in train],
                'test': [(fd.name, fd.group, fd.w, fd.h) for fd in test]}
    record = out / 'identity.json'
    normalized = json.loads(json.dumps(identity))
    if record.exists() and json.loads(record.read_text()) != normalized:
        raise ValueError('Input images, grouped split or configuration changed; use a new output directory')
    if not record.exists() and list(out.glob('fold*.npz')):
        raise ValueError('Existing predictions have no input identity record; use a new output directory')
    record.write_text(json.dumps(identity, indent=2))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    for k in folds:
        dest = out / f'fold{k}.npz'
        if dest.exists():
            with np.load(dest, allow_pickle=False) as saved:
                expected = [fd.name for fd in train if fd.fold == k]
                if saved['val_names'].astype(str).tolist() != expected:
                    raise ValueError('Cached validation images differ from the grouped split')
                for key in saved.files:
                    if key.startswith(('val|', 'test|')) and not np.isfinite(saved[key]).all():
                        raise ValueError('Nonfinite cached predictions')
            continue
        tr = [fd for fd in train if fd.fold != k]
        va = [fd for fd in train if fd.fold == k]
        t0 = time.time()
        target = np.array([fd.xy for fd in va])

        def val_fn(model):
            return {f'val_{key}': rmse(target, v) for key, v in infer(model, va).items()
                    if key.split('|')[0] in ('global', 'sharp', 'p25', 'sp25')}
        model, hist = train_model(cfg, tr, make, torch.device(device), 2026 + k + cfg.get('seed_offset', 0),
                                  log=lambda r: print(name, k, {a: round(b, 4) for a, b in r.items()}, flush=True),
                                  val_fn=val_fn if cfg.get('val_every') else None, val_every=cfg.get('val_every') or 100)
        result = {f'{split}|{key}': v for split, items in (('val', va), ('test', test))
                  for key, v in infer(model, items).items()}
        pd.DataFrame(hist).to_csv(out / f'history_fold{k}.csv', index=False)
        if cfg.get('save_weights'):
            torch.save({n: v.half() if v.is_floating_point() else v for n, v in model.state_dict().items()},
                       out / f'model_fold{k}.pt')
        np.savez(dest, val_names=np.array([fd.name for fd in va]), seconds=time.time() - t0, **result)
        print(name, 'fold', k, f'{time.time() - t0:.0f}s', flush=True)
        del model
    return summarize(name, train, out) if all((out / f'fold{k}.npz').exists() for k in range(5)) else None

def run_coarse(name, cfg, train, test, root, device, folds=range(5)):
    size = cfg['size']

    def make(fd, rng):
        return coarse_sample(fd, size, rng, cfg['aug'])

    def infer(model, items):
        res = predict(model, items, lambda fd: coarse_base(fd, size, crop=cfg['aug'].get('crop', 'fov')), size,
                      torch.device(device))
        return {f'{d}|{m}': v for (d, m), v in res.items()}
    return _fold_loop(name, cfg, train, test, Path(root) / name, device, folds, make, infer)

PRIORS = (10, 15, 20, 25, 35, 50)

def fine_decoders(cfg):
    """Gaussian prior around the crop centre (the coarse estimate); sigma given in original pixels."""
    k = cfg['size'] / cfg['crop']
    dec = {'global': dict(), 'sharp': dict(temperature=.5)}
    for p in PRIORS:
        dec[f'p{p}'] = dict(prior_sigma=p * k)
        dec[f'sp{p}'] = dict(prior_sigma=p * k, temperature=.5)
    return dec

def run_fine(name, cfg, coarse, coarse_key, train, test, root, device, folds=range(5)):
    """Crops are centred on the coarse OOF prediction (validation) or the coarse fold-mean (test)."""
    size, crop = cfg['size'], cfg['crop']
    centers = dict(zip([fd.name for fd in train], np.load(Path(root) / coarse / 'oof.npz')[coarse_key]))
    centers.update(zip([fd.name for fd in test], np.load(Path(root) / coarse / 'test.npz')[coarse_key]))

    def make(fd, rng):
        return fine_sample(fd, size, crop, rng, cfg['aug'])

    decoders = fine_decoders(cfg)

    def infer(model, items):
        cur, result = {fd.name: centers[fd.name] for fd in items}, {}
        for it in range(cfg['iters']):
            res = predict(model, items, lambda fd, c=dict(cur): square(*c[fd.name], crop, size), size,
                          torch.device(device), decoders=decoders)
            result.update({f'{d}|{m}|it{it + 1}': v for (d, m), v in res.items()})
            cur = dict(zip([fd.name for fd in items], res[(cfg.get('center_decoder', 'p25'), 'flip')]))
        return result
    cfg = dict(cfg, coarse=coarse, coarse_key=coarse_key)
    return _fold_loop(name, cfg, train, test, Path(root) / name, device, folds, make, infer)
