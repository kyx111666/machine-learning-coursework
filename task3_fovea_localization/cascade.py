"""Final pipeline: equal-weight coarse ensemble -> fine models re-centred on it -> equal-weight fine ensemble."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stages import fine_decoders
from inference import load_fold_model, fine_infer

def coarse_ensemble(root, runs, key):
    oof = np.mean([np.load(Path(root) / r / 'oof.npz')[key] for r in runs], 0)
    test = np.mean([np.load(Path(root) / r / 'test.npz')[key] for r in runs], 0)
    return oof, test

def recentre_fine(root, run, train, test, oof_centers, test_centers, device, iters=2, center_dec='p25'):
    """Out-of-fold re-inference of one fine run around new centres; cached by a hash of the centres."""
    run_dir = Path(root) / run
    cfg = json.loads((run_dir / 'config.json').read_text())
    decoders = fine_decoders(cfg)
    digest = hashlib.sha256(np.round(np.r_[oof_centers.ravel(), test_centers.ravel()], 3).tobytes()
                            + json.dumps(decoders, sort_keys=True).encode()).hexdigest()[:12]
    cache = run_dir / f'recentre_{digest}_{center_dec}_it{iters}.npz'
    if cache.exists():
        z = np.load(cache)
        return {k[4:]: z[k] for k in z.files if k.startswith('oof|')}, {k[5:]: z[k] for k in z.files if k.startswith('test|')}
    centers = dict(zip([fd.name for fd in train], oof_centers))
    centers.update(zip([fd.name for fd in test], test_centers))
    idx = {fd.name: i for i, fd in enumerate(train)}
    oof, tests = {}, []
    for k in range(5):
        model, cfg = load_fold_model(run_dir, k, device)
        va = [fd for fd in train if fd.fold == k]
        res = fine_infer(model, va, centers, cfg, device, decoders, iters=iters, center_key=(center_dec, 'flip'))
        for key, v in res.items():
            oof.setdefault('|'.join(map(str, key)), np.zeros((len(train), 2)))[[idx[fd.name] for fd in va]] = v
        tests.append(fine_infer(model, test, centers, cfg, device, decoders, iters=iters, center_key=(center_dec, 'flip')))
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()
    test_mean = {'|'.join(map(str, key)): np.mean([t[key] for t in tests], 0) for key in tests[0]}
    np.savez(cache, **{f'oof|{k}': v for k, v in oof.items()}, **{f'test|{k}': v for k, v in test_mean.items()})
    return oof, test_mean

def final_predictions(root, train, test, device, coarse_runs, coarse_key, fine_runs, fine_key, iters=2,
                      center_dec='p25'):
    co_oof, co_test = coarse_ensemble(root, coarse_runs, coarse_key)
    fine = {r: recentre_fine(root, r, train, test, co_oof, co_test, device, iters, center_dec) for r in fine_runs}
    oof = np.mean([fine[r][0][fine_key] for r in fine_runs], 0)
    tst = np.mean([fine[r][1][fine_key] for r in fine_runs], 0)
    return dict(coarse_oof=co_oof, coarse_test=co_test, oof=oof, test=tst, per_run=fine)
