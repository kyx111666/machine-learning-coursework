"""Train either fixed final three-branch ensemble from official competition data."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import runtime as rt
from training import train_seed_fold
from predictions import CALIBRATION, expanded_predictions, image_metrics
from gamma import gamma_recipes, train_gamma_fold


def require_kaggle():
    if rt.DEVICE.type != 'cuda' or not Path('/kaggle/working').is_dir():
        raise RuntimeError('Run training on Kaggle with GPU enabled.')


def source_hash():
    root = Path(__file__).resolve().parent
    return hashlib.sha256(b''.join(p.name.encode() + p.read_bytes()
                                  for p in sorted(root.glob('*.py')))).hexdigest()


def final_recipes(route):
    if route == 'seed_blend':
        return {'rgb1': rt.Recipe(name='input_rgb', mode='rgb'),
                'rgb2': rt.Recipe(name='input_rgb_seed2', mode='rgb', seed=20261102),
                'green': rt.Recipe(name='model_plain_unet', arch='unet', mode='green')}
    if route == 'gamma':
        recipes = gamma_recipes()
        return {'rgb1': recipes['first'], 'rgb2': recipes['gamma'], 'green': recipes['green']}
    raise ValueError(f'Unknown final route: {route}')


def fuse(branches, route):
    if set(branches) != {'rgb1', 'rgb2', 'green'}:
        raise ValueError('Expected RGB1, RGB2 and green probabilities')
    shapes = {np.asarray(p).shape for p in branches.values()}
    if len(shapes) != 1 or any(not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
                               for p in branches.values()):
        raise ValueError('Invalid probability arrays')
    if route == 'seed_blend':
        return .75*(.5*branches['rgb1'] + .5*branches['rgb2']) + .25*branches['green']
    if route == 'gamma':
        return .375*branches['rgb1'] + .375*branches['rgb2'] + .25*branches['green']
    raise ValueError(f'Unknown final route: {route}')


def run_final(data, out, route, code_hash, workers=2):
    require_kaggle()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    recipes = final_recipes(route)
    plan = dict(route=route, data=data.fingerprint, code_hash=code_hash,
                recipes={key: asdict(cfg) for key, cfg in recipes.items()},
                steps=4000, batch=12, workers=workers, calibration=CALIBRATION)
    record = out/'plan.json'
    if record.exists() and json.loads(record.read_text()) != plan:
        raise ValueError('Data or configuration changed; use a new output directory')
    rt.write_json(record, plan)
    oof = {key: np.full((20, 584, 565), np.nan, np.float32) for key in recipes}
    models = {}
    for key, cfg in recipes.items():
        for fold in range(5):
            folder = out/'models'/cfg.name/f'fold{fold}'
            if route == 'seed_blend' and key != 'rgb2':
                checkpoint, _ = rt.train_fold(data, cfg, fold, 4000, 12, folder, code_hash, workers)
            elif route == 'gamma' and key == 'rgb2':
                checkpoint, _ = train_gamma_fold(data, cfg, fold, folder, code_hash, workers)
            else:
                checkpoint, _ = train_seed_fold(data, cfg, fold, folder, code_hash, workers=workers)
            models[key, fold] = checkpoint
            oof[key][data.folds == fold] = expanded_predictions(data, cfg, fold, checkpoint,
                out/'probabilities'/cfg.name/f'fold{fold}', code_hash)
    final_oof = fuse(oof, route)
    rows = image_metrics(data, final_oof, range(20), route)
    pd.DataFrame(rows).to_csv(out/'oof_per_image.csv', index=False)
    rt.write_json(out/'metrics.json', dict(route=route, oof_proxy_dice=float(np.mean([r['dice'] for r in rows])),
                  model_count=15, **CALIBRATION))
    # Fixed routes and calibration are saved before any test prediction is used.
    rt.write_json(out/'frozen_route.json', plan)
    tests = {}
    for key, cfg in recipes.items():
        total = np.zeros((20, 584, 565), np.float32)
        for fold in range(5):
            total += expanded_predictions(data, cfg, fold, models[key, fold],
                out/'probabilities'/cfg.name/f'fold{fold}', code_hash, test=True)/5
        tests[key] = total
    final_test = fuse(tests, route)
    for key in recipes:
        directory = out/'components'/key
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory/'probabilities.npz', oof=oof[key], test=tests[key],
                            train_ids=np.asarray(data.ids, dtype=str), test_ids=np.asarray(data.test_ids, dtype=str),
                            folds=data.folds, data_fingerprint=data.fingerprint, route=route, code_hash=code_hash)
    np.savez_compressed(out/'selected_probabilities.npz', oof=final_oof, test=final_test,
                        train_ids=np.asarray(data.ids, dtype=str), test_ids=np.asarray(data.test_ids, dtype=str),
                        folds=data.folds, data_fingerprint=data.fingerprint, route=route, code_hash=code_hash)
    rt.make_submission(data, final_test, CALIBRATION, out/'selected')
    return final_oof, final_test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=None)
    parser.add_argument('--out', type=Path, default=Path('/kaggle/working/task4'))
    parser.add_argument('--route', choices=['seed_blend', 'gamma'], default='gamma')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    require_kaggle()
    if args.workers < 0:
        parser.error('--workers must be nonnegative')
    data = rt.OfficialData(rt.find_data(args.data), args.out/'data_checks')
    rt.verify_official_encoder(data.root, args.out/'data_checks')
    run_final(data, args.out/args.route, args.route, source_hash(), args.workers)


if __name__ == '__main__':
    main()
