"""The two fixed coordinate blends selected for the final task 3 submission."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from data import find_data_dir, dataset_fingerprint, stem_key, FROZEN_FOLDS, SPLIT2_FOLDS
from submission import write_submission

FORMULAS = {
    'four_component': {'cascade': .35, 'resnet34_transfer': .35, 'dropout_split1': .15, 'dropout_split2': .15},
    'cascade_scratch': {'cascade': .7, 'dropout_split1': .3},
}


def blend(parts, formula):
    if not formula or not np.isclose(sum(formula.values()), 1) or any(weight < 0 for weight in formula.values()):
        raise ValueError('Blend weights must be nonnegative and sum to one')
    shape = np.asarray(parts[next(iter(formula))]).shape
    for name in formula:
        array = np.asarray(parts[name])
        if array.shape != shape or len(shape) != 2 or shape[1] != 2 or not np.isfinite(array).all():
            raise ValueError(f'Invalid coordinate matrix: {name}')
    return sum(weight * np.asarray(parts[name], dtype=np.float64) for name, weight in formula.items())


def load_predictions(root, data, recipe):
    labels = pd.read_csv(data / 'fovea_localization_train_GT.csv', dtype={'data': str})
    train_paths = {stem_key(path.name): path for path in (data / 'train').glob('*.jpg')}
    names = np.asarray([train_paths[stem_key(name)].name for name in labels.data])
    tests = sorted((data / 'test').glob('*.jpg'))
    test_names = np.asarray([path.name for path in tests])
    target = labels[['Fovea_X', 'Fovea_Y']].to_numpy()
    fingerprint = dataset_fingerprint(data)
    oof, test = {}, {}
    for name in FORMULAS[recipe]:
        path = root / 'cascade_predictions.npz' if name == 'cascade' else root / 'components' / name / 'predictions_sharp.npz'
        with np.load(path, allow_pickle=False) as saved:
            if (not np.array_equal(saved['train_names'], names) or not np.array_equal(saved['test_names'], test_names)
                    or str(saved['dataset_fingerprint']) != fingerprint):
                raise ValueError(f'{name}: input images or prediction order differs')
            np.testing.assert_allclose(saved['target'], target, rtol=0, atol=1e-8)
            assignment = SPLIT2_FOLDS if name == 'dropout_split2' else FROZEN_FOLDS
            expected = np.asarray([int(assignment[Path(item).stem]) for item in names])
            if not np.array_equal(saved['folds'], expected):
                raise ValueError(f'{name}: grouped split differs from its training protocol')
            groups = saved['groups'].astype(str)
            for fold in range(5):
                if not set(groups[expected == fold]).isdisjoint(groups[expected != fold]):
                    raise ValueError(f'{name}: duplicate images cross folds')
            op, tp = saved['oof'].copy(), saved['test'].copy()
        if op.shape != (80, 2) or tp.shape != (20, 2) or not np.isfinite(op).all() or not np.isfinite(tp).all():
            raise ValueError(f'{name}: incomplete prediction matrix')
        oof[name], test[name] = op, tp
    bounds = []
    for path in tests:
        with Image.open(path) as image:
            width, height = image.size
        bounds.append(SimpleNamespace(name=path.name, w=width, h=height))
    return target, bounds, pd.read_csv(data / 'sample_submission.csv'), oof, test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=None)
    parser.add_argument('--results', type=Path, default=Path('outputs'))
    parser.add_argument('--recipe', choices=FORMULAS, default='four_component')
    parser.add_argument('--out', type=Path, default=Path('submissions'))
    args = parser.parse_args(argv)
    target, test, sample, oof, predictions = load_predictions(args.results, find_data_dir(args.data), args.recipe)
    formula = FORMULAS[args.recipe]
    final_oof, final_test = blend(oof, formula), blend(predictions, formula)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f'submission_{args.recipe}.csv'
    write_submission(test, final_test, sample, path)
    metrics = {'recipe': args.recipe, 'weights': formula, 'oof_rmse': float(np.sqrt(np.mean((target - final_oof) ** 2))),
               'training_images': 80, 'test_images': 20, 'submission_rows': 40}
    (args.out / f'{args.recipe}_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(path)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
