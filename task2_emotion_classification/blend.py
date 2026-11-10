"""Fixed family-weight ensembles for the completed task 2 submissions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = 'ema_tta10'
# run key: (output folder, experiment name)
RUNS = {
    'resnet18_seed2026': ('cv_resnet18', 'resnet18_pt112'),
    'resnet18_seed2027': ('cv_resnet18_s2027', 'resnet18_pt112_s2027'),
    'resnet32_seed2026': ('cv_resnet32', 'resnet32_scratch48'),
    'resnet32_seed2027': ('cv_resnet32_s2027', 'resnet32_scratch48_s2027'),
    'efficientnet': ('cv_effnet_b0', 'effnet_b0_pt112'),
    'deit': ('cv_deit_tiny', 'deit_tiny_pt112'),
    'vit': ('cv_vit_small', 'vit_small_pt112'),
    'convnext': ('cv_convnext_tiny', 'convnext_tiny_pt112'),
    'vgg': ('cv_vgg_scratch48', 'vgg_scratch48'),
}
BASE_PRETRAINED = {'resnet18_seed2026': .05, 'resnet18_seed2027': .05,
                   'efficientnet': .1, 'deit': .1, 'vit': .1, 'convnext': .1}
WEIGHTS = {
    'seeds': {**BASE_PRETRAINED, 'holdout_resnet32': .25,
              'resnet32_seed2026': .125, 'resnet32_seed2027': .125},
    'vgg': {'vgg': .25, **BASE_PRETRAINED, 'holdout_resnet32': .125,
            'resnet32_seed2026': .0625, 'resnet32_seed2027': .0625},
}


def check_probabilities(probabilities, rows):
    if (probabilities.shape != (rows, 6) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-4)):
        raise ValueError(f'Expected a normalized {rows}x6 probability matrix')


def weighted_mean(probabilities, weights):
    if not weights or not np.isclose(sum(weights.values()), 1) or any(w < 0 for w in weights.values()):
        raise ValueError('Ensemble weights must be nonnegative and sum to one')
    rows = len(probabilities[next(iter(weights))])
    for name in weights:
        check_probabilities(probabilities[name], rows)
    return sum(weight * probabilities[name] for name, weight in weights.items())


def cv_weights(recipe):
    weights = {name: weight for name, weight in WEIGHTS[recipe].items() if name != 'holdout_resnet32'}
    for name in weights:
        if name.startswith('resnet32_'):
            weights[name] *= 2
    return weights


def read_cv_run(directory, experiment):
    status = json.loads((directory / 'status.json').read_text())
    if status['pending'] or sorted(status['folds_done'].get(experiment, [])) != list(range(5)):
        raise ValueError(f'{directory}: all five folds must be complete')
    index = pd.read_csv(directory / 'train_index.csv')
    columns = ['relative_path', 'label', 'fold', 'group']
    if not set(columns).issubset(index) or index.relative_path.duplicated().any():
        raise ValueError(f'{directory}: invalid training index')
    for group, frame in index.groupby('group'):
        if frame['fold'].nunique() != 1:
            raise ValueError(f'{directory}: a duplicate group crosses validation folds')
    with np.load(directory / f'oof_{experiment}.npz', allow_pickle=False) as saved:
        if not saved['mask'].all() or not np.array_equal(saved['labels'], index.label):
            raise ValueError(f'{directory}: incomplete or misaligned OOF')
        oof = saved[KEY].copy()
    with np.load(directory / f'test_{experiment}.npz', allow_pickle=False) as saved:
        test = saved[KEY].copy()
        ids = saved['ids'].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError(f'{directory}: duplicate test IDs')
    check_probabilities(oof, len(index))
    check_probabilities(test, len(ids))
    per_fold = []
    for fold in range(5):
        with np.load(directory / 'experiments' / experiment / f'fold{fold}' / 'test_probs.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'].astype(str), ids):
                raise ValueError(f'{directory}: test IDs differ between folds')
            probability = saved[KEY].copy()
            check_probabilities(probability, len(ids))
            per_fold.append(probability)
    if not np.allclose(test, np.mean(per_fold, 0), atol=1e-6):
        raise ValueError(f'{directory}: test probabilities differ from the five-fold mean')
    return index[columns], oof, test, ids, status


def load_runs(root, recipe):
    keys = [key for key in WEIGHTS[recipe] if key != 'holdout_resnet32']
    reference = None
    oof, test = {}, {}
    for key in keys:
        folder, experiment = RUNS[key]
        index, op, tp, ids, status = read_cv_run(root / folder, experiment)
        identity = (status['data_fingerprint'], status['split_fingerprint'])
        if reference is None:
            reference = (index, ids, identity)
        elif (not index.equals(reference[0]) or not np.array_equal(ids, reference[1])
              or identity != reference[2]):
            raise ValueError(f'{key}: image order or shared split differs from other runs')
        oof[key], test[key] = op, tp
    return reference[0], reference[1], oof, test


def load_holdout(root, index, ids):
    directory = root / 'single_resnet32'
    probabilities = np.load(directory / 'test_probabilities.npy', allow_pickle=False)
    check_probabilities(probabilities, len(ids))
    source_ids = pd.read_csv(directory / 'test_ids.csv', dtype={'ID': str}).ID.to_numpy()
    if not np.array_equal(source_ids, ids):
        raise ValueError('Single-model test IDs differ from the cross-validation outputs')
    manifest = pd.read_csv(directory / 'split_manifest.csv')
    if not manifest[['relative_path', 'label', 'group']].equals(index[['relative_path', 'label', 'group']]):
        raise ValueError('Single-model training images differ from cross-validation inputs')
    return probabilities


def write_submission(ids, probabilities, path):
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate test IDs')
    check_probabilities(probabilities, len(ids))
    pd.DataFrame({'ID': ids, 'Emotion': probabilities.argmax(1).astype(int)}).to_csv(path, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--recipe', choices=WEIGHTS, default='seeds')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    index, ids, oof, test = load_runs(args.results, args.recipe)
    if len(index) != 28275 or len(ids) != 7065:
        raise ValueError('Final submissions require all 28,275 training and 7,065 test images')
    test['holdout_resnet32'] = load_holdout(args.results, index, ids)
    probabilities = weighted_mean(test, WEIGHTS[args.recipe])
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f'submission_{args.recipe}.csv'
    write_submission(ids, probabilities, path)
    cv = weighted_mean(oof, cv_weights(args.recipe))
    metrics = {'recipe': args.recipe, 'weights': WEIGHTS[args.recipe],
               'oof_accuracy_without_holdout_model': float((cv.argmax(1) == index.label).mean()),
               'oof_rows': len(index), 'test_rows': len(ids)}
    (args.out / f'{args.recipe}_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(path)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
