"""Paired channel-wise gamma training augmentation with unchanged raw-RGB inference."""
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import runtime as rt
import predictions as metrics
from training import train_seed_fold
from predictions import expanded_predictions, bundle_expanded_outputs

TRAIN_STEPS = 4000
FORMAL_BATCH = 12
GAMMA_PROBABILITY = .5
GAMMA_RANGE = (.5, 2.)
PILOT_FOLDS = [0, 2, 4]
CONFIRM_FOLDS = [1, 3]
BASE_PATCH_DATASET = rt.PatchDataset


def gamma_chw(image, seed, index):
    """A separate RNG leaves original crop/geometry/photometric draws unchanged."""
    image = np.asarray(image)
    assert image.ndim == 3 and image.shape[0] == 3 and image.dtype == np.float32
    assert np.isfinite(image).all() and image.min() >= -1 and image.max() <= 1
    rng = np.random.default_rng([int(seed), int(index), 20261104])
    if rng.random() >= GAMMA_PROBABILITY:
        return image.copy()
    powers = np.exp(rng.uniform(np.log(GAMMA_RANGE[0]), np.log(GAMMA_RANGE[1]), (3, 1, 1))).astype(np.float32)
    values = np.clip((image+1)*.5, 0, 1)
    return np.ascontiguousarray(2*np.power(values, powers)-1, dtype=np.float32)


class GammaPatchDataset(BASE_PATCH_DATASET):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        assert self.cfg.mode == 'rgb' and self.cfg.augmentation
        augmented = gamma_chw(image.numpy(), self.seed, index)
        return torch.from_numpy(augmented), target


def train_gamma_fold(data, recipe, fold, out, code_hash, workers=2):
    assert rt.DEVICE.type == 'cuda' and Path('/kaggle/working').is_dir()
    original = rt.PatchDataset
    assert original is BASE_PATCH_DATASET
    try:
        rt.PatchDataset = GammaPatchDataset
        return train_seed_fold(data, recipe, fold, out, code_hash,
            steps=TRAIN_STEPS, batch=FORMAL_BATCH, workers=workers)
    finally:
        rt.PatchDataset = original


def gamma_recipes():
    rgb = rt.Recipe(name='rgb_seed2', mode='rgb', seed=20261102)
    return dict(reference=rgb, gamma=replace(rgb, name='rgb_seed2_gamma'),
                first=replace(rgb, name='rgb_seed1', seed=20261031),
                green=rt.Recipe(name='green', arch='unet', mode='green', seed=20261031))


def gamma_fusions(branches):
    reference = .375*branches['first']+.375*branches['reference']+.25*branches['green']
    candidate = .375*branches['first']+.375*branches['gamma']+.25*branches['green']
    return reference, candidate


def gamma_dataset_selfcheck(data, recipes, out):
    assert rt.DEVICE.type == 'cuda' and Path('/kaggle/working').is_dir()
    indices = np.flatnonzero(data.folds != 0)
    images = data.prepared('rgb')
    original = BASE_PATCH_DATASET(images, data.labels, indices, recipes['reference'], 24, 20261102)
    augmented = GammaPatchDataset(images, data.labels, indices, recipes['gamma'], 24, 20261102)
    changed = 0
    for index in range(24):
        before, target_before = original[index]
        after, target_after = augmented[index]
        assert torch.equal(target_before, target_after)
        assert before.shape == after.shape == (3, 192, 192)
        expected = gamma_chw(before.numpy(), augmented.seed, index)
        assert np.array_equal(after.numpy(), expected)
        assert torch.isfinite(after).all() and after.min() >= -1 and after.max() <= 1
        changed += int(not torch.equal(before, after))
    assert 0 < changed < 24
    rt.write_json(Path(out)/'augmentation_selfcheck.json', dict(samples=24, identical_targets=24,
        changed_images=changed, optimizer_updates=0,
        scope='Kaggle-only dataset semantics check; no model or accuracy evaluation'))


def gamma_gate(scores, phase):
    table = scores.set_index('fold').sort_index()
    expected = [0] if phase == 'first' else (list(range(5)) if phase == 'confirmation' else PILOT_FOLDS)
    assert table.index.tolist() == sorted(expected) and np.isfinite(table.to_numpy()).all()
    gains = table.gain
    if phase == 'first':
        return dict(accepted=False, pruned=bool(gains.mean() <= -.003), mean_gain=float(gains.mean()),
                    improved_folds=int((gains > 0).sum()))
    assert phase in ['rgb_pilot', 'fusion_pilot', 'confirmation']
    minimum = .001 if phase == 'rgb_pilot' else .0005
    checks = dict(mean_gain_at_least_minimum=bool(gains.mean() >= minimum),
                  enough_improved_folds=bool((gains > 0).sum() >= (3 if phase == 'confirmation' else 2)))
    if phase == 'confirmation':
        fresh = gains.loc[CONFIRM_FOLDS]
        checks.update(confirmation_mean_positive=bool(fresh.mean() > 0),
                      confirmation_has_improvement=bool((fresh > 0).any()))
    return dict(accepted=all(checks.values()), mean_gain=float(gains.mean()),
                improved_folds=int((gains > 0).sum()), minimum_gain=minimum, checks=checks,
                rejection_reasons=[key for key, value in checks.items() if not value])


def run_gamma_augmentation(data, out, code_hash, workers=2):
    assert rt.DEVICE.type == 'cuda' and Path('/kaggle/working').is_dir(), '只允许在Kaggle GPU训练。'
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    recipes = gamma_recipes()
    spec = dict(protocol='paired_rgb_channel_gamma', data=data.fingerprint, code_hash=code_hash,
        recipes={k: asdict(cfg) for k, cfg in recipes.items()}, steps=TRAIN_STEPS, batch=FORMAL_BATCH,
        workers=workers, gamma_probability=GAMMA_PROBABILITY, gamma_range=list(GAMMA_RANGE),
        gamma_distribution='independent log-uniform per RGB channel; post-original patch augmentation',
        labels_changed=False, inference='unchanged original raw-RGB whole flip4',
        reference_weights=dict(first=.375, reference=.375, green=.25),
        candidate_weights=dict(first=.375, gamma=.375, green=.25),
        first_fold_prune_gain_at_most=-.003, rgb_pilot_minimum_gain=.001,
        fusion_minimum_gain=.0005, calibration=metrics.CALIBRATION,
        python=rt.platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        opencv=rt.cv2.__version__, device=str(rt.DEVICE), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name())
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if (out/'experiment_spec.json').exists():
        assert json.loads((out/'experiment_spec.json').read_text())['signature'] == signature, '配置变化，请换OUTPUT_NAME。'
    rt.write_json(out/'experiment_spec.json', dict(signature=signature, spec=spec))
    gamma_dataset_selfcheck(data, recipes, out)
    branches = {key: np.zeros((20, 584, 565), np.float32) for key in recipes}
    models, trained, decisions = {}, {}, {}

    def train_one(key, fold):
        cfg = recipes[key]
        folder = out/'models'/cfg.name/f'fold{fold}'
        if key == 'gamma':
            checkpoint, meta = train_gamma_fold(data, cfg, fold, folder, code_hash, workers=workers)
        else:
            assert rt.PatchDataset is BASE_PATCH_DATASET
            checkpoint, meta = train_seed_fold(data, cfg, fold, folder, code_hash,
                steps=TRAIN_STEPS, batch=FORMAL_BATCH, workers=workers)
        assert meta['requested_updates'] == meta['optimizer_updates'] == TRAIN_STEPS and not meta['warm_start']
        assert Path(checkpoint).resolve().is_relative_to(out.resolve())
        models[key, fold] = checkpoint
        trained[key, fold] = dict(recipe=cfg.name, seed=cfg.seed, fold=fold, signature=meta['signature'],
            checkpoint_sha256=rt.digest_file(checkpoint), actual_updates=meta['optimizer_updates'],
            attempted_batches=meta['attempted_batches'], skipped_updates=meta['skipped_updates'],
            best_step=meta['best_step'], initial_state_sha256=meta['initial_state_sha256'],
            batch=FORMAL_BATCH, gamma_enabled=key == 'gamma', warm_start=False,
            train_ids=[data.ids[i] for i in np.flatnonzero(data.folds != fold)],
            validation_ids=[data.ids[i] for i in np.flatnonzero(data.folds == fold)])
        rt.write_json(out/'training_provenance.json', list(trained.values()))
        if key == 'gamma':
            assert meta['initial_state_sha256'] == trained['reference', fold]['initial_state_sha256']
        branches[key][data.folds == fold] = expanded_predictions(data, cfg, fold, checkpoint,
            out/'probabilities'/cfg.name/f'fold{fold}', code_hash)

    def compare(reference, candidate, folds, prefix, phase):
        indices = np.flatnonzero(np.isin(data.folds, folds))
        frame = pd.DataFrame(metrics.image_metrics(data, reference[indices], indices, 'reference')+
                             metrics.image_metrics(data, candidate[indices], indices, 'gamma_candidate'))
        assert len(frame) == 8*len(folds) and not frame.duplicated(['route', 'Id']).any()
        frame.to_csv(out/(prefix+'_per_image.csv'), index=False)
        pairs = frame.pivot(index=['fold', 'Id'], columns='route', values='dice')
        assert not pairs.isna().any().any() and pairs.groupby(level='fold').size().eq(4).all()
        pairs['gain'] = pairs.gamma_candidate-pairs.reference
        pairs.reset_index().to_csv(out/(prefix+'_paired.csv'), index=False)
        scores = pairs.groupby('fold')[['reference', 'gamma_candidate', 'gain']].mean().reset_index()
        scores.to_csv(out/(prefix+'_fold_scores.csv'), index=False)
        gate = gamma_gate(scores, phase)
        decisions[prefix] = dict(**gate, folds=list(folds), reference_dice=float(scores.reference.mean()),
                                  candidate_dice=float(scores.gamma_candidate.mean()))
        rt.write_json(out/(prefix+'_selection.json'), decisions[prefix])
        with (out/'oof_route_probabilities.tmp').open('wb') as stream:
            np.savez_compressed(stream, reference=reference[indices], candidate=candidate[indices],
                targets=np.stack([data.proxy[i] for i in indices]), ids=[data.ids[i] for i in indices],
                folds=data.folds[indices], stage=prefix, calibration_json=json.dumps(metrics.CALIBRATION), code_hash=code_hash)
        (out/'oof_route_probabilities.tmp').replace(out/'oof_route_probabilities.npz')
        return gate

    def save_selection(stage, accepted=False, submission_available=False):
        latest = next(reversed(decisions.values()))
        selection = dict(protocol=spec['protocol'], stage=stage, signature=signature,
            route='gamma_fusion' if accepted else None, new_route_accepted=accepted,
            submission_available=submission_available,
            comparison_scope='full_three_branch_fusion' if 'fusion_pilot' in decisions else 'single_rgb_only',
            evaluated_oof_images=4*len(latest['folds']), decisions=decisions,
            baseline_oof=latest['reference_dice'], candidate_oof=latest['candidate_dice'],
            completed_models=len(trained), completed_training_updates=sum(r['actual_updates'] for r in trained.values()),
            external_checkpoint_inputs=0, public_score_verified=False, best_known_public=.86150, **metrics.CALIBRATION)
        rt.write_json(out/'selection.json', selection)
        (out/'试验结果.md').write_text('# 逐通道Gamma增强试验\n\n'+json.dumps(selection, ensure_ascii=False, indent=2)+
            '\n\n仅训练时对RGB种子2增加50%概率的独立逐通道Gamma，指数log-uniform[0.5,2]。标签不变、推理不变。'
            '参考是本轮新训练的双RGB种子方案；OOF参与选优，不是独立测试。实际Public仍需提交确认。\n', encoding='utf8')
        return selection

    train_one('reference', 0)
    train_one('gamma', 0)
    first = compare(branches['reference'], branches['gamma'], [0], 'rgb_first', 'first')
    if first['pruned']:
        return save_selection('rgb_first_rejected')
    for fold in [2, 4]:
        train_one('reference', fold)
        train_one('gamma', fold)
    rgb_pilot = compare(branches['reference'], branches['gamma'], PILOT_FOLDS, 'rgb_pilot', 'rgb_pilot')
    if not rgb_pilot['accepted']:
        return save_selection('rgb_pilot_rejected')
    for fold in PILOT_FOLDS:
        for key in ['first', 'green']:
            train_one(key, fold)
    reference, candidate = gamma_fusions(branches)
    fusion_pilot = compare(reference, candidate, PILOT_FOLDS, 'fusion_pilot', 'fusion_pilot')
    if not fusion_pilot['accepted']:
        return save_selection('fusion_pilot_rejected')
    rt.write_json(out/'frozen_pilot.json', dict(signature=signature, candidate_weights=spec['candidate_weights'],
        calibration=metrics.CALIBRATION, pilot=decisions['fusion_pilot']))
    for fold in CONFIRM_FOLDS:
        for key in ['reference', 'gamma', 'first', 'green']:
            train_one(key, fold)
    reference, candidate = gamma_fusions(branches)
    full = compare(reference, candidate, list(range(5)), 'fusion_confirmation', 'confirmation')
    if not full['accepted']:
        return save_selection('confirmation_rejected')
    selection = save_selection('validation_accepted', accepted=True)
    rt.write_json(out/'frozen_selection.json', selection)
    tests = {}
    for key, cfg in recipes.items():
        total = np.zeros((20, 584, 565), np.float32)
        for fold in range(5):
            total += expanded_predictions(data, cfg, fold, models[key, fold],
                out/'probabilities'/cfg.name/f'fold{fold}', code_hash, test=True)/5
        tests[key] = total
    reference_test, candidate_test = gamma_fusions(tests)
    rt.make_submission(data, reference_test, metrics.CALIBRATION, out/'reference')
    rt.make_submission(data, candidate_test, metrics.CALIBRATION, out/'selected')
    np.savez_compressed(out/'selected_probabilities.npz', oof=candidate, test=candidate_test,
                        train_ids=data.ids, test_ids=data.test_ids)
    return save_selection('complete', accepted=True, submission_available=True)
