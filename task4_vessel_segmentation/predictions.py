"""Fold-aligned probability caches, fixed postprocessing and per-image validation metrics."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import runtime as rt

CALIBRATION = dict(threshold=.425, min_size=8, output_protocol='native_then_official')

def image_metrics(data, probs, indices, route):
    rows = []
    for i, prob in zip(indices, probs):
        mask = rt.output_mask(prob, **CALIBRATION)
        y = data.proxy[i].astype(bool)
        tp = np.logical_and(y, mask).sum()
        rows.append(dict(route=route, Id=data.ids[i], fold=int(data.folds[i]),
            dice=rt.dice_score(y, mask), native_dice=rt.dice_score(data.labels[i], prob >= .425),
            precision=float(tp/max(1, mask.sum())), recall=float(tp/max(1, y.sum()))))
    return rows

def expanded_predictions(data, recipe, fold, checkpoint, out, code_hash, test=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    indices = list(range(20)) if test else np.flatnonzero(data.folds == fold).tolist()
    ids = data.test_ids if test else [data.ids[i] for i in indices]
    spec = dict(checkpoint_sha256=rt.digest_file(checkpoint), data=data.fingerprint,
                recipe=asdict(recipe), fold=fold, test=test, code_hash=code_hash,
                tta='original_whole_flip4', torch=torch.__version__, numpy=np.__version__,
                opencv=rt.cv2.__version__, device=str(rt.DEVICE))
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    path = out/('test.npz' if test else 'oof.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            assert cache['signature'].item() == signature and cache['ids'].tolist() == ids
            probabilities = cache['probs'].copy()
    else:
        model, record = rt.load_model(checkpoint)
        assert record['recipe'] == asdict(recipe) and record['fold'] == fold
        assert record['spec']['data'] == data.fingerprint and record['spec']['code_hash'] == code_hash
        assert record['signature'] == hashlib.sha256(json.dumps(record['spec'], sort_keys=True).encode()).hexdigest()
        images = data.prepared(recipe.mode, test=test)
        probabilities = np.stack([rt.predict(model, images[i], True) for i in indices])
        del model
        torch.cuda.empty_cache()
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, probs=probabilities, ids=ids, signature=signature)
        path.with_suffix('.tmp').replace(path)
    assert probabilities.shape == (len(ids), 584, 565) and probabilities.dtype == np.float32
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    rt.write_json(out/('test.json' if test else 'oof.json'), dict(signature=signature, spec=spec))
    return probabilities


def bundle_expanded_outputs(out, destination):
    import zipfile
    rt.bundle_outputs(out, destination)
    probability_path = Path(out)/'oof_route_probabilities.npz'
    if probability_path.exists():
        with zipfile.ZipFile(destination, 'a', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(probability_path, probability_path.name)
