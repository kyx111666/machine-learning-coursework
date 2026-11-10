"""Re-run inference of saved fold models with other decoders / TTA; nothing is retrained, OOF stays out-of-fold."""
import json
from pathlib import Path
import numpy as np
import torch
from data import square, coarse_base
from training import build_model, predict

def load_fold_model(run_dir, k, device):
    cfg = json.loads((Path(run_dir) / 'config.json').read_text())
    model = build_model(cfg, pretrained=False)
    state = torch.load(Path(run_dir) / f'model_fold{k}.pt', map_location='cpu')
    model.load_state_dict({n: v.float() if v.is_floating_point() else v for n, v in state.items()})
    return model.to(device).eval(), cfg

def _mean(dicts):
    return {k: np.mean([d[k] for d in dicts], 0) for k in dicts[0]}

def fine_views(model, items, centers, crop, size, device, decoders, scales=(1.,), shifts=((0., 0.),)):
    """Average over crop scales and centre shifts (fractions of the crop) of flip-TTA predictions."""
    views = []
    for s in scales:
        for dx, dy in shifts:
            base = (lambda fd, s=s, dx=dx, dy=dy:
                    square(centers[fd.name][0] + dx * crop, centers[fd.name][1] + dy * crop, crop * s, size))
            views.append(predict(model, items, base, size, device, decoders=decoders))
    return _mean(views)

def fine_infer(model, items, centers, cfg, device, decoders, iters=2, center_key=('global', 'flip'), **tta):
    out, cur = {}, dict(centers)
    for it in range(iters):
        res = fine_views(model, items, cur, cfg['crop'], cfg['size'], device, decoders, **tta)
        out.update({(d, m, f'it{it + 1}'): v for (d, m), v in res.items()})
        cur = dict(zip([fd.name for fd in items], res[center_key]))
    return out

def coarse_infer(model, items, cfg, device, decoders, scales=(1.,)):
    crop = cfg['aug'].get('crop', 'fov')
    views = [predict(model, items, lambda fd, s=s: coarse_base(fd, cfg['size'], margin=.03 + (s - 1), crop=crop),
                     cfg['size'], device, decoders=decoders) for s in scales]
    return _mean(views)
