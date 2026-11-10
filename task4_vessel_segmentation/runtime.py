"""Official-data loading, patch U-Nets, baseline training and exact competition output protocol."""
from __future__ import annotations
import ast
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

VERSION = 'vessel-segmentation'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cv2.setNumThreads(0)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown_table(frame, index=False):
    frame = frame.reset_index() if index else frame
    rows = [[str(x).replace('|', '\\|') for x in frame.columns]]
    rows += [['---'] * len(frame.columns)]
    for row in frame.itertuples(index=False, name=None):
        rows.append([f'{x:.6g}' if isinstance(x, float) else str(x).replace('|', '\\|') for x in row])
    return '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows)


def find_data(root=None):
    if root is not None:
        root = Path(root)
        candidates = [root] if (root / 'sample_submission.csv').exists() else list(root.rglob('sample_submission.csv'))
        candidates = [x if x.is_dir() else x.parent for x in candidates]
    else:
        candidates = [p.parent for p in Path('/kaggle/input').rglob('sample_submission.csv')
                      if (p.parent / 'segmentation_to_csv.py').exists()
                      and (p.parent / 'train/image').is_dir()]
    assert len(candidates) == 1, f'须仅挂载本比赛官方数据；找到 {candidates}。可设置 DATA_ROOT。'
    root = candidates[0]
    assert all((root / p).exists() for p in ['train/image', 'train/label', 'test/image', 'segmentation_to_csv.py'])
    return root


class OfficialData:
    def __init__(self, root, out):
        self.root, self.out = Path(root), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sample = pd.read_csv(self.root / 'sample_submission.csv', dtype={'Id': str}, keep_default_na=False)
        assert self.sample.columns.tolist() == ['Id', 'Predicted']
        self.ids = sorted([p.stem for p in (self.root / 'train/image').glob('*.jpg')], key=int)
        self.test_ids = self.sample.Id.tolist()
        assert len(self.ids) == len(self.test_ids) == 20
        assert len(set(self.ids)) == len(set(self.test_ids)) == 20
        self.images, self.labels, self.proxy, self.test_images, self.rows = [], [], [], [], []
        pixel_hashes = []
        for split, ids in [('train', self.ids), ('test', self.test_ids)]:
            for name in ids:
                p = self.root / split / 'image' / f'{name}.jpg'
                a = np.asarray(Image.open(p).convert('RGB')).copy()
                assert a.shape == (584, 565, 3), (p, a.shape)
                pixel_hash = hashlib.sha256(a.tobytes()).hexdigest()
                pixel_hashes.append(pixel_hash)
                row = dict(split=split, id=name, height=a.shape[0], width=a.shape[1],
                           sha256=digest_file(p), pixel_sha256=pixel_hash)
                if split == 'train':
                    lp = self.root / 'train/label' / f'{name}.jpg'
                    im = Image.open(lp).convert('L')
                    label = np.asarray(im)
                    assert label.shape == a.shape[:2] and set(np.unique(label)).issubset({0, 255})
                    y = (label > 127).astype(np.uint8)
                    assert .01 < y.mean() < .3, '标签约定变化：停止并重新核对前景。'
                    proxy = (np.asarray(im.resize((512, 512), Image.Resampling.BILINEAR)) > 0).astype(np.uint8)
                    self.images.append(a)
                    self.labels.append(y)
                    self.proxy.append(proxy)
                    row.update(label_sha256=digest_file(lp), vessel_fraction=float(y.mean()),
                               proxy_fraction=float(proxy.mean()))
                else:
                    self.test_images.append(a)
                self.rows.append(row)
        assert len(pixel_hashes) == len(set(pixel_hashes)), '发现重复原图，需要按重复组重新划分。'
        self.folds = np.full(20, -1, dtype=int)
        for fold, (_, va) in enumerate(KFold(5, shuffle=True, random_state=3407).split(self.ids)):
            self.folds[va] = fold
        assert np.all(np.bincount(self.folds) == 4)
        pd.DataFrame(self.rows).to_csv(self.out / 'data_audit.csv', index=False)
        pd.DataFrame({'Id': self.ids, 'fold': self.folds}).to_csv(self.out / 'folds.csv', index=False)
        # Dataset hash includes labels, sample order and the official converter.
        self.fingerprint = hashlib.sha256(json.dumps(self.rows, sort_keys=True).encode()
            + (self.root / 'sample_submission.csv').read_bytes()
            + (self.root / 'segmentation_to_csv.py').read_bytes()).hexdigest()
        self.cache = {}
        print('官方数据：20 train / 20 test；584×565；白色血管；按原图固定五折。')

    def prepared(self, mode, test=False):
        key = (mode, test)
        if key not in self.cache:
            self.cache[key] = [preprocess(x, mode) for x in (self.test_images if test else self.images)]
        return self.cache[key]



def preprocess(rgb, mode):
    if mode == 'rgb':
        return rgb.copy()
    g = rgb[..., 1]
    if mode == 'green_raw':
        return np.repeat(g[..., None], 3, axis=2)
    assert mode == 'green'
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    contrast = np.clip(4 * g.astype(np.float32) - 4 * cv2.GaussianBlur(g, (0, 0), 8).astype(np.float32) + 128, 0, 255).astype(np.uint8)
    return np.stack([g, clahe, contrast], axis=-1)


@dataclass(frozen=True)
class Recipe:
    name: str = 'resunet_green'
    arch: str = 'resunet'
    mode: str = 'green'
    width: int = 24
    patch: int = 192
    lr: float = 8e-4
    wd: float = 1e-4
    dice_weight: float = .5
    augmentation: bool = True
    multiscale: bool = True
    ema: bool = True
    context: bool = False
    seed: int = 20261031


class PatchDataset(Dataset):
    def __init__(self, images, labels, indices, recipe, count, seed):
        self.x, self.y, self.indices, self.cfg = images, labels, list(indices), recipe
        self.count, self.seed = count, seed
        # Purely uniform crops avoid changing the class prior through label-conditioned sampling.

    def __len__(self): return self.count

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        i = self.indices[int(rng.integers(len(self.indices)))]
        x, y = self.x[i], self.y[i]
        p = self.cfg.patch
        scale = float(rng.choice([.8, 1., 1.25])) if self.cfg.multiscale else 1.
        span = p * scale
        cx = float(rng.uniform(span / 2, x.shape[1] - span / 2))
        cy = float(rng.uniform(span / 2, x.shape[0] - span / 2))
        angle = float(rng.uniform(-180, 180)) if self.cfg.augmentation else 0.
        # WARP_INVERSE_MAP maps each output patch pixel into the source image.
        a = math.radians(angle)
        mat = np.array([[scale * math.cos(a), -scale * math.sin(a), 0],
                        [scale * math.sin(a), scale * math.cos(a), 0]], dtype=np.float32)
        mat[:, 2] = [cx, cy] - mat[:, :2] @ np.array([(p - 1) / 2, (p - 1) / 2])
        x = cv2.warpAffine(x, mat, (p, p), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REFLECT_101)
        y = cv2.warpAffine(y, mat, (p, p), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REFLECT_101)
        if self.cfg.augmentation:
            if rng.random() < .5: x, y = x[:, ::-1], y[:, ::-1]
            if rng.random() < .5: x, y = x[::-1], y[::-1]
        x = x.astype(np.float32) / 255.
        if self.cfg.augmentation:
            x = np.clip(x * rng.uniform(.85, 1.15) + rng.uniform(-.06, .06), 0, 1)
            if rng.random() < .25:
                x = np.clip(x + rng.normal(0, .012, x.shape).astype(np.float32), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1) * 2 - 1)), torch.from_numpy(np.ascontiguousarray(y[None])).float()


class Block(nn.Module):
    def __init__(self, cin, cout, residual):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
                                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.skip = (nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)) if residual else None

    def forward(self, x):
        y = self.body(x)
        return F.silu(y if self.skip is None else y + self.skip(x), inplace=True)


class VesselUNet(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        assert recipe.arch in {'unet', 'resunet'}
        w, residual = recipe.width, recipe.arch == 'resunet'
        self.enc = nn.ModuleList([Block(3, w, residual), Block(w, 2*w, residual), Block(2*w, 4*w, residual),
                                  Block(4*w, 8*w, residual), Block(8*w, 16*w, residual)])
        self.context = nn.ModuleList([nn.Conv2d(16*w, 16*w, 3, padding=d, dilation=d, groups=16*w)
                                      for d in [1, 2, 4]]) if recipe.context else None
        self.up = nn.ModuleList([nn.Conv2d(16*w, 8*w, 1), nn.Conv2d(8*w, 4*w, 1), nn.Conv2d(4*w, 2*w, 1), nn.Conv2d(2*w, w, 1)])
        self.dec = nn.ModuleList([Block(16*w, 8*w, residual), Block(8*w, 4*w, residual),
                                  Block(4*w, 2*w, residual), Block(2*w, w, residual)])
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        skips = []
        for j, block in enumerate(self.enc):
            if j: x = F.max_pool2d(x, 2)
            x = block(x); skips.append(x)
        if self.context is not None:
            x = x + sum(layer(x) for layer in self.context) / len(self.context)
        for j, (up, block) in enumerate(zip(self.up, self.dec)):
            s = skips[-2-j]
            x = up(F.interpolate(x, size=s.shape[-2:], mode='bilinear', align_corners=False))
            x = block(torch.cat([x, s], dim=1))
        return self.head(x)


def loss_fn(logits, target, dice_weight):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    dice = 1 - ((2 * (p * target).sum((1, 2, 3)) + 1) /
                (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()
    return (1 - dice_weight) * bce + dice_weight * dice


def autocast():
    return torch.autocast('cuda', dtype=torch.float16) if DEVICE.type == 'cuda' else contextlib.nullcontext()


@torch.no_grad()
def predict(model, image, tta=False):
    model.eval()
    x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None].to(DEVICE) / 127.5 - 1
    h, w = x.shape[-2:]
    x = F.pad(x, (0, (-w) % 16, 0, (-h) % 16), mode='reflect')
    variants = [(), (-1,), (-2,), (-2, -1)] if tta else [()]
    probs = []
    for dims in variants:
        z = x.flip(dims) if dims else x
        with autocast(): p = model(z).float().sigmoid()
        if dims: p = p.flip(dims)
        probs.append(p[0, 0, :h, :w].cpu().numpy())
    return np.mean(probs, axis=0).astype(np.float32)


def dice_score(y, m):
    y, m = np.asarray(y, bool), np.asarray(m, bool)
    return float((2 * np.logical_and(y, m).sum() + 1e-8) / (y.sum() + m.sum() + 1e-8))


def proxy_prob(p):
    # Resize probabilities once, then binarize on the exact submission grid.
    return cv2.resize(p.astype(np.float32), (512, 512), interpolation=cv2.INTER_LINEAR)


def validate(model, data, indices, mode):
    native, proxy = [], []
    images = data.prepared(mode)
    for i in indices:
        p = predict(model, images[i])
        native.append(dice_score(data.labels[i], p >= .5))
        proxy.append(dice_score(data.proxy[i], output_mask(p, .5, 0, 'native_then_official')))
    return float(np.mean(native)), float(np.mean(proxy))


def signature(data, recipe, fold, steps, batch, code_hash, eval_every):
    spec = dict(version=VERSION, data=data.fingerprint, recipe=asdict(recipe), fold=fold,
                steps=steps, batch=batch, code_hash=code_hash, eval_every=eval_every,
                torch=torch.__version__, device=DEVICE.type)
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(), spec


def train_fold(data, recipe, fold, steps, batch, out, code_hash, workers=2, eval_every=250):
    """Checkpoints are reusable only for an exact matching, completed run."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sig, spec = signature(data, recipe, fold, steps, batch, code_hash, eval_every)
    path = out / 'best.pt'
    done = out / 'completed.json'
    if done.exists():
        meta = json.loads(done.read_text())
        assert meta['signature'] == sig and path.exists(), f'输出目录含不同实验：{out}，请换目录。'
        return path, meta
    if (out / 'spec.json').exists():
        assert json.loads((out / 'spec.json').read_text())['signature'] == sig, f'禁止混用旧实验目录：{out}'
    write_json(out / 'spec.json', dict(signature=sig, spec=spec))
    tr = np.flatnonzero(data.folds != fold); va = np.flatnonzero(data.folds == fold)
    assert not set(tr) & set(va) and len(tr) == 16 and len(va) == 4
    seed_all(recipe.seed)
    model = VesselUNet(recipe).to(DEVICE)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    ds = PatchDataset(data.prepared(recipe.mode), data.labels, tr, recipe, steps * batch, recipe.seed + fold * 1000003)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                        pin_memory=DEVICE.type == 'cuda', persistent_workers=workers > 0)
    optim = torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.wd)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
    warmup = min(100, max(1, steps // 10))
    history, best, best_step, start = [], -1., 0, time.time()
    ema_state = ema.state_dict()
    nparams = sum(p.numel() for p in model.parameters())
    print(f'{recipe.name} fold={fold} steps={steps} params={nparams:,}', flush=True)
    for step, (x, y) in enumerate(loader, 1):
        model.train()
        frac = max(0., (step - warmup) / max(1, steps - warmup))
        lr = recipe.lr * (step / warmup if step <= warmup else .05 + .95 * .5 * (1 + math.cos(math.pi * frac)))
        for group in optim.param_groups: group['lr'] = lr
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        with autocast(): loss = loss_fn(model(x), y, recipe.dice_weight)
        assert torch.isfinite(loss), '训练 loss 非有限值。'
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(optim); scaler.update()
        # Copy BN running statistics; averaging them would add a second lag.
        decay = min(.995, (step - 1) / (step + 9))
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if value.is_floating_point() and 'running_' not in key:
                    ema_state[key].mul_(decay).add_(value, alpha=1-decay)
                else: ema_state[key].copy_(value)
        if step % eval_every == 0 or step == steps:
            chosen = ema if recipe.ema else model
            native, proxy = validate(chosen, data, va, recipe.mode)
            row = dict(step=step, loss=float(loss.detach().cpu()), lr=lr, native_dice=native,
                       proxy_dice=proxy, elapsed_seconds=time.time()-start)
            history.append(row)
            pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
            print(f'  {step:5d} loss={row["loss"]:.4f} native={native:.5f} proxy={proxy:.5f} seconds={row["elapsed_seconds"]:.0f}', flush=True)
            if proxy > best:
                best, best_step = proxy, step
                payload = dict(signature=sig, recipe=asdict(recipe), fold=fold, step=step,
                               model={k: v.detach().cpu() for k, v in chosen.state_dict().items()},
                               raw={k: v.detach().cpu() for k, v in model.state_dict().items()},
                               spec=spec)
                tmp = path.with_suffix('.tmp'); torch.save(payload, tmp); tmp.replace(path)
    meta = dict(signature=sig, best_proxy_dice=best, best_step=best_step, seconds=time.time()-start,
                parameters=nparams, recipe=asdict(recipe), fold=int(fold))
    write_json(done, meta)
    del loader, model, ema, optim
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    return path, meta


def load_model(path, raw=False):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = VesselUNet(Recipe(**ck['recipe'])).to(DEVICE)
    model.load_state_dict(ck['raw' if raw else 'model'])
    return model.eval(), ck


def rle_encode(mask):
    """Exactly the official C-order, one-based encoding of nonzero pixels."""
    a = np.asarray(mask, dtype=np.uint8).ravel(order='C')
    padded = np.r_[0, a, 0]
    edges = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    edges[1::2] -= edges[::2]
    return ' '.join(map(str, edges))


def rle_decode(s, shape=(512, 512)):
    a = np.zeros(int(np.prod(shape)), np.uint8)
    words = np.array([int(v) for v in str(s).split()], dtype=np.int64)
    assert len(words) % 2 == 0
    prev = 0
    for start, length in words.reshape(-1, 2):
        assert length > 0 and start >= 1 and start - 1 >= prev and start - 1 + length <= a.size
        a[start-1:start-1+length] = 1; prev = start-1+length
    return a.reshape(shape)


def verify_official_encoder(root, out):
    """Execute official function definitions, without its main filesystem operations."""
    import torchvision
    source = (Path(root) / 'segmentation_to_csv.py').read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    defs = ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef)], type_ignores=[])
    scope = dict(np=np, pd=pd, Image=Image, torchvision=torchvision, os=os)
    exec(compile(defs, 'official_segmentation_to_csv.py', 'exec'), scope)
    assert 'turn_to_str' in scope, '官方转换脚本发生变化，请重新核对。'
    out = Path(out) / 'encoder_check'; out.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(7)
    checks = [np.zeros((512, 512), np.uint8), np.ones((512, 512), np.uint8),
              (rng.random((512, 512)) < .03).astype(np.uint8)]
    p = np.zeros((512, 512), np.uint8); p[0, 0] = p[-1, -1] = p[13, 77] = 1; checks.append(p)
    paths = []
    for i, a in enumerate(checks):
        path = out / f'{i}.png'; Image.fromarray(a * 255).save(path); paths.append(str(path))
    reference = scope['turn_to_str'](paths)
    for a, expected in zip(checks, reference):
        assert rle_encode(a) == expected
        assert np.array_equal(rle_decode(expected), a)
    write_json(out / 'verified.json', dict(converter_sha256=digest_file(Path(root) / 'segmentation_to_csv.py'),
                                          cases=len(checks), order='C', first_index=1, positive='white/nonzero', shape=[512,512]))
    print('已与官方 turn_to_str 逐字比对：空图、满图、随机图、首尾像素全部一致。')


def component_filter(mask, min_size):
    if min_size == 0: return np.asarray(mask, bool)
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    counts = np.bincount(labels.ravel()); keep = counts >= min_size; keep[0] = False
    return keep[labels]


def output_mask(prob, threshold, min_size=0, output_protocol='probability512', shape=(512,512)):
    h,w = shape
    if output_protocol == 'native_then_official':
        native = Image.fromarray((np.asarray(prob) >= threshold).astype(np.uint8)*255)
        mask = np.asarray(native.resize((w,h),Image.Resampling.BILINEAR)) > 0
    else:
        assert output_protocol == 'probability512'
        mask = cv2.resize(np.asarray(prob,np.float32),(w,h),interpolation=cv2.INTER_LINEAR) >= threshold
    return component_filter(mask, min_size)


def score_at(probs, targets, threshold, min_size=0, output_protocol='probability512'):
    return np.array([dice_score(y, output_mask(p,threshold,min_size,output_protocol,y.shape)) for p, y in zip(probs, targets)])


def screening_recipes():
    base = Recipe()
    return [
        replace(base, name='baseline_unet_rgb_bce', arch='unet', mode='rgb', dice_weight=0., augmentation=False, multiscale=False, ema=False),
        base,
        replace(base, name='model_plain_unet', arch='unet'),
        replace(base, name='model_context', context=True),
        replace(base, name='input_rgb', mode='rgb'),
        replace(base, name='input_green_raw', mode='green_raw'),
        replace(base, name='ablation_no_augmentation', augmentation=False),
        replace(base, name='ablation_no_multiscale', multiscale=False),
        replace(base, name='ablation_bce_only', dice_weight=0.),
        replace(base, name='lr_4e4', lr=4e-4),
        replace(base, name='lr_16e4', lr=1.6e-3),
        replace(base, name='dice_weight_025', dice_weight=.25),
        replace(base, name='dice_weight_075', dice_weight=.75),
        replace(base, name='patch_128', patch=128),
        replace(base, name='patch_256', patch=256),
        replace(base, name='width_32', width=32),
    ]


def save_paired_analysis(data, out, results):
    df = pd.DataFrame(results).sort_values('proxy_dice', ascending=False)
    df.to_csv(out / 'screening.csv', index=False)
    base = df[df.name == 'resunet_green'].iloc[0]
    pairs = []
    for name, factor in [('model_plain_unet', 'residual blocks'), ('input_green_raw', 'CLAHE + local contrast'),
                         ('input_rgb', 'green feature representation'), ('ablation_no_augmentation', 'geometric/photometric augmentation'),
                         ('ablation_no_multiscale', 'multiscale crops'), ('ablation_bce_only', 'Dice loss')]:
        row = df[df.name == name].iloc[0]
        pairs.append(dict(factor=factor, full_recipe=float(base.proxy_dice), removed=float(row.proxy_dice),
                          difference=float(base.proxy_dice-row.proxy_dice), interpretation='single split screening; not causal proof'))
    pd.DataFrame(pairs).to_csv(out / 'ablation.csv', index=False)
    return df


def run_analysis(data, out, code_hash, steps=1000, confirm_steps=2000, batch=12, workers=2):
    out = Path(out); out.mkdir(exist_ok=True, parents=True)
    recipes = screening_recipes()
    results = []
    for cfg in recipes:
        path, meta = train_fold(data, cfg, 0, steps, batch, out / cfg.name / 'fold0', code_hash, workers)
        model, ck = load_model(path)
        va = np.flatnonzero(data.folds == 0)
        native, proxy = validate(model, data, va, cfg.mode)
        results.append(dict(**asdict(cfg), native_dice=native, proxy_dice=proxy,
                            seconds=meta['seconds'], parameters=meta['parameters'], best_step=meta['best_step']))
        del model
        pd.DataFrame(results).to_csv(out / 'screening_partial.csv', index=False)
    df = save_paired_analysis(data, out, results)
    # EMA and TTA are evaluated on the SAME training trajectory/checkpoint.
    path = out / 'resunet_green/fold0/best.pt'
    paired = []
    for raw in [True, False]:
        model, _ = load_model(path, raw=raw)
        for tta in [False, True]:
            for i in np.flatnonzero(data.folds == 0):
                p = predict(model, data.prepared('green')[i], tta)
                paired.append(dict(Id=data.ids[i], ema=not raw, tta=tta,
                    native_dice=dice_score(data.labels[i], p >= .5),
                    proxy_dice=dice_score(data.proxy[i], output_mask(p,.5,0,'native_then_official'))))
        del model
    pd.DataFrame(paired).to_csv(out / 'paired_ema_tta.csv', index=False)
    # New images for confirmation, and longer equal budgets for shortlisted models.
    shortlist = [next(c for c in recipes if c.name == name) for name in df.head(3).name]
    confirmed = []
    for cfg in shortlist:
        for fold in [1, 2]:
            path, meta = train_fold(data, cfg, fold, confirm_steps, batch,
                                    out / 'confirmation' / cfg.name / f'fold{fold}', code_hash, workers)
            model, _ = load_model(path)
            scores = []
            for i in np.flatnonzero(data.folds == fold):
                p = predict(model, data.prepared(cfg.mode)[i], True)
                scores.append(dice_score(data.proxy[i], output_mask(p,.5,0,'native_then_official')))
            confirmed.append(dict(name=cfg.name, fold=fold, dice=float(np.mean(scores)), seconds=meta['seconds']))
            del model
    conf = pd.DataFrame(confirmed); conf.to_csv(out / 'confirmation.csv', index=False)
    ranking = conf.groupby('name').dice.agg(['mean','std']).sort_values('mean', ascending=False)
    ranking.to_csv(out / 'confirmation_ranking.csv')
    winner = next(c for c in recipes if c.name == ranking.index[0])
    # Diversity candidate is confirmed rather than automatically assumed beneficial.
    ranked_recipes = [next(c for c in recipes if c.name == name) for name in ranking.index]
    second = next((c for c in ranked_recipes if c.mode != winner.mode), None)
    if second is None:
        second = replace(winner, name=winner.name+'_seed2', seed=20261101)
    selected = [asdict(winner), asdict(second)]
    write_json(out / 'selected_recipes.json', selected)
    print('筛选推荐配置已保存至 selected_recipes.json；正式入口使用固定的最终方案。')
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    # All these are model-selection results, not an unbiased final score estimate.
    text = f'''# 课设实验记录（由实际运行生成）\n\n官方数据20张训练图、20张测试图。所有模型从零训练；按原图固定五折。\n\n共比较{len(recipes)}种设定，筛选预算每项{steps}步、batch={batch}。\n短名单在另外两折以每项{confirm_steps}步复核。筛选和复核分数参与了选模，不能作为无偏测试成绩。\n\n官方标签白色表示血管。原图标签为584×565，官方脚本先双线性缩放至512×512再取非零像素，正例边缘会扩展。\n本实验同时记录原始标签Dice和转换协议代理Dice；隐藏评分标签未公开，代理并不等于已验证的Kaggle分数。\n\n## 模型比较\n\n{markdown_table(df[['name','native_dice','proxy_dice','parameters','seconds']])}\n\n## 独立折复核\n\n{markdown_table(ranking,index=True)}\n\n消融结果见ablation.csv；同轨迹EMA/TTA见paired_ema_tta.csv；学习曲线位于各实验history.csv。\n差值来自有限的训练图；正差值只支持当前划分和预算下的有效性。\n'''
    (out / '实验报告.md').write_text(text, encoding='utf8')
    return selected


def make_submission(data, probs, calibration, out):
    probs = np.asarray(probs)
    if (probs.shape != (20, 584, 565) or not np.isfinite(probs).all()
            or ((probs < 0) | (probs > 1)).any()):
        raise ValueError('Expected twenty finite native-resolution probability maps in [0, 1]')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    masks = [output_mask(p,calibration['threshold'],calibration['min_size'],
                          calibration.get('output_protocol','probability512')) for p in probs]
    df = data.sample.copy()
    df['Predicted'] = [rle_encode(m) for m in masks]
    assert df.Id.tolist() == data.test_ids and len(df) == 20 and not df.Predicted.isna().any()
    pred_dir = out / 'prediction_png'; pred_dir.mkdir(exist_ok=True)
    for name, mask, encoded in zip(df.Id, masks, df.Predicted):
        assert np.array_equal(rle_decode(encoded), mask)
        Image.fromarray(mask.astype(np.uint8) * 255).save(pred_dir / f'{name}.png')
    df.to_csv(out / 'submission.csv', index=False)
    # Validate real predictions against the provided converter, not only synthetic patterns.
    import torchvision
    tree = ast.parse((data.root / 'segmentation_to_csv.py').read_text(encoding='utf-8-sig'))
    scope = dict(np=np, pd=pd, Image=Image, torchvision=torchvision, os=os)
    exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef)], type_ignores=[]),
                 'official_converter', 'exec'), scope)
    reference = scope['turn_to_str']([str(pred_dir / f'{name}.png') for name in df.Id])
    assert reference == df.Predicted.tolist(), '真实预测与官方编码不一致。'
    print('submission.csv：20行，Id顺序、512×512、RLE往返、官方脚本逐字比对全部通过。')
    return df


def bundle_outputs(out, destination):
    """Small analysis bundle; large reproducibility checkpoints remain in Notebook Output."""
    import zipfile
    out=Path(out); destination=Path(destination)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.json','.csv','.md','.png'}:
                z.write(p,p.relative_to(out))
    print('分析包:',destination,'；模型权重另保存在Output/models。')
