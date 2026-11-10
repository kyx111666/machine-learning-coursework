"""Official-data loading, retina geometry, pyramid rendering and label-synchronised augmentation."""
import math
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2

cv2.setNumThreads(1)

FROZEN_FOLDS = dict(s.split(':') for s in (
    '0001:1,0002:1,0003:3,0004:0,0005:0,0006:3,0007:2,0008:1,0009:4,0010:3,0011:2,0012:1,0013:1,0014:1,0015:0,'
    '0016:4,0017:2,0018:0,0019:3,0020:4,0021:0,0022:4,0023:2,0024:4,0025:0,0026:2,0027:4,0028:0,0029:3,0030:0,'
    '0031:0,0032:4,0033:1,0034:2,0035:0,0036:2,0037:1,0038:3,0039:2,0040:0,0041:2,0042:1,0043:4,0044:2,0045:3,'
    '0046:0,0047:3,0048:1,0049:2,0050:4,0051:1,0052:2,0053:3,0054:2,0055:2,0056:1,0057:0,0058:0,0059:2,0060:4,'
    '0061:3,0062:3,0063:4,0064:4,0065:1,0066:4,0067:3,0068:3,0069:1,0070:4,0071:0,0072:4,0073:4,0074:3,0075:1,'
    '0076:2,0077:2,0078:3,0079:1,0080:3').split(','))

SPLIT2_FOLDS = dict(s.split(':') for s in (
    '0001:0,0002:4,0003:1,0004:2,0005:4,0006:1,0007:4,0008:4,0009:4,0010:1,0011:1,0012:3,0013:2,0014:3,0015:4,'
    '0016:2,0017:3,0018:4,0019:2,0020:2,0021:2,0022:1,0023:3,0024:0,0025:0,0026:4,0027:1,0028:3,0029:3,0030:1,'
    '0031:3,0032:3,0033:2,0034:1,0035:1,0036:0,0037:2,0038:2,0039:0,0040:1,0041:3,0042:1,0043:4,0044:4,0045:3,'
    '0046:2,0047:3,0048:0,0049:2,0050:0,0051:1,0052:4,0053:4,0054:3,0055:2,0056:1,0057:2,0058:0,0059:1,0060:0,'
    '0061:0,0062:2,0063:4,0064:4,0065:4,0066:0,0067:3,0068:3,0069:3,0070:1,0071:0,0072:1,0073:2,0074:0,0075:4,'
    '0076:0,0077:0,0078:2,0079:0,0080:3').split(','))

def find_data_dir(hint=None):
    roots = [Path(hint)] if hint else [Path('/kaggle/input')]
    for root in roots:
        hits = sorted(root.rglob('fovea_localization_train_GT.csv')) if root.exists() else []
        if hits:
            return hits[0].parent
    raise FileNotFoundError('Attach the official 2026task3 competition data')

def stem_key(name):
    s = Path(str(name)).stem
    return str(int(s)) if s.isdigit() else s

def retina_circle(rgb):
    """Centre and radius of the bright field of view (largest connected component), original pixels."""
    h, w = rgb.shape[:2]
    f = 512 / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    n, _, stats, _ = cv2.connectedComponentsWithStats((small.max(2) > 20).astype(np.uint8), 8)
    x, y, bw, bh = stats[1 + np.argmax(stats[1:, 4]), :4] if n > 1 else (0, 0, small.shape[1], small.shape[0])
    sx, sy = w / small.shape[1], h / small.shape[0]
    return (x + bw / 2) * sx, (y + bh / 2) * sy, max(bw * sx, bh * sy) / 2

class Fundus:
    """One image kept as an anti-aliased pyramid (1/2, 1/4, 1/8) plus its field-of-view circle."""

    def __init__(self, name, rgb, xy=None, fold=-1):
        self.name, self.fold = name, fold
        self.group = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
        self.h, self.w = rgb.shape[:2]
        self.xy = None if xy is None else np.asarray(xy, float)
        self.cx, self.cy, self.r = retina_circle(rgb)
        self.levels, self.to_level, src = {}, {}, rgb
        for f in (2, 4, 8):
            lw, lh = max(1, round(self.w / f)), max(1, round(self.h / f))
            src = cv2.resize(src, (lw, lh), interpolation=cv2.INTER_AREA)
            fx, fy = self.w / lw, self.h / lh
            self.levels[f] = src

            self.to_level[f] = np.array([[1 / fx, 0, .5 / fx - .5], [0, 1 / fy, .5 / fy - .5], [0, 0, 1.]])

def _read(path):
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def load_dataset(data_dir, split=1):
    if split not in (1, 2):
        raise ValueError("Choose grouped split 1 or 2")
    folds = FROZEN_FOLDS if split == 1 else SPLIT2_FOLDS
    data_dir = Path(data_dir)
    labels = pd.read_csv(data_dir / 'fovea_localization_train_GT.csv', dtype={'data': str})
    sample = pd.read_csv(data_dir / 'sample_submission.csv')
    files = {stem_key(p.name): p for p in (data_dir / 'train').glob('*.jpg')}
    rows = [(files[stem_key(r.data)], (r.Fovea_X, r.Fovea_Y)) for r in labels.itertuples()]
    tests = sorted((data_dir / 'test').glob('*.jpg'))
    with ThreadPoolExecutor(8) as pool:
        train = list(pool.map(lambda r: Fundus(r[0].name, _read(r[0]), r[1], int(folds[r[0].stem])), rows))
        test = list(pool.map(lambda p: Fundus(p.name, _read(p)), tests))
    assert len(train) == 80 and len(test) == 20 and len(sample) == 40 and list(sample) == ['ImageID', 'value']
    assert all(np.isfinite(fd.xy).all() and 0 <= fd.xy[0] < fd.w and 0 <= fd.xy[1] < fd.h for fd in train)
    for fold in range(5):
        if not {fd.group for fd in train if fd.fold == fold}.isdisjoint(
                {fd.group for fd in train if fd.fold != fold}):
            raise ValueError("Identical training images cross validation folds")
    return train, test, sample

def render(fd, M, size):
    """Warp from the pyramid level just above the needed resolution; M maps original pixels to input pixels."""
    scale = math.sqrt(abs(np.linalg.det(M[:2, :2])))
    ok = [f for f in fd.levels if 1 / f >= .98 * scale]
    f = max(ok) if ok else min(fd.levels)
    A = M @ np.linalg.inv(fd.to_level[f])
    return cv2.warpAffine(fd.levels[f], A[:2], (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

def square(cx, cy, side, size):
    """Isotropic map of a side x side window centred on (cx, cy) to a size x size input."""
    s = size / side
    return np.array([[s, 0, (size - 1) / 2 - cx * s], [0, s, (size - 1) / 2 - cy * s], [0, 0, 1.]])

def coarse_base(fd, size, margin=.03, crop='fov'):
    """'fov': isotropic square around the field-of-view circle; 'image': whole image squeezed to size x size."""
    if crop == 'image':
        sx, sy = size / fd.w, size / fd.h
        return np.array([[sx, 0, .5 * sx - .5], [0, sy, .5 * sy - .5], [0, 0, 1.]])
    return square(fd.cx, fd.cy, 2 * fd.r * (1 + margin), size)

def hflip(size):
    return np.array([[-1, 0, size - 1], [0, 1, 0], [0, 0, 1.]])

def apply(M, xy):
    p = M @ np.array([xy[0], xy[1], 1.])
    return p[:2] / p[2]

def random_affine(size, rng, rot, scale, shift, flip=True):
    c = (size - 1) / 2
    s = math.exp(rng.uniform(-math.log(scale), math.log(scale)))
    A = np.vstack([cv2.getRotationMatrix2D((c, c), rng.uniform(-rot, rot), s), [0, 0, 1.]])
    A[:2, 2] += rng.uniform(-shift, shift, 2) * size
    return hflip(size) @ A if flip and rng.random() < .5 else A

def photometric(img, rng, p=1.):
    """Colour, contrast, gamma, saturation, blur and noise jitter inside the retina; labels untouched."""
    x = img.astype(np.float32) / 255
    if p <= 0:
        return x
    inside = (img.max(2, keepdims=True) > 10).astype(np.float32)
    x = x * rng.uniform(1 - .1 * p, 1 + .1 * p, 3).astype(np.float32)
    x = (x - .5) * rng.uniform(1 - .25 * p, 1 + .25 * p) + .5 + rng.uniform(-.1 * p, .1 * p)
    x = np.clip(x, 0, 1) ** rng.uniform(1 - .2 * p, 1 + .25 * p)
    g = x.mean(2, keepdims=True)
    x = g + (x - g) * rng.uniform(1 - .3 * p, 1 + .3 * p)
    if rng.random() < .3 * p:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(.3, 1.2))
    if rng.random() < .3 * p:
        x = x + rng.normal(0, .02 * p, x.shape).astype(np.float32)
    return (np.clip(x, 0, 1) * inside).astype(np.float32)

def random_erase(img, rng, p, frac=.25):
    """Blank one random rectangle (any position, the fovea included) so context must also be used."""
    if p <= 0 or rng.random() >= p:
        return img
    s = img.shape[0]
    h, w = (rng.uniform(.08, frac, 2) * s).astype(int)
    y, x = rng.integers(0, s - h), rng.integers(0, s - w)
    img = img.copy()
    img[y:y + h, x:x + w] = img[img.max(2) > 10].mean(0) if (img.max(2) > 10).any() else 0
    return img

def coarse_sample(fd, size, rng, aug):
    M = coarse_base(fd, size, crop=aug.get('crop', 'fov'))
    if aug['geo']:
        M = random_affine(size, rng, aug['rot'], aug['scale'], aug['shift'], aug.get('flip', True)) @ M
    img = random_erase(render(fd, M, size), rng, aug.get('erase', 0.))
    return photometric(img, rng, aug['photo']), apply(M, fd.xy)

def fine_sample(fd, size, crop, rng, aug):
    """Window around the label plus a random offset wider than typical coarse errors (avoids a centre bias)."""
    jitter = aug['jitter_px'] if 'jitter_px' in aug else aug['jitter'] * crop
    wide = aug['wide_px'] if 'wide_px' in aug else aug['wide'] * crop
    for _ in range(50):
        d = rng.normal(0, jitter, 2) if rng.random() < aug.get('p_narrow', .7) else rng.uniform(-wide, wide, 2)
        M = square(fd.xy[0] + d[0], fd.xy[1] + d[1], crop, size)
        if aug['geo']:
            M = random_affine(size, rng, aug['rot'], aug['scale'], 0, aug.get('flip', True)) @ M
        t = apply(M, fd.xy)
        if (t > .08 * size).all() and (t < .92 * size).all():
            break
    return photometric(render(fd, M, size), rng, aug['photo']), t


def dataset_fingerprint(root):
    """Bind every pipeline's outputs to the same official image files and label/template CSVs."""
    root = Path(root)
    digest = hashlib.sha256()
    files = [root / 'fovea_localization_train_GT.csv', root / 'sample_submission.csv']
    files += sorted((root / 'train').glob('*.jpg')) + sorted((root / 'test').glob('*.jpg'))
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
