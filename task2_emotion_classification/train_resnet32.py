"""Grouped five-fold scratch ResNet32 training at 48 pixels, with two fixed model seeds."""
from cli import configure
ARGS = configure(['cv_resnet32', 'cv_resnet32_s2027'], 'cv_resnet32')
RUN = 'cv_resnet32'
SESSION_MINUTES = 460
PROBE_ONLY = False
MIN_FREE_GB = 3.0
DATA_ROOT = ARGS.data_root
RESUME_ROOT = ARGS.resume
SEED = 2026
N_FOLDS = 5
FOLD_IDS = [0, 1, 2, 3, 4]

print('TASK2 训练启动，加载依赖……', flush=True)
import os, sys, json, math, time, copy, gc, random, hashlib, shutil, zipfile, platform, subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
try:
    from IPython.display import display
except ImportError:
    display = print

RUN = os.environ.get('TASK2_RUN', RUN)
SMOKE = os.environ.get('TASK2_SMOKE') == '1'
PROBE_ONLY = os.environ.get('TASK2_PROBE_ONLY', '1' if PROBE_ONLY else '0') == '1'
SESSION_MINUTES = float(os.environ.get('TASK2_MINUTES', SESSION_MINUTES))
INPUT_ROOT = Path(os.environ.get('TASK2_INPUT', '/kaggle/input'))
OUTPUT_ROOT = Path(os.environ.get('TASK2_OUTPUT', '/kaggle/working/task2')) / RUN
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
LABEL_NAMES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
ALIASES = {'angry': 0, 'anger': 0, 'fear': 1, 'fearful': 1, 'happy': 2, 'happiness': 2,
           'sad': 3, 'sadness': 3, 'surprise': 4, 'surprised': 4, 'neutral': 5,
           '愤怒': 0, '恐惧': 1, '快乐': 2, '悲伤': 3, '惊讶': 4, '中性': 5}
EXPECTED_TEST_ROWS = None if SMOKE else 7065
SUBMISSION_LABEL_COLUMN = 'Emotion'
NGPU = torch.cuda.device_count()
DEVICE = torch.device('cuda:0' if NGPU else 'cpu')
if DEVICE.type != 'cuda' and not SMOKE:
    raise RuntimeError('正式运行需要 GPU：Settings → Accelerator 选择 GPU 后重新运行。')
AMP = DEVICE.type == 'cuda'
if AMP:
    torch.cuda.set_device(0)
START_TIME = time.time()
DO_TEST = RUN.startswith('cv_')

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if AMP: torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def host_mem():
    """读取进程RSS和容器可用内存；非Linux返回NaN，避免把宿主剩余RAM当容器额度。"""
    try:
        rss = next(float(l.split()[1]) / 1048576 for l in Path('/proc/self/status').read_text().splitlines()
                   if l.startswith('VmRSS'))
        avail = next(float(l.split()[1]) / 1048576 for l in Path('/proc/meminfo').read_text().splitlines()
                     if l.startswith('MemAvailable'))
        for limit_path,current_path in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                                        ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
            try:
                limit = Path(limit_path).read_text().strip()
                if limit!='max':
                    remain = (int(limit)-int(Path(current_path).read_text())) / 2**30
                    avail = min(avail,max(0.0,remain))
            except (OSError,ValueError): pass
        return rss,avail
    except (OSError,StopIteration,ValueError):
        return float('nan'),float('nan')

def gpu_peak_gb():
    return torch.cuda.max_memory_allocated() / 2 ** 30 if AMP else 0.0

def mem_tag():
    rss, avail = host_mem()
    return f'rss={rss:.1f}G 可用={avail:.1f}G 显存峰值={gpu_peak_gb():.1f}G'

ENV = {'pipeline': 'task2', 'run': RUN, 'smoke': SMOKE, 'python': platform.python_version(),
       'torch': torch.__version__, 'model_implementation': 'local SmallResNet32', 'device': str(DEVICE),
       'n_gpu_visible': NGPU, 'parallel': 'single-gpu (DataParallel removed)',
       'gpu': torch.cuda.get_device_name(0) if AMP else None, 'host_mem': host_mem()}
save_json(OUTPUT_ROOT / 'environment.json', ENV)
print(ENV, flush=True)

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def image_files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                  and '__MACOSX' not in p.parts and not p.name.startswith('._'))

def discover_data_root():
    if DATA_ROOT is not None:
        return Path(DATA_ROOT)
    cands = sorted({p for p in INPUT_ROOT.rglob('*') if p.is_dir() and '2026task2' in p.name.lower()})
    cands = [p for p in cands if not any(q != p and q in p.parents for q in cands)]
    if len(cands) != 1:
        raise RuntimeError(f'不能唯一确定比赛数据目录：{cands}。请设置 DATA_ROOT。')
    return cands[0]

def extract_if_needed(root):
    if image_files(root): return root
    archives = sorted(root.rglob('*.zip'))
    if not archives: raise RuntimeError(f'{root} 中没有图像或 ZIP，请添加官方比赛数据。')
    dest = OUTPUT_ROOT.parent / '_extracted'
    for k, archive in enumerate(archives):
        target = dest / f'archive_{k}'
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if not (target / m.filename).resolve().is_relative_to(target.resolve()):
                    raise ValueError('ZIP 路径越界')
            z.extractall(target)
    return dest

def discover_layout(root):
    roots = []
    for d in [root] + sorted(p for p in root.rglob('*') if p.is_dir()):
        labs = sorted(ALIASES[c.name.lower()] for c in d.iterdir() if c.is_dir() and c.name.lower() in ALIASES)
        if labs == list(range(6)): roots.append(d)
    if len(roots) != 1:
        raise RuntimeError(f'应只有一个含六类目录的训练根目录，发现：{roots}')
    records = [{'path': str(p), 'label': ALIASES[f.name.lower()]}
               for f in sorted(roots[0].iterdir()) if f.is_dir() and f.name.lower() in ALIASES
               for p in image_files(f)]
    train = pd.DataFrame(records).sort_values(['label', 'path']).reset_index(drop=True)
    known = set(train.path)
    others = [p for p in image_files(root) if str(p) not in known]
    test = [p for p in others if any(s.lower().startswith('test') for s in p.relative_to(root).parts[:-1])]
    if not test or len(test) != len(others):
        raise RuntimeError(f'非训练图像 {len(others)} 张，其中位于 test 目录 {len(test)} 张，无法确定用途。')
    return train, test

def read_pixels(path):
    with Image.open(path) as im:
        if im.size != (48, 48): raise ValueError(f'尺寸不是 48×48：{path} {im.size}')
        return np.array(im.convert('L'), dtype=np.uint8)

def align_submission(test_paths, roots):
    by_name, by_stem = defaultdict(list), defaultdict(list)
    for p in test_paths:
        by_name[p.name].append(p); by_stem[p.stem].append(p)
    valid = []
    for csv in sorted({c for r in roots for c in Path(r).rglob('*.csv')}):
        try: frame = pd.read_csv(csv, dtype=str, keep_default_na=False)
        except Exception: continue
        if len(frame) != len(test_paths) or 'ID' not in frame or len(frame.columns) != 2 or frame.ID.duplicated().any():
            continue
        matched = [(by_name.get(v) or by_stem.get(v) or [None, None]) for v in frame.ID]
        if all(len(m) == 1 for m in matched) and len({m[0] for m in matched}) == len(test_paths):
            valid.append((csv, frame, [m[0] for m in matched]))
    if len(valid) > 1: raise RuntimeError(f'有多个合法示例 CSV：{[str(v[0]) for v in valid]}')
    if valid:
        csv, frame, ordered = valid[0]
        print('提交模板：', csv, list(frame.columns))
        return frame[['ID']].copy(), ordered, next(c for c in frame.columns if c != 'ID')
    print('未找到示例 CSV，按文件名作为 ID、列名', SUBMISSION_LABEL_COLUMN)
    ordered = sorted(test_paths, key=lambda p: p.name)
    return pd.DataFrame({'ID': [p.name for p in ordered]}), ordered, SUBMISSION_LABEL_COLUMN

print('开始数据预检', flush=True)
source_root = discover_data_root()
data_root = extract_if_needed(source_root)
train_df, test_paths = discover_layout(data_root)
if not SMOKE and len(train_df) != 28275:
    raise RuntimeError(f'训练图像{len(train_df)}张，官方预期28275；请只添加官方比赛图像。')
if EXPECTED_TEST_ROWS and len(test_paths) != EXPECTED_TEST_ROWS:
    raise RuntimeError(f'测试图像 {len(test_paths)} 张，官方预期 {EXPECTED_TEST_ROWS}。')
sample_submission, test_paths, TARGET_COL = align_submission(test_paths, [source_root, data_root])
X = np.stack([read_pixels(p) for p in train_df.path])
X_test = np.stack([read_pixels(p) for p in test_paths])
h = [hashlib.sha256(a.tobytes()).hexdigest() for a in X]
hm = [hashlib.sha256(np.ascontiguousarray(a[:, ::-1]).tobytes()).hexdigest() for a in X]
train_df['group'] = [min(a, b) for a, b in zip(h, hm)]
train_df['pixel_std'] = X.reshape(len(X), -1).std(1) / 255
train_df['pixel_mean'] = X.reshape(len(X), -1).mean(1) / 255
train_df['relative_path'] = [str(Path(p).relative_to(data_root)) for p in train_df.path]
y = train_df.label.to_numpy(np.int64)
DATA_FP = digest({'train': list(zip(train_df.relative_path, train_df.label.tolist(), h)),
                  'test': sample_submission.ID.tolist(), 'test_hash': hashlib.sha256(X_test.tobytes()).hexdigest()})
conflict = train_df.groupby('group').label.nunique()
conflict_groups = set(conflict[conflict > 1].index)
blank = train_df.pixel_std.to_numpy() < 1e-3
AUDIT = {'n_train': len(X), 'n_test': len(X_test),
         'class_counts': {LABEL_NAMES[k]: int(v) for k, v in train_df.label.value_counts().sort_index().items()},
         'duplicate_or_mirror_extra_rows': int(len(X) - train_df.group.nunique()),
         'conflicting_label_groups': len(conflict_groups),
         'conflicting_label_rows': int(train_df.group.isin(conflict_groups).sum()),
         'blank_images_flagged': int(blank.sum()), 'blank_images_excluded': 0, 'data_fingerprint': DATA_FP,
         'submission_columns': ['ID', TARGET_COL]}
save_json(OUTPUT_ROOT / 'data_audit.json', AUDIT)
print(AUDIT, flush=True)

folds = np.full(len(y), -1, np.int64)
usable = np.arange(len(y))
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for k, (_, vi) in enumerate(sgkf.split(usable, y[usable], train_df.group.to_numpy()[usable])):
    folds[usable[vi]] = k
train_df['fold'] = folds
for a in range(N_FOLDS):
    assert not set(train_df.group[folds == a]) & set(train_df.group[(folds != a) & (folds >= 0)])
SPLIT_FP = digest(folds.tolist())
train_df[['relative_path', 'label', 'fold', 'group']].to_csv(OUTPUT_ROOT / 'train_index.csv', index=False)
sample_submission.to_csv(OUTPUT_ROOT / 'test_ids.csv', index=False)
fold_counts = pd.crosstab(train_df.label.map(dict(enumerate(LABEL_NAMES))), train_df.fold)
fold_counts.to_csv(OUTPUT_ROOT / 'fold_counts.csv'); display(fold_counts)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
counts = train_df.label.value_counts().sort_index()
ax[0].bar(LABEL_NAMES, counts.values, color='tab:blue'); ax[0].set_title('Training images per class')
for i, v in enumerate(counts.values): ax[0].text(i, v, str(v), ha='center', va='bottom', fontsize=8)
for c in range(6):
    ax[1].hist(train_df.pixel_mean[(y == c) & ~blank], bins=40, histtype='step', label=LABEL_NAMES[c])
ax[1].set_title('Mean brightness by class'); ax[1].legend(fontsize=7)
ax[2].hist(train_df.pixel_std[~blank], bins=40); ax[2].set_title('Per-image contrast (pixel std)')
fig.tight_layout(); fig.savefig(OUTPUT_ROOT / 'eda_distribution.png', dpi=150); plt.close(fig)
fig, ax = plt.subplots(7, 8, figsize=(10, 9))
rng = np.random.default_rng(SEED)
for c in range(6):
    pick = rng.choice(np.flatnonzero((y == c) & ~blank), 7, replace=False)
    ax[c, 0].imshow(X[(y == c) & ~blank].mean(0), cmap='gray'); ax[c, 0].set_ylabel(LABEL_NAMES[c])
    for j, i in enumerate(pick): ax[c, j + 1].imshow(X[i], cmap='gray', vmin=0, vmax=255)
ax[6, 0].imshow(X[~blank].mean(0), cmap='gray'); ax[6, 0].set_ylabel('All (mean)')
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
for a in ax[6, 1:]: a.axis('off')
ax[0, 0].set_title('class mean'); fig.tight_layout()
fig.savefig(OUTPUT_ROOT / 'eda_samples_and_mean_faces.png', dpi=150); plt.close(fig)

BASE = dict(arch='small_resnet32', pretrained=False, img_size=48, epochs=60, batch=128,
            opt_name='adamw', lr=.001, head_lr_mult=1., wd=.0001, momentum=.9, warmup=3, drop_path=0.,
            smoothing=.05, mixup=.2, ema=.999, class_weight=False, freeze=False,
            flip=True, rot=10., translate=.05, scale=(.95,1.05), jitter=.15, autocontrast_p=.25,
            erase_p=0., seed=SEED, folds=FOLD_IDS)
def exp(name, **kw): return {**BASE, **kw, 'name':name}
PLANS={'cv_resnet32':[exp('resnet32_scratch48')],
       'cv_resnet32_s2027':[exp('resnet32_scratch48_s2027', seed=SEED+1)]}
if RUN not in PLANS: raise ValueError(f'未知 RUN={RUN}')
PLAN = PLANS[RUN]
if SMOKE:
    PLAN = [{**c, 'epochs': int(os.environ.get('TASK2_SMOKE_EPOCHS', 1)), 'batch': 16, 'warmup': 0, 'img_size': int(os.environ.get('TASK2_SMOKE_SIZE', c['img_size'])),
             'pretrained': c['pretrained'] and os.environ.get('TASK2_SMOKE_PRETRAINED') == '1'}
            for c in PLAN][:int(os.environ.get('TASK2_SMOKE_EXPS', 99))]
if SMOKE and os.environ.get('TASK2_SMOKE_FOLDS'):
    FOLD_IDS=list(range(int(os.environ['TASK2_SMOKE_FOLDS'])))
    for c in PLAN:c['folds']=FOLD_IDS
if os.environ.get('TASK2_FOLDS'):
    FOLD_IDS = json.loads(os.environ['TASK2_FOLDS'])
    for c in PLAN: c['folds'] = FOLD_IDS
for c in PLAN: c['scale'] = tuple(c['scale'])
PLANS[RUN] = PLAN
display(pd.DataFrame(PLAN).set_index('name'))
save_json(OUTPUT_ROOT / 'plan.json', PLAN)

def affine_sample(x, size, flip, ang, sc, tx, ty):
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (x.shape[0], 1, size, size), align_corners=False)
    return F.grid_sample(x-128/255, grid, mode='bilinear', padding_mode='zeros', align_corners=False)+128/255

def random_erase(x, p, area=(0.02, 0.2), ratio=(0.3, 3.3)):
    B, _, H, W = x.shape; dev = x.device
    a = torch.empty(B, device=dev).uniform_(*area) * H * W
    r = torch.exp(torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1])))
    hh, ww = (a * r).sqrt().clamp(1, H), (a / r).sqrt().clamp(1, W)
    cy, cx = torch.rand(B, device=dev) * H, torch.rand(B, device=dev) * W
    yy, xx = torch.arange(H, device=dev)[None, :, None], torch.arange(W, device=dev)[None, None, :]
    m = ((yy - cy[:, None, None]).abs() < hh[:, None, None] / 2) & ((xx - cx[:, None, None]).abs() < ww[:, None, None] / 2)
    m &= (torch.rand(B, device=dev) < p)[:, None, None]
    return torch.where(m[:, None], torch.randn_like(x), x)

def augment_batch(xb, cfg, norm):
    B, dev = xb.shape[0], xb.device
    x = xb.float() / 255
    if cfg['autocontrast_p'] > 0:
        lo, hi = x.amin((1, 2, 3), keepdim=True), x.amax((1, 2, 3), keepdim=True)
        pick = (torch.rand(B, 1, 1, 1, device=dev) < cfg['autocontrast_p']) & (hi - lo > 1e-3)
        x = torch.where(pick, (x - lo) / (hi - lo).clamp_min(1e-3), x)
    if cfg['jitter'] > 0:
        j = cfg['jitter']
        c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * j
        b = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * j / 2
        mu = x.mean((1, 2, 3), keepdim=True)
        x = ((x - mu) * c + mu + b).clamp(0, 1)
    u = lambda: torch.rand(B, device=dev) * 2 - 1
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0) if cfg['flip'] else torch.ones(B, device=dev)
    ang = u() * math.radians(cfg['rot'])
    sc = torch.empty(B, device=dev).uniform_(*cfg['scale'])
    x = affine_sample(x, cfg['img_size'], flip, ang, sc, u() * 2 * cfg['translate'], u() * 2 * cfg['translate'])
    x = (x - norm[0]) / norm[1]
    return random_erase(x, cfg['erase_p']) if cfg['erase_p'] > 0 else x

SHIFT = 2 * 2 / 48
VIEWS = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)] + [(f, dx, dy) for f in (1.0, -1.0)
         for dx, dy in [(SHIFT, 0), (-SHIFT, 0), (0, SHIFT), (0, -SHIFT)]]

def view_batch(xb, size, norm, view):
    B, dev = xb.shape[0], xb.device
    f, dx, dy = view
    one = torch.ones(B, device=dev)
    x = affine_sample(xb.float() / 255, size, one * f, one * 0, one, one * dx, one * dy)
    return (x - norm[0]) / norm[1]

XG = torch.from_numpy(X)[:, None].to(DEVICE)
XTG = torch.from_numpy(X_test)[:, None].to(DEVICE)
YG = torch.from_numpy(y.copy()).to(DEVICE)
TEST_IDX = np.arange(len(X_test))

_pick = XG[[np.flatnonzero((y == c) & ~blank)[0] for c in range(6)]]
fig, ax = plt.subplots(6, 8, figsize=(10, 7.5))
for j in range(8):
    imgs = view_batch(_pick, 48, (0., 1.), VIEWS[0]) if j == 0 else augment_batch(_pick, BASE, (0., 1.))
    for c in range(6):
        ax[c, j].imshow(imgs[c, 0].float().cpu().numpy(), cmap='gray', vmin=0, vmax=1); ax[c, j].axis('off')
    ax[0, j].set_title('original' if j == 0 else f'aug {j}', fontsize=9)
fig.suptitle('prior augmentation (48x48; no random erasing)'); fig.tight_layout()
fig.savefig(OUTPUT_ROOT / 'augmentation_preview.png', dpi=150); plt.close(fig)

def conv_bn(cin,cout,stride=1):
    return nn.Sequential(nn.Conv2d(cin,cout,3,stride,1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True))
class ResidualBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.main=nn.Sequential(conv_bn(cin,cout,stride),nn.Conv2d(cout,cout,3,1,1,bias=False),nn.BatchNorm2d(cout))
        self.skip=nn.Identity() if cin==cout and stride==1 else nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout))
        self.act=nn.ReLU(inplace=True)
    def forward(self,x):return self.act(self.main(x)+self.skip(x))
class SmallResNet(nn.Module):
    def __init__(self,blocks=5):
        super().__init__();self.stem=conv_bn(1,32);layers=[];cin=32
        for stage,cout in enumerate([32,64,128]):
            for k in range(blocks):
                layers.append(ResidualBlock(cin,cout,2 if stage and k==0 else 1));cin=cout
        self.features=nn.Sequential(*layers);self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Dropout(.2),nn.Linear(128,6))
    def forward(self,x):return self.fc(self.pool(self.features(self.stem(x))).flatten(1))
    def get_classifier(self):return self.fc[-1]
def build_model(cfg):
    assert cfg['arch']=='small_resnet32' and not cfg['pretrained'] and cfg['img_size']==48
    return SmallResNet(5)

class EMA:
    def __init__(self, model, decay):
        self.decay, self.module = decay, copy.deepcopy(model).eval()
        for p in self.module.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point: e.mul_(d).add_(m.detach(), alpha=1 - d)
            else: e.copy_(m)

def build_optimizer(model,cfg):
    return torch.optim.AdamW([{'params':list(model.parameters()),'lr_scale':1.}],
                             lr=cfg['lr'],weight_decay=cfg['wd'])

def freeze_backbone(model):
    """只训练分类头（线性探针），用于衡量预训练特征本身的判别力。BN 统计量仍随训练更新。"""
    head = model.get_classifier() if hasattr(model, 'get_classifier') else model.head
    for p in model.parameters(): p.requires_grad_(False)
    for p in head.parameters(): p.requires_grad_(True)

@torch.inference_mode()
def predict(model, XS, idx, cfg, norm, views=(VIEWS[0],), bs=None):
    """返回每个视图的 softmax 概率，形状 (视图数, N, 6)。"""
    bs = min(cfg['batch'], 128) if bs is None else bs
    model.eval(); out = []
    for v in views:
        probs = []
        for s in range(0, len(idx), bs):
            xb = view_batch(XS[idx[s:s + bs]], cfg['img_size'], norm, v).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
                probs.append(model(xb).float().softmax(1).cpu())
        out.append(torch.cat(probs).numpy())
    return np.stack(out)

def tta_variants(p):
    return {'none': p[0], 'flip': p[:2].mean(0), 'tta10': p.mean(0)}

def metrics(labels, probs):
    pred = probs.argmax(1)
    return {'acc': float(accuracy_score(labels, pred)),
            'macro_f1': float(f1_score(labels, pred, average='macro', labels=range(6), zero_division=0)),
            'nll': float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-9, 1)).mean())}

def lr_at(step, total, warm, base):
    if step < warm: return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

def minutes_left():
    return SESSION_MINUTES - (time.time() - START_TIME) / 60

def _train_steps(net, model, opt, scaler, crit, cfg, norm, idx, steps):
    model.train()
    for _ in range(steps):
        xb = augment_batch(XG[idx], cfg, norm).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
            loss = crit(net(xb), YG[idx])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if AMP: torch.cuda.synchronize()

PROBE_BUDGET_S = 1.5 if SMOKE else 8.0
PROBE_INFER_N = 256 if SMOKE else 4096

def measure(arch, img_size, batch, budget_s=PROBE_BUDGET_S):
    cfg = {**BASE, 'arch': arch, 'img_size': img_size, 'batch': batch, 'pretrained': False}
    seed_everything(SEED)
    model = build_model(cfg).to(DEVICE).to(memory_format=torch.channels_last)
    net = model
    opt = build_optimizer(model, cfg); scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    crit = nn.CrossEntropyLoss(); norm = (0.5, 0.25)
    idx = torch.arange(batch, device=DEVICE) % len(X)
    tick = time.time(); _train_steps(net, model, opt, scaler, crit, cfg, norm, idx, 3)
    steps = int(max(3, min(40, 3 * budget_s / max(time.time() - tick, 1e-3))))
    rss0 = host_mem()[0]; tick = time.time()
    _train_steps(net, model, opt, scaler, crit, cfg, norm, idx, steps)
    train_ips = steps * batch / (time.time() - tick)
    rss_mb_per_step = (host_mem()[0] - rss0) * 1024 / steps
    n = min(PROBE_INFER_N, len(X)); tick = time.time()
    predict(model, XG, np.arange(n), cfg, norm)
    infer_ips = n / (time.time() - tick)
    out = {'arch': arch, 'img_size': img_size, 'batch': batch, 'parallel': 'single',
           'params_M': sum(p.numel() for p in model.parameters()) / 1e6, 'steps_timed': steps,
           'train_img_per_s': train_ips, 'infer_img_per_s': infer_ips,
           'rss_growth_MB_per_step': rss_mb_per_step, 'gpu_peak_GB': gpu_peak_gb()}
    del net, model, opt, scaler; gc.collect()
    if AMP: torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    print('  ', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    return out

if SMOKE:
    PROBE_SET = list(dict.fromkeys([(c['arch'], c['img_size'], c['batch']) for c in PLAN]))
else:
    probe_plans = PLANS.values() if PROBE_ONLY else [PLAN]
    PROBE_SET = list(dict.fromkeys([(c['arch'], c['img_size'], c['batch']) for plan in probe_plans for c in plan]))
print(f'开始吞吐量实测（{len(PROBE_SET)} 组配置）', flush=True)
probe = [measure(*s) for s in PROBE_SET]
PROBE = pd.DataFrame(probe)
PROBE.to_csv(OUTPUT_ROOT / 'throughput.csv', index=False); display(PROBE)
IPS = {(r.arch, r.img_size): (r.train_img_per_s, r.infer_img_per_s)
       for r in PROBE[PROBE.parallel == 'single'].itertuples()}

N_VAL = int((folds == 0).sum()); N_TR = int((folds >= 0).sum()) - N_VAL

def project_fold(cfg, test_views=True):
    """按实测吞吐量换算单折分钟数：每轮训练 + 两次全验证集推理，折末多视图 TTA 与测试集推理。"""
    tr_ips, inf_ips = IPS[(cfg['arch'], cfg['img_size'])]
    per_epoch = N_TR / tr_ips + 2 * N_VAL / inf_ips
    fold_end = (12 * N_VAL + (len(VIEWS) * len(X_test) if test_views else 0) + 5000) / inf_ips
    return (cfg['epochs'] * per_epoch + fold_end) / 60, per_epoch / 60

budget_rows = []
for run, plan in PLANS.items():
    for cfg in plan:
        if (cfg['arch'], cfg['img_size']) not in IPS: continue
        per_fold, per_epoch = project_fold(cfg, run.startswith('cv_'))
        budget_rows.append({'run': run, 'experiment': cfg['name'], 'arch': cfg['arch'],
                            'img_size': cfg['img_size'], 'epochs': cfg['epochs'], 'folds': len(cfg['folds']),
                            'min_per_epoch': per_epoch, 'min_per_fold': per_fold,
                            'min_total': per_fold * len(cfg['folds'])})
BUDGET = pd.DataFrame(budget_rows)
EST_EPOCH_MIN = dict(zip(BUDGET.experiment, BUDGET.min_per_epoch))
per_run = BUDGET.groupby('run').min_total.sum().sort_values()
BUDGET.to_csv(OUTPUT_ROOT / 'budget_projection.csv', index=False); display(BUDGET.round(2))
print('每个 Notebook 的预计总分钟数：', per_run.round(1).to_dict(), flush=True)
need = float(per_run.get(RUN, 0.0))
print(f'本组 RUN={RUN} 预计 {need:.0f} 分钟，软预算剩余 {minutes_left():.0f} 分钟 → '
      f'{"可以跑完" if need < minutes_left() else "跑不完，会按断点停下并保留已完成的折"}', flush=True)
save_json(OUTPUT_ROOT / 'budget_verdict.json',
          {'run': RUN, 'projected_minutes': need, 'minutes_left_after_probe': minutes_left(),
           'fits': bool(need < minutes_left()), 'per_run_projection': per_run.round(1).to_dict(),
           'n_train_per_fold': N_TR, 'n_val_per_fold': N_VAL, 'probe_only': PROBE_ONLY})

if len(PROBE) > 1:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    lab = [f"{r.arch.split('.')[0]}@{r.img_size}" + ('(DP)' if r.parallel == 'dp' else '')
           for r in PROBE.itertuples()]
    ax[0].barh(lab, PROBE.train_img_per_s, color=['tab:red' if p == 'dp' else 'tab:blue' for p in PROBE.parallel])
    ax[0].set_xlabel('training images / s (measured on this GPU)'); ax[0].invert_yaxis()
    for i, v in enumerate(PROBE.train_img_per_s): ax[0].text(v, i, f' {v:.0f}', va='center', fontsize=8)
    ax[0].set_title('Throughput measured on a single GPU')
    if len(BUDGET):
        ax[1].barh(BUDGET.experiment, BUDGET.min_total, color='tab:green')
        for i, v in enumerate(BUDGET.min_total): ax[1].text(v, i, f' {v:.0f}min', va='center', fontsize=8)
        ax[1].axvline(SESSION_MINUTES, color='k', ls='--', lw=1, label=f'session budget {SESSION_MINUTES:.0f}min')
        ax[1].legend(fontsize=7)
    ax[1].set_xlabel('projected minutes for the whole plan'); ax[1].invert_yaxis()
    ax[1].set_title('Projected cost per notebook')
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / 'throughput_and_budget.png', dpi=150); plt.close(fig)
print('实测与预算完成，', mem_tag(), flush=True)

EXP_ROOT = OUTPUT_ROOT / 'experiments'
EXP_ROOT.mkdir(exist_ok=True)

def import_previous_outputs():
    roots = [Path(RESUME_ROOT)] if RESUME_ROOT else [p for p in INPUT_ROOT.rglob(RUN)
             if p.is_dir() and p.parent.name == 'task2' and (p / 'experiments').is_dir()]
    for root in roots:
        for d in sorted((root / 'experiments').glob('*/fold*')):
            dest = EXP_ROOT / d.parent.name / d.name
            if (dest / 'done.json').exists(): continue
            if (d / 'done.json').exists() or (d / 'last.pt').exists():
                if dest.exists(): shutil.rmtree(dest)
                shutil.copytree(d, dest); print('导入上次输出：', d, flush=True)

import_previous_outputs()

def cfg_hash(cfg):
    keep = {k: v for k, v in cfg.items() if k != 'folds'}
    return digest({'cfg': keep, 'data': DATA_FP, 'split': SPLIT_FP, 'pipeline': 'scratch-resnet32',
                   'torch': str(torch.__version__), 'model_implementation': 'local SmallResNet32'})[:16]

def to_cpu(obj):
    if torch.is_tensor(obj): return obj.detach().to('cpu', copy=True)
    if isinstance(obj, dict): return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return type(obj)(to_cpu(v) for v in obj)
    return obj

def atomic_save(obj, path):
    tmp = Path(str(path) + '.tmp'); torch.save(obj, tmp); os.replace(tmp, path)

def write_submission(name, done_folds):
    """用当前已完成的折重写提交文件（K 折集成 + EMA + 水平翻转 TTA）。"""
    probs = [np.load(EXP_ROOT / name / f'fold{f}' / 'test_probs.npz')['ema_flip'] for f in sorted(done_folds)]
    for f, prob in zip(sorted(done_folds), probs):
        cache = np.load(EXP_ROOT / name / f'fold{f}' / 'test_probs.npz')
        assert cache['ids'].astype(str).tolist()==sample_submission.ID.tolist(), '缓存测试ID未对齐'
        assert prob.shape==(len(X_test),6) and np.isfinite(prob).all() and (prob>=0).all()
        assert np.allclose(prob.sum(1),1,atol=1e-4)
    sub = sample_submission.copy()
    sub[TARGET_COL] = np.mean(probs, 0).argmax(1).astype(int)
    assert len(sub) == len(X_test) and sub.ID.is_unique and sub.ID.tolist() == sample_submission.ID.tolist()
    sub.to_csv(OUTPUT_ROOT / 'submission.csv', index=False)
    save_json(OUTPUT_ROOT / 'submission_info.json',
              {'experiment': name, 'folds_used': sorted(done_folds), 'n_folds_planned': N_FOLDS, 'current_run_planned_folds':FOLD_IDS,
               'inference': 'EMA weights + horizontal-flip TTA, mean of fold probabilities',
               'class_distribution': sub[TARGET_COL].value_counts().sort_index().to_dict()})
    print(f'  已写出 submission.csv（折 {sorted(done_folds)} 的集成）', flush=True)

def run_fold(cfg, fold):
    fdir = EXP_ROOT / cfg['name'] / f'fold{fold}'; fdir.mkdir(parents=True, exist_ok=True)
    key = cfg_hash(cfg)
    if (fdir / 'done.json').exists():
        done = json.loads((fdir / 'done.json').read_text())
        if done['cfg_hash'] == key:
            print(f"[{cfg['name']} fold{fold}] 已完成，复用结果", flush=True); return done
        raise RuntimeError(f'{fdir} 的配置/数据指纹与当前不一致，请删除旧输出或改实验名。')
    est = EST_EPOCH_MIN.get(cfg['name'], 3.0)
    infer_ips = IPS[(cfg['arch'], cfg['img_size'])][1]
    val_rows = int((folds == fold).sum())
    finish_reserve = max(6.0, (12*val_rows + (len(VIEWS)*len(X_test) if DO_TEST else 0) + 5000) / infer_ips / 60 + 2)
    if minutes_left() < est + finish_reserve:
        print(f"预算不足（每轮约 {est:.1f} 分钟），跳过 {cfg['name']} fold{fold}；保存输出后续跑。", flush=True)
        return None
    tr = np.flatnonzero((folds != fold) & (folds >= 0)); va = np.flatnonzero(folds == fold)
    norm = (float(X[tr].mean() / 255), max(1e-6,float(X[tr].std() / 255)))
    seed_everything(cfg['seed'] * 100 + fold)
    model = build_model(cfg).to(DEVICE).to(memory_format=torch.channels_last)
    if cfg['freeze']: freeze_backbone(model)
    ema = EMA(model, cfg['ema'])
    opt = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    B = cfg['batch']; spe = max(1, math.ceil(len(tr) / B)); total = spe * cfg['epochs']; warm = spe * cfg['warmup']
    history, start = [], 0
    if (fdir / 'last.pt').exists():
        ck = torch.load(fdir / 'last.pt', map_location='cpu', weights_only=False)
        if ck['cfg_hash'] != key: raise RuntimeError(f'{fdir}/last.pt 与当前配置不一致')
        model.load_state_dict(ck['model']); ema.module.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        history, start = ck['history'], ck['epoch']
        random.setstate(ck['rng_python']); np.random.set_state(ck['rng_numpy'])
        torch.set_rng_state(ck['rng_torch'])
        if AMP: torch.cuda.set_rng_state_all(ck['rng_cuda'])
        del ck; gc.collect()
        print(f"[{cfg['name']} fold{fold}] 从第 {start} 轮后续跑", flush=True)
    weight = None
    if cfg['class_weight']:
        cnt = np.bincount(y[tr], minlength=6).astype(np.float64)
        weight = torch.tensor(cnt.sum() / (6 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg['smoothing'], weight=weight)
    trG = torch.from_numpy(tr).to(DEVICE)
    for epoch in range(start, cfg['epochs']):
        per_epoch = np.mean([h['seconds'] for h in history[-3:]]) / 60 if history else est
        if minutes_left() < 1.3 * per_epoch + finish_reserve:
            print(f'会话软预算即将用完，在 {cfg["name"]} fold{fold} 第 {epoch} 轮后停止；保存输出后续跑。', flush=True)
            return None
        avail = host_mem()[1]
        if avail == avail and avail < MIN_FREE_GB:
            print(f'系统可用内存仅 {avail:.1f}GB（阈值 {MIN_FREE_GB}GB），在第 {epoch} 轮后主动停止，'
                  f'避免内核被 OOM killer 杀掉；保存输出后续跑。', flush=True)
            return None
        tick = time.time(); model.train(); tot, cnt_seen = 0.0, 0
        perm = trG[torch.randperm(len(tr), device=DEVICE)]
        for s in range(spe):
            step = epoch * spe + s
            for g in opt.param_groups: g['lr'] = lr_at(step, total, warm, cfg['lr']) * g['lr_scale']
            bi = perm[s * B:(s + 1) * B]
            xb = augment_batch(XG[bi], cfg, norm).contiguous(memory_format=torch.channels_last); yb = YG[bi]
            lam = float(np.random.beta(cfg['mixup'], cfg['mixup'])) if cfg['mixup'] > 0 else 1.0
            mix = torch.randperm(len(yb), device=DEVICE)
            if lam < 1: xb = lam * xb + (1 - lam) * xb[mix]
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
                logits = model(xb)
                loss = lam * crit(logits, yb) + (1 - lam) * crit(logits, yb[mix]) if lam < 1 else crit(logits, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(opt); scaler.update(); ema.update(model, step)
            tot += loss.item() * len(yb); cnt_seen += len(yb)
        raw = metrics(y[va], predict(model, XG, va, cfg, norm)[0])
        sm = metrics(y[va], predict(ema.module, XG, va, cfg, norm)[0])
        rss, avail = host_mem()
        history.append({'epoch': epoch + 1, 'train_loss': tot / cnt_seen, 'lr_end': lr_at(step, total, warm, cfg['lr']), 'head_lr_end': lr_at(step, total, warm, cfg['lr'])*cfg['head_lr_mult'],
                        'val_acc_raw': raw['acc'], 'val_nll_raw': raw['nll'], 'val_acc_ema': sm['acc'],
                        'val_nll_ema': sm['nll'], 'seconds': time.time() - tick, 'rss_gb': rss,
                        'mem_available_gb': avail, 'gpu_peak_gb': gpu_peak_gb()})
        print(f"[{cfg['name']} fold{fold}] epoch {epoch + 1}/{cfg['epochs']} loss={tot / cnt_seen:.4f} "
              f"val_raw={raw['acc']:.4f} val_ema={sm['acc']:.4f} {time.time() - tick:.0f}s "
              f"剩余{minutes_left():.0f}min {mem_tag()}", flush=True)
        atomic_save({'cfg_hash': key, 'model': to_cpu(model.state_dict()), 'ema': to_cpu(ema.module.state_dict()),
                     'opt': to_cpu(opt.state_dict()), 'scaler': scaler.state_dict(), 'history': history,
                     'epoch': epoch + 1, 'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                     'rng_torch': torch.get_rng_state(),
                     'rng_cuda': torch.cuda.get_rng_state_all() if AMP else []}, fdir / 'last.pt')
        if SMOKE and os.environ.get('TASK2_SMOKE_STOP_AFTER_EPOCH') == str(epoch + 1):
            print('合成自检：在已保存的轮间断点停止', flush=True)
            return None
    return finish_fold(cfg, fold, fdir, key, model, ema.module, tr, va, norm, history)

def finish_fold(cfg, fold, fdir, key, model, ema_model, tr, va, norm, history):
    pv_ema = tta_variants(predict(ema_model, XG, va, cfg, norm, VIEWS))
    pv_raw = tta_variants(predict(model, XG, va, cfg, norm, VIEWS[:2]))
    sub_idx = np.random.default_rng(fold).choice(tr, min(5000, len(tr)), replace=False)
    clean_train = metrics(y[sub_idx], predict(ema_model, XG, sub_idx, cfg, norm)[0])['acc']
    np.savez_compressed(fdir / 'val_probs.npz', index=va, labels=y[va],
                        **{f'ema_{k}': v for k, v in pv_ema.items()}, raw_flip=pv_raw['flip'])
    if DO_TEST:
        pt_ema = tta_variants(predict(ema_model, XTG, TEST_IDX, cfg, norm, VIEWS))
        np.savez_compressed(fdir / 'test_probs.npz', ids=sample_submission.ID.to_numpy(dtype=str), **{f'ema_{k}': v for k, v in pt_ema.items()})
    pd.DataFrame(history).to_csv(fdir / 'history.csv', index=False)
    if DO_TEST:
        torch.save({k: (v.half() if v.dtype.is_floating_point else v)
                    for k, v in to_cpu(ema_model.state_dict()).items()}, fdir / 'final_ema_fp16.pt')
    hist = pd.DataFrame(history)
    done = {'cfg_hash': key, 'name': cfg['name'], 'fold': fold, 'n_train': len(tr), 'n_val': len(va),
            'norm': norm, 'params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'train_minutes': float(hist.seconds.sum() / 60), 'clean_train_acc_ema': clean_train,
            'best_epoch_ema': int(hist.val_acc_ema.idxmax() + 1), 'best_val_acc_ema_none': float(hist.val_acc_ema.max()),
            'peak_rss_gb': float(hist.rss_gb.max()), 'min_mem_available_gb': float(hist.mem_available_gb.min()),
            **{f'val_{k}': metrics(y[va], v) for k, v in [('ema_none', pv_ema['none']), ('ema_flip', pv_ema['flip']),
               ('ema_tta10', pv_ema['tta10']), ('raw_flip', pv_raw['flip'])]}}
    save_json(fdir / 'done.json', done)
    (fdir / 'last.pt').unlink(missing_ok=True)
    print(f"[{cfg['name']} fold{fold}] 完成：EMA+flip 验证 {done['val_ema_flip']['acc']:.4f}，"
          f"干净训练 {clean_train:.4f}，用时 {done['train_minutes']:.1f}min", flush=True)
    return done

RESULTS = {}
for cfg in (PLAN if not PROBE_ONLY else []):
    for fold in cfg['folds']:
        r = run_fold(cfg, fold)
        gc.collect()
        if AMP: torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        if r is None: break
        RESULTS.setdefault(cfg['name'], {})[fold] = r
        if DO_TEST: write_submission(cfg['name'], RESULTS[cfg['name']].keys())
    if minutes_left() < 8: break
COMPLETE = [c for c in PLAN if len(RESULTS.get(c['name'], {})) == len(c['folds'])]
print(f'完成实验 {len(COMPLETE)}/{len(PLAN)}；{mem_tag()}', flush=True)

VAL_KEYS = ['ema_none', 'ema_flip', 'ema_tta10', 'raw_flip']
TEST_KEYS = ['ema_none', 'ema_flip', 'ema_tta10']

def load_fold(name, fold):
    d = EXP_ROOT / name / f'fold{fold}'
    test = np.load(d / 'test_probs.npz') if (d / 'test_probs.npz').exists() else None
    return np.load(d / 'val_probs.npz'), test, pd.read_csv(d / 'history.csv')

rows, curves = [], {}
for cfg in COMPLETE:
    name = cfg['name']; oof = {k: np.full((len(y), 6), np.nan) for k in VAL_KEYS}
    test = {k: [] for k in TEST_KEYS}; curves[name] = {}
    for fold in cfg['folds']:
        v, t, hist = load_fold(name, fold); curves[name][fold] = hist; r = RESULTS[name][fold]
        for k in VAL_KEYS: oof[k][v['index']] = v[k]
        if t is not None:
            for k in TEST_KEYS: test[k].append(t[k])
        rows.append({'experiment': name, 'fold': fold, **{f'acc_{k}': r[f'val_{k}']['acc'] for k in VAL_KEYS},
                     'macro_f1_ema_flip': r['val_ema_flip']['macro_f1'], 'nll_ema_flip': r['val_ema_flip']['nll'],
                     'best_epoch_ema': r['best_epoch_ema'], 'best_epoch_acc_ema_none': r['best_val_acc_ema_none'],
                     'clean_train_acc': r['clean_train_acc_ema'],
                     'gap_train_minus_val': r['clean_train_acc_ema'] - r['val_ema_none']['acc'],
                     'params_M': r['params'] / 1e6, 'train_minutes': r['train_minutes'],
                     'peak_rss_gb': r['peak_rss_gb']})
    mask = ~np.isnan(oof['ema_flip'][:, 0])
    np.savez_compressed(OUTPUT_ROOT / f'oof_{name}.npz', mask=mask, labels=y, folds=folds,
                        **{k: np.nan_to_num(v) for k, v in oof.items()})
    if test['ema_flip']:
        np.savez_compressed(OUTPUT_ROOT / f'test_{name}.npz', ids=sample_submission.ID.to_numpy(dtype=str),
                            **{k: np.mean(v, 0) for k, v in test.items()})
        sub = sample_submission.copy(); sub[TARGET_COL] = np.mean(test['ema_flip'], 0).argmax(1).astype(int)
        sub.to_csv(OUTPUT_ROOT / f'submission_{name}.csv', index=False)
    om = {k: metrics(y[mask], oof[k][mask]) for k in VAL_KEYS}
    fold_rows = [r for r in rows if r['experiment'] == name]
    rows.append({'experiment': name, 'fold': 'OOF' if len(cfg['folds']) == N_FOLDS else 'pooled',
                 **{f'acc_{k}': om[k]['acc'] for k in VAL_KEYS},
                 'macro_f1_ema_flip': om['ema_flip']['macro_f1'], 'nll_ema_flip': om['ema_flip']['nll'],
                 'fold_std_acc_ema_flip': float(np.std([r['acc_ema_flip'] for r in fold_rows])),
                 **{k: float(np.mean([r[k] for r in fold_rows])) for k in
                    ['best_epoch_acc_ema_none', 'clean_train_acc', 'gap_train_minus_val', 'params_M']},
                 'train_minutes': float(np.sum([r['train_minutes'] for r in fold_rows]))})
    pred, lab = oof['ema_flip'][mask].argmax(1), y[mask]
    pd.DataFrame(classification_report(lab, pred, target_names=LABEL_NAMES, output_dict=True, zero_division=0)).T \
        .to_csv(OUTPUT_ROOT / f'class_report_{name}.csv')
    cm = confusion_matrix(lab, pred, labels=range(6), normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    for i in range(6):
        for j in range(6): ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
                                   color='white' if cm[i, j] > .5 else 'black', fontsize=8)
    ax.set_xticks(range(6), LABEL_NAMES, rotation=30); ax.set_yticks(range(6), LABEL_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'{name} confusion (row-normalized)')
    fig.colorbar(im); fig.tight_layout(); fig.savefig(OUTPUT_ROOT / f'confusion_{name}.png', dpi=150); plt.close(fig)
summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT_ROOT / 'summary.csv', index=False)
if len(summary): display(summary)

if curves:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, per_fold in curves.items():
        for fold, h in per_fold.items():
            lab = f'{name} f{fold}' if len(per_fold) > 1 else name
            ax[0].plot(h.epoch, h.val_acc_ema, label=lab)
            ax[1].plot(h.epoch, h.train_loss, label=lab)
            ax[2].plot(h.epoch, h.rss_gb, label=lab)
    ax[0].set_title('Validation accuracy (EMA, no TTA)'); ax[1].set_title('Training loss (augmented, mixup)')
    ax[2].set_title('Process RSS (GB)')
    for a in ax: a.set_xlabel('epoch'); a.grid(alpha=.3)
    ax[0].legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / 'learning_curves.png', dpi=150); plt.close(fig)

if DO_TEST and COMPLETE:
    name = COMPLETE[0]['name']
    write_submission(name, RESULTS[name].keys())
    sub = pd.read_csv(OUTPUT_ROOT / 'submission.csv', dtype={'ID': str}, keep_default_na=False)
    assert len(sub) == len(X_test) and sub.ID.is_unique and sub.ID.tolist() == sample_submission.ID.tolist()
    assert sub[TARGET_COL].astype(int).between(0, 5).all()
    oof_acc = summary.query("fold == 'OOF'").acc_ema_flip.dropna()
    if len(oof_acc): print(f'OOF(EMA+flip) = {float(oof_acc.iloc[0]):.4f}', flush=True)
    print('预测类别分布：', sub[TARGET_COL].value_counts().sort_index().to_dict(), flush=True)
status = {'run': RUN, 'planned_folds':FOLD_IDS, 'full_5fold_complete':FOLD_IDS==list(range(5)) and bool(COMPLETE), 'probe_only': PROBE_ONLY, 'complete_experiments': [c['name'] for c in COMPLETE],
          'pending': [c['name'] for c in PLAN if c not in COMPLETE],
          'folds_done': {k: sorted(v) for k, v in RESULTS.items()},
          'elapsed_minutes': (time.time() - START_TIME) / 60, 'data_fingerprint': DATA_FP,
          'split_fingerprint': SPLIT_FP, 'host_mem_end': host_mem(), 'gpu_peak_gb': gpu_peak_gb()}
save_json(OUTPUT_ROOT / 'status.json', status)
with zipfile.ZipFile(OUTPUT_ROOT / f'{RUN}_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUTPUT_ROOT.rglob('*')):
        if p.is_file() and p.suffix in {'.csv', '.json', '.png', '.npz'}:
            z.write(p, p.relative_to(OUTPUT_ROOT))
print(json.dumps(status, ensure_ascii=False, indent=1, default=str))
if status['pending']:
    print('仍有未完成实验：Save Version 后，把本次 Notebook 的 Output 添加为 Input，再运行一次即可续跑。')
else:
    print('全部完成。下载', OUTPUT_ROOT / f'{RUN}_results.zip')
