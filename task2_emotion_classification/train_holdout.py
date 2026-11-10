"""Scratch ResNet32 on the original grouped 70/15/15 split, with best-checkpoint selection and flip TTA."""
from cli import configure
ARGS = configure(['single_resnet32'], 'single_resnet32')
from pathlib import Path
import os, json, math, time, random, hashlib, platform, copy, gc, zipfile, shutil
from dataclasses import dataclass, asdict, replace
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from IPython.display import display, Markdown

MODE = 'smoke' if os.environ.get('TASK2_SMOKE') == '1' else 'full'
INPUT_ROOT = Path(os.environ.get('TASK2_INPUT', '/kaggle/input'))
DATA_ROOT = ARGS.data_root
OUTPUT_ROOT = Path(os.environ.get('TASK2_OUTPUT', '/kaggle/working/task2'))
RUN_GROUP = 'depth'
EXPERIMENT_SEEDS = [2026]
RESUME_ROOT = ARGS.resume
SESSION_MINUTES = 120
STUDY_EPOCHS = 60
BATCH_SIZE = 128
NUM_WORKERS = 0
STUDY_SEED = 2026

SUBMISSION_TEMPLATE = None
SUBMISSION_LABEL_COLUMN = 'Emotion'
CONFIRM_FILENAME_AS_ID = True
EXPECTED_TEST_ROWS = None if MODE == 'smoke' else 7065
LABEL_NAMES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
ALIASES = {'angry': 0, 'anger': 0, 'fear': 1, 'fearful': 1,
           'happy': 2, 'happiness': 2, 'sad': 3, 'sadness': 3,
           'surprise': 4, 'surprised': 4, 'neutral': 5,
           '愤怒': 0, '恐惧': 1, '快乐': 2, '悲伤': 3, '惊讶': 4, '中性': 5}
PIPELINE_VERSION = 'task2'
assert MODE in {'smoke', 'full'}
if MODE == 'smoke':
    OUTPUT_ROOT = OUTPUT_ROOT / 'smoke_run'
    STUDY_EPOCHS, NUM_WORKERS = 1, 0

OUTPUT_ROOT = OUTPUT_ROOT / 'single_resnet32'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if MODE == 'full' and DEVICE.type != 'cuda':
    raise RuntimeError('正式训练需要在 Kaggle 设置 Accelerator=GPU，再重新运行。')
AMP = DEVICE.type == 'cuda'

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)

ENV = {'pipeline': PIPELINE_VERSION, 'mode': MODE, 'python': platform.python_version(),
       'torch': torch.__version__, 'torchvision': torchvision.__version__,
       'device': str(DEVICE), 'gpu': torch.cuda.get_device_name(0) if AMP else None,
       'batch_size': BATCH_SIZE, 'study_epochs': STUDY_EPOCHS,
       'seed': STUDY_SEED}
save_json(OUTPUT_ROOT / 'environment.json', ENV)
print(ENV)

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def image_files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                  and '__MACOSX' not in p.parts and not p.name.startswith('._'))

def label_directory(path):
    return ALIASES.get(Path(path).name.lower())

def discover_data_root():
    if DATA_ROOT is not None:
        root = Path(DATA_ROOT)
        if not root.exists(): raise FileNotFoundError(root)
        return root
    candidates = [p for p in INPUT_ROOT.rglob('*') if p.is_dir() and p.name == '2026task2']
    candidates += [p for p in INPUT_ROOT.iterdir() if p.is_dir() and '2026task2' in p.name.lower()]
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise RuntimeError(f'不能唯一确定比赛输入：{candidates}。请设置 DATA_ROOT；不要指定包含其他数据集的总目录。')
    return candidates[0]

def extract_if_needed(root):
    if image_files(root): return root
    archives = sorted(root.rglob('*.zip'))
    if not archives: raise RuntimeError(f'{root} 无图像或 ZIP。请添加官方比赛数据。')
    dest = OUTPUT_ROOT / 'extracted'
    dest.mkdir(exist_ok=True)
    for k, archive in enumerate(archives):
        target = dest / f'archive_{k}'
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                out = (target / member.filename).resolve()
                if not out.is_relative_to(target.resolve()): raise ValueError('ZIP 路径越界')
            z.extractall(target)
    return dest

def discover_layout(root):
    roots = []
    for directory in [root] + sorted(p for p in root.rglob('*') if p.is_dir()):
        children = [p for p in directory.iterdir() if p.is_dir()]
        labs = [label_directory(p) for p in children if label_directory(p) is not None]
        if sorted(labs) == list(range(6)): roots.append(directory)
    if len(roots) != 1:
        raise RuntimeError(f'应只有一个含六类目录的训练根目录，发现：{roots}。请设置更精确 DATA_ROOT。')
    train_root = roots[0]
    records = []
    for folder in sorted(train_root.iterdir()):
        label = label_directory(folder)
        if folder.is_dir() and label is not None:
            records += [{'path': str(p), 'label': label} for p in image_files(folder)]
    train = pd.DataFrame(records)
    train_paths = set(train.path)
    others = [p for p in image_files(root) if str(p) not in train_paths]

    test = [p for p in others if any(part.lower().startswith('test') for part in p.relative_to(root).parts[:-1])]
    if not test or len(test) != len(others):
        raise RuntimeError(f'非训练图像 {len(others)} 张，其中测试目录图像 {len(test)} 张；不能猜测未识别图像用途。请检查目录结构。')
    return train.sort_values(['label', 'path']).reset_index(drop=True), test

def read_pixels(path):
    with Image.open(path) as im:
        if im.size != (48, 48): raise ValueError(f'尺寸不是48×48: {path}: {im.size}')
        a = np.array(im.convert('L'), dtype=np.uint8)
    return a

def audit_train(train):
    arrays, rows = [], []
    for idx, row in train.iterrows():
        if idx % 5000 == 0: print(f'Audit {idx}/{len(train)} images', flush=True)
        a = read_pixels(row.path)
        h = hashlib.sha256(a.tobytes()).hexdigest()
        hf = hashlib.sha256(a[:, ::-1].tobytes()).hexdigest()
        arrays.append(a)
        rows.append({'pixel_hash': h, 'group': min(h, hf), 'mean': float(a.mean()/255),
                     'std': float(a.std()/255), 'min': int(a.min()), 'max': int(a.max())})
    return pd.concat([train, pd.DataFrame(rows)], axis=1), np.stack(arrays)

def align_submission(test_paths, roots):
    paths_by_name, paths_by_stem = defaultdict(list), defaultdict(list)
    for p in test_paths:
        paths_by_name[p.name].append(p); paths_by_stem[p.stem].append(p)
    def match(ids):
        matched = []
        for value in ids:

            exact = paths_by_name.get(value, [])
            choices = exact if exact else paths_by_stem.get(value, [])
            if len(choices) != 1: return None
            matched.append(choices[0])
        return matched if len(set(matched)) == len(test_paths) else None
    candidates = [Path(SUBMISSION_TEMPLATE)] if SUBMISSION_TEMPLATE else sorted(set(
        p for root in roots for p in root.rglob('*.csv')))
    valid = []
    for p in candidates:
        try: frame = pd.read_csv(p, dtype=str, keep_default_na=False)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError): continue
        if len(frame) != len(test_paths) or 'ID' not in frame or len(frame.columns) != 2: continue
        if frame.ID.duplicated().any(): continue
        aligned = match(frame.ID.tolist())
        if aligned is not None: valid.append((p, frame, aligned))
    if len(valid) == 1:
        p, frame, aligned = valid[0]
        label_col = next(c for c in frame.columns if c != 'ID')
        print('Submission template:', p, list(frame.columns))
        return frame, aligned, label_col
    if len(valid) > 1: raise RuntimeError(f'有多个合法模板 {[str(v[0]) for v in valid]}，请设置 SUBMISSION_TEMPLATE。')
    if SUBMISSION_LABEL_COLUMN and CONFIRM_FILENAME_AS_ID:
        ordered = sorted(test_paths, key=lambda p: p.name)
        ids = [p.name for p in ordered]
        if len(set(ids)) != len(ids): raise ValueError('测试文件名不唯一')
        return pd.DataFrame({'ID': ids, SUBMISSION_LABEL_COLUMN: 0}), ordered, SUBMISSION_LABEL_COLUMN
    raise RuntimeError('没有找到能完整匹配测试图像的官方示例 CSV。请检查 Input，或按题面设置 SUBMISSION_LABEL_COLUMN 与 CONFIRM_FILENAME_AS_ID；不会猜列名。')

print('开始数据预检', flush=True)
source_root = discover_data_root()
data_root = extract_if_needed(source_root)
train_df, test_paths = discover_layout(data_root)
if EXPECTED_TEST_ROWS and len(test_paths) != EXPECTED_TEST_ROWS:
    raise RuntimeError(f'测试图像为 {len(test_paths)}，官方预期 {EXPECTED_TEST_ROWS}；请确认数据版本。')
sample_submission, test_paths, target_column = align_submission(test_paths, [source_root, data_root])
train_df, X = audit_train(train_df)
X_test = np.stack([read_pixels(p) for p in test_paths])
train_df['relative_path'] = [str(Path(p).relative_to(data_root)) for p in train_df.path]
DATA_FINGERPRINT = digest({'train': train_df[['relative_path', 'label', 'pixel_hash']].to_dict('records'),
                           'test_ids': sample_submission.ID.tolist(),
                           'test_hash': hashlib.sha256(X_test.tobytes()).hexdigest()})
conflicts = train_df.groupby('group').label.nunique()
conflict_groups = set(conflicts[conflicts > 1].index)
train_df[train_df.group.isin(conflict_groups)].to_csv(OUTPUT_ROOT/'label_conflicts.csv', index=False)
AUDIT = {'n_train': len(X), 'n_test': len(X_test), 'classes': LABEL_NAMES,
         'class_counts': {LABEL_NAMES[int(k)]: int(v) for k,v in train_df.label.value_counts().sort_index().items()},
         'duplicate_or_mirror_extra_rows': int(len(X)-train_df.group.nunique()),
         'conflicting_groups': len(conflict_groups), 'data_fingerprint': DATA_FINGERPRINT,
         'submission_columns': sample_submission.columns.tolist()}
save_json(OUTPUT_ROOT/'data_audit.json', AUDIT)
print(AUDIT)

y = train_df.label.to_numpy(dtype=np.int64)
splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=STUDY_SEED)
fold = np.full(len(y), -1, dtype=np.int16)
for k, (_, indices) in enumerate(splitter.split(np.zeros(len(y)), y, train_df.group)):
    fold[indices] = k
train_df['split'] = np.where(fold < 3, 'holdout', np.where(fold < 6, 'val', 'train'))
SPLITS = {s: np.flatnonzero(train_df.split.to_numpy() == s) for s in ['train','val','holdout']}
for s, idx in SPLITS.items():
    assert set(y[idx]) == set(range(6)), f'{s} 未覆盖全部类别'
for a,b in [('train','val'),('train','holdout'),('val','holdout')]:
    assert not (set(train_df.iloc[SPLITS[a]].group) & set(train_df.iloc[SPLITS[b]].group))
train_df.to_csv(OUTPUT_ROOT/'split_manifest.csv', index=False)
split_counts = pd.crosstab(train_df.label, train_df.split).reindex(range(6))
split_counts.index = LABEL_NAMES
split_counts.to_csv(OUTPUT_ROOT/'split_counts.csv')
display(split_counts)
if MODE == 'smoke':

    SPLITS = {s: np.concatenate([idx[y[idx] == c][:12] for c in range(6)]) for s,idx in SPLITS.items()}

train_idx, val_idx, hold_idx = (SPLITS[s] for s in ['train','val','holdout'])
NORM = (float(X[train_idx].mean()/255), max(float(X[train_idx].std()/255), 1e-6))
SPLIT_FINGERPRINT = digest({k:v.tolist() for k,v in SPLITS.items()})
fig, axes = plt.subplots(1, 2, figsize=(11,4))
split_counts.plot.bar(ax=axes[0], rot=25)
axes[0].set_title('Class counts by split'); axes[0].set_ylabel('Images')
axes[1].hist(train_df.iloc[train_idx]['mean'], bins=40)
axes[1].set_title('Training image brightness'); axes[1].set_xlabel('Mean pixel / 255')
fig.tight_layout(); fig.savefig(OUTPUT_ROOT/'data_distribution.png', dpi=180); plt.close(fig)
fig, axes = plt.subplots(6, 6, figsize=(9,9))
for c in range(6):
    ids = train_idx[y[train_idx] == c][:6]
    for j,idx in enumerate(ids):
        axes[c,j].imshow(X[idx], cmap='gray', vmin=0, vmax=255)
        axes[c,j].set_title(LABEL_NAMES[c] if j == 0 else '', fontsize=9); axes[c,j].axis('off')
fig.tight_layout(); fig.savefig(OUTPUT_ROOT/'training_samples.png', dpi=180); plt.close(fig)
majority = int(np.bincount(y[train_idx], minlength=6).argmax())
save_json(OUTPUT_ROOT/'majority_baseline.json', {'majority_label':majority,
          'validation_accuracy':float(np.mean(y[val_idx] == majority))})

def make_transform(augment, norm):
    ops = []
    if augment:
        ops = [T.RandomHorizontalFlip(), T.RandomAffine(10, translate=(.05,.05), scale=(.95,1.05),
                 interpolation=T.InterpolationMode.BILINEAR, fill=128),
               T.ColorJitter(brightness=.15, contrast=.15)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize([norm[0]], [norm[1]])])

class FaceDataset(Dataset):
    def __init__(self, images, labels, indices, augment, norm):
        self.images, self.labels, self.indices = images, labels, np.asarray(indices)
        self.transform = make_transform(augment, norm)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        a = self.transform(Image.fromarray(self.images[idx]))
        return a, int(self.labels[idx]) if self.labels is not None else -1

def loader(indices, augment=False, seed=STUDY_SEED, images=None, labels=None, norm=None, train=False):
    ds = FaceDataset(X if images is None else images, y if labels is None else labels,
                     indices, augment, NORM if norm is None else norm)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train, num_workers=NUM_WORKERS,
                      pin_memory=AMP, drop_last=False, worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=False)

def conv_bn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels,max(channels//8,4),1),
            nn.ReLU(inplace=True), nn.Conv2d(max(channels//8,4),channels,1), nn.Sigmoid())
    def forward(self,x): return x * self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, se=False):
        super().__init__()
        self.main = nn.Sequential(conv_bn(cin,cout,stride), nn.Conv2d(cout,cout,3,1,1,bias=False), nn.BatchNorm2d(cout))
        self.se = SE(cout) if se else nn.Identity()
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)
    def forward(self,x): return self.act(self.se(self.main(x)) + self.skip(x))

class SmallResNet(nn.Module):
    def __init__(self, blocks=3, se=False):
        super().__init__()
        self.stem = conv_bn(1,32)
        layers, cin = [], 32
        for stage, cout in enumerate([32,64,128]):
            for k in range(blocks):
                layers.append(ResidualBlock(cin,cout,2 if stage and k == 0 else 1,se)); cin=cout
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(.2), nn.Linear(128,6))
    def forward(self,x): return self.fc(self.pool(self.features(self.stem(x))).flatten(1))

@dataclass(frozen=True)
class Experiment:
    name: str
    model: str = 'resnet20'
    augment: bool = True
    mixup: float = .2
    smoothing: float = .05
    se: bool = False
    lr: float = .001
    weight_decay: float = .0001

def build_model(cfg):
    if cfg.model == 'plain':
        if cfg.se: raise ValueError('PlainCNN 没有 SE 选项')
        return PlainCNN()
    if cfg.model not in {'resnet20','resnet32'}: raise ValueError(cfg.model)
    return SmallResNet(3 if cfg.model == 'resnet20' else 5, cfg.se)

@torch.inference_mode()
def predict(model, batches, tta=False):
    model.eval(); probs, labels = [], []
    for xb,yb in batches:
        xb=xb.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=AMP):
            p = model(xb).float().softmax(1)
            if tta: p = (p + model(torch.flip(xb, dims=[3])).float().softmax(1))/2
        probs.append(p.cpu().numpy()); labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)

def metrics(labels, probs):
    assert probs.shape == (len(labels),6) and np.isfinite(probs).all()
    preds = probs.argmax(1)
    return {'accuracy':float(accuracy_score(labels,preds)),
            'macro_f1':float(f1_score(labels,preds,labels=range(6),average='macro',zero_division=0)),
            'nll':float(-np.log(np.clip(probs[np.arange(len(labels)),labels],1e-9,1)).mean())}

def checkpoint_load(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def train_epoch(model, batches, optimizer, scaler, cfg):
    model.train(); total, count = 0., 0
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.smoothing)
    for xb,yb in batches:
        xb=xb.to(DEVICE,non_blocking=True); yb=yb.to(DEVICE,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        lam = float(np.random.beta(cfg.mixup,cfg.mixup)) if cfg.mixup > 0 else 1.
        perm = torch.randperm(len(yb),device=DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=AMP):
            logits = model(lam*xb+(1-lam)*xb[perm])
            loss = lam*criterion(logits,yb)+(1-lam)*criterion(logits,yb[perm])
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(optimizer); scaler.update()
        total += loss.item()*len(yb); count += len(yb)
    return total/count

def set_lr(optimizer, base, epoch, epochs):
    warmup = min(3,max(1,epochs//10))
    factor = (epoch+1)/warmup if epoch < warmup else .5*(1+math.cos(math.pi*(epoch-warmup)/max(epochs-warmup,1)))
    for group in optimizer.param_groups: group['lr']=base*factor
    return base*factor

def eval_state(cfg,state,tta=False,ids=None):
    net=build_model(cfg).to(DEVICE);net.load_state_dict(state)
    p,_=predict(net,loader(val_idx if ids is None else ids),tta=tta)
    del net
    return p

def policy_loader(ids,cfg,policy,seed):
    batches=loader(ids,cfg.augment,seed,train=True)
    ops=list(make_transform(cfg.augment,NORM).transforms)
    if policy=='autocontrast':ops.insert(0,T.RandomAutocontrast(p=.25))
    if policy=='erase':ops.append(T.RandomErasing(p=.15,scale=(.02,.08),ratio=(.5,2.),value=0.))
    batches.dataset.transform=T.Compose(ops)
    return batches

def run_one(specification,seed):
    cfg=Experiment(name=specification['name'],model=specification['model'],lr=specification['lr'],se=False)
    identity={'pipeline':PIPELINE_VERSION,'spec':specification,'cfg':asdict(cfg),'seed':seed,
              'data':DATA_FINGERPRINT,'split':SPLIT_FINGERPRINT,'source':'scratch',
              'batch':BATCH_SIZE,'torch':str(torch.__version__),'torchvision':str(torchvision.__version__)}
    folder=OUTPUT_ROOT/'experiments'/f'{cfg.name}_s{seed}_{digest(identity)[:10]}'
    folder.mkdir(parents=True,exist_ok=True);cp=folder/'last.pt'
    if not cp.exists() and RESUME_ROOT:
        resume=Path(RESUME_ROOT)
        if (resume/RUN_GROUP/'experiments').is_dir():resume=resume/RUN_GROUP
        old=resume/'experiments'/folder.name/'last.pt'
        if old.exists():shutil.copy2(old,cp)
    seed_everything(seed);net=build_model(cfg).to(DEVICE)
    opt=torch.optim.AdamW(net.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=AMP)
    history=[];best=None;best_key=(-1.,-float('inf'));best_epoch=0
    if cp.exists():
        state=checkpoint_load(cp)
        if state['identity']!=identity:raise RuntimeError('缓存配置不一致')
        net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        history=state['history'];best=state['best'];best_key=tuple(state['best_key']);best_epoch=state['best_epoch']
    for epoch in range(len(history),specification['epochs']):
        if time.perf_counter()-stage_start>=SESSION_MINUTES*60:break
        seed_everything(seed+epoch);tick=time.perf_counter();lr=set_lr(opt,cfg.lr,epoch,specification['epochs'])
        print(f'{cfg.name} seed={seed} epoch={epoch+1}/{specification["epochs"]} 开始',flush=True)
        loss=train_epoch(net,policy_loader(train_idx,cfg,specification['policy'],seed+epoch),opt,scaler,cfg)
        p,_=predict(net,loader(val_idx));score=metrics(y[val_idx],p)
        if (score['accuracy'],-score['nll'])>best_key:
            best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            best_key=(score['accuracy'],-score['nll']);best_epoch=epoch+1
        history.append({'epoch':epoch+1,'train_loss':loss,'lr':lr,**score,'seconds':time.perf_counter()-tick})
        torch.save({'identity':identity,'model':net.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
                    'history':history,'best':best,'best_key':best_key,'best_epoch':best_epoch},folder/'last.tmp')
        os.replace(folder/'last.tmp',cp);pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        print(cfg.name,score,'已保存',flush=True)
    complete=len(history)==specification['epochs']
    result={'name':cfg.name,'seed':seed,'complete':complete,'epochs':len(history),'budget':specification['epochs'],
            'best_epoch':best_epoch,'directory':str(folder),'cfg':asdict(cfg),'spec':specification,
            'params':sum(p.numel() for p in net.parameters()),'train_val_seconds':sum(r['seconds'] for r in history)}
    del net,opt,scaler;gc.collect()
    if AMP:torch.cuda.empty_cache()
    if not complete:
        save_json(folder/'status.json',result);return result,None
    for tta in [False,True]:
        p=eval_state(cfg,best,tta)
        result['tta' if tta else 'plain']=metrics(y[val_idx],p)
        np.savez_compressed(folder/f'validation_tta{int(tta)}.npz',indices=val_idx,labels=y[val_idx],probs=p)

    clean=eval_state(cfg,best,False,train_idx);result['train_clean']=metrics(y[train_idx],clean)
    result['generalization_gap']=result['train_clean']['accuracy']-result['plain']['accuracy']
    save_json(folder/'status.json',result)
    return result,best

SUBMISSION_NAME = 'submission.csv'
stage_start = time.perf_counter()
specification = {'name':'resnet32_scratch','model':'resnet32','policy':'autocontrast',
                 'lr':.001,'initial':'scratch','se':False,'epochs':STUDY_EPOCHS}
result, state = run_one(specification, STUDY_SEED)
if state is None:
    print('Training paused after an epoch checkpoint; rerun with the same output directory.')
else:
    cfg = Experiment(**result['cfg'])
    model = build_model(cfg).to(DEVICE)
    model.load_state_dict(state)
    probabilities, _ = predict(model, loader(np.arange(len(X_test)), images=X_test,
                                             labels=np.full(len(X_test), -1)), tta=True)
    sample_submission[target_column] = probabilities.argmax(1).astype(np.int64)
    sample_submission.to_csv(OUTPUT_ROOT / 'submission.csv', index=False)
    sample_submission[['ID']].to_csv(OUTPUT_ROOT / 'test_ids.csv', index=False)
    np.save(OUTPUT_ROOT / 'test_probabilities.npy', probabilities)
    torch.save({'state_dict':state,'config':result['cfg'],'norm':list(NORM),'tta':True,
                'label_names':LABEL_NAMES}, OUTPUT_ROOT / 'selected.pt')
    save_json(OUTPUT_ROOT / 'selection.json', {'name':result['name'],'seed':STUDY_SEED,'tta':True,
                                              'accuracy':result['tta']['accuracy'],'cfg':result['cfg']})
