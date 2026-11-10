"""Scratch heatmap/regression and ImageNet ResNet component training; exact original affine/decode protocol."""
from pathlib import Path
import copy, hashlib, json, math, os, random, re, time, zipfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
from torchvision.ops.misc import FrozenBatchNorm2d
from data import FROZEN_FOLDS, SPLIT2_FOLDS

SEED = 2026
PIPELINE_VERSION = 'fovea-component-training'
MODES = ('global', 'sharp', 'local9')
cv2.setNumThreads(0)
torch.set_num_threads(2)
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rmse(target, pred):
    return float(np.sqrt(np.mean((np.asarray(target)-np.asarray(pred))**2)))

def key(value):
    stem = Path(str(value)).stem
    return str(int(stem)) if stem.isdigit() else stem

def apply_point(xy, M):
    point = np.r_[xy, 1.] @ M.T
    return point[:2] / point[2]

def load_official(root, split=1):
    paths = list(Path(root).rglob('fovea_localization_train_GT.csv'))
    assert len(paths)==1, 'Mount exactly the official 2026task3 competition data'
    base=paths[0].parent
    labels=pd.read_csv(paths[0],dtype={'data':str})
    assert {'data','Fovea_X','Fovea_Y'} <= set(labels)
    lookup={key(p.name):p for p in (base/'train').glob('*.jpg')}
    records=[]; pixels={}
    for r in labels.to_dict('records'):
        path=lookup[key(r['data'])]
        rgb=np.array(Image.open(path).convert('RGB'))
        h,w=rgb.shape[:2]; xy=np.array([r['Fovea_X'],r['Fovea_Y']],dtype=float)
        assert np.isfinite(xy).all() and ((xy>=0)&(xy<[w,h])).all()
        digest=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()

        cw=round(w*min(1,768/max(w,h))); ch=round(h*min(1,768/max(w,h)))
        small=cv2.resize(rgb,(cw,ch),interpolation=cv2.INTER_AREA)

        mask=(small.max(2)>20).astype('uint8')
        n,comp,stats,_=cv2.connectedComponentsWithStats(mask,8)
        if n>1:
            x,y,bw,bh,area=stats[1+np.argmax(stats[1:,4])]
            margin=round(.015*max(cw,ch)); x0=max(0,x-margin); y0=max(0,y-margin)
            x1=min(cw,x+bw+margin); y1=min(ch,y+bh+margin)
        else: x0,y0,x1,y1=0,0,cw,ch
        pixels[path.name]=(small,(x0,y0,x1,y1))
        records.append(dict(image=path.name,path=str(path),width=w,height=h,cw=cw,ch=ch,
                            x=xy[0],y=xy[1],visible=bool(xy.any()),group=digest))
    df=pd.DataFrame(records)
    assert len(df)==80 and len(lookup)==80 and len(list((base/'test').glob('*.jpg')))==20
    assert df.image.nunique()==80

    if split not in (1, 2):
        raise ValueError('Choose grouped split 1 or 2')
    assignment = FROZEN_FOLDS if split == 1 else SPLIT2_FOLDS
    df['fold'] = [int(assignment[Path(name).stem]) for name in df.image]
    for fold in range(5):
        assert set(df[df.fold!=fold].group).isdisjoint(df[df.fold==fold].group)
    assert (df.fold>=0).all()
    sample=pd.read_csv(base/'sample_submission.csv')
    assert list(sample)==['ImageID','value'] and len(sample)==40
    return base,df,pixels,sample

class Fundus(Dataset):
    def __init__(self,df,pixels,cfg,train=False):
        self.df=df.reset_index(drop=True); self.pixels=pixels; self.cfg=cfg; self.train=train
    def __len__(self): return len(self.df)*(self.cfg['repeat'] if self.train else 1)
    def __getitem__(self,i):
        r=self.df.iloc[i%len(self.df)]; image,box=self.pixels[r.image]
        h,w=image.shape[:2]; s=self.cfg['size']
        x0,y0,x1,y1=box if self.cfg['crop'] else (0,0,w,h)
        M=np.array([[s/(x1-x0),0,-x0*s/(x1-x0)],[0,s/(y1-y0),-y0*s/(y1-y0)],[0,0,1.]])
        original=np.array([r.x/r.width*w,r.y/r.height*h])
        if self.train and self.cfg['geometry']:
            a=cv2.getRotationMatrix2D((s/2,s/2),np.random.uniform(-15,15),np.random.uniform(.9,1.1))
            a[:,2]+=np.random.uniform(-.04,.04,2)*s
            A=np.vstack([a,[0,0,1]])
            if np.random.rand()<.5: A=np.array([[-1,0,s-1],[0,1,0],[0,0,1.]])@A
            trial=A@M; pt=apply_point(original,trial)
            if not r.visible or ((pt>=0)&(pt<s)).all(): M=trial
        if self.cfg['clahe']:
            lab=cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
            lab[:,:,0]=cv2.createCLAHE(2.,(8,8)).apply(lab[:,:,0])
            image=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
        image=cv2.warpAffine(image,M[:2],(s,s),flags=cv2.INTER_LINEAR).astype('float32')/255
        if self.train and self.cfg['photo']:
            image=np.clip((image-.5)*np.random.uniform(.8,1.2)+.5+np.random.uniform(-.08,.08),0,1)
            image=np.power(image,np.random.uniform(.85,1.15)).astype('float32')
        xy=apply_point(original,M)/s if r.visible else np.array([0.,0.])

        M_original=M@np.diag([w/r.width,h/r.height,1.])
        return (torch.from_numpy(image.transpose(2,0,1).copy()),torch.tensor(xy,dtype=torch.float32),
                torch.tensor(r.visible,dtype=torch.float32),torch.tensor(np.linalg.inv(M_original),dtype=torch.float32))

def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU(),
                         nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU())

class ScratchLocator(nn.Module):
    def __init__(self,width=24,model='heatmap'):
        super().__init__(); self.model=model
        self.e1=block(3,width); self.e2=block(width,width*2); self.e3=block(width*2,width*4)
        self.e4=block(width*4,width*8); self.center=block(width*8,width*8)
        self.visibility=nn.Linear(width*8,1)
        if model=='heatmap':
            self.d3=block(width*12,width*4); self.d2=block(width*6,width*2)
            self.head=nn.Conv2d(width*2,1,1)
        else:
            self.reg=nn.Sequential(nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(width*8*16,128),nn.SiLU(),nn.Linear(128,2))
    def forward(self,x):
        a=self.e1(F.avg_pool2d(x,2)); b=self.e2(F.avg_pool2d(a,2)); c=self.e3(F.avg_pool2d(b,2))
        d=self.e4(F.avg_pool2d(c,2)); z=self.center(F.avg_pool2d(d,2))
        vis=self.visibility(z.mean((2,3))).squeeze(1)
        if self.model!='heatmap': return self.reg(z).sigmoid(),vis,None
        u=self.d3(torch.cat([F.interpolate(z,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        u=self.d2(torch.cat([F.interpolate(u,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        logits=self.head(u).squeeze(1); h,w=logits.shape[-2:]
        prob=logits.flatten(1).softmax(1).view(-1,h,w)
        gx=(torch.arange(w,device=x.device,dtype=x.dtype)+.5)/w
        gy=(torch.arange(h,device=x.device,dtype=x.dtype)+.5)/h
        xy=torch.stack([(prob.sum(1)*gx).sum(1),(prob.sum(2)*gy).sum(1)],1)
        return xy,vis,logits

def loss_fn(output,xy,visible,cfg):
    pred,vis,logits=output
    coordinate=((pred-xy)**2).sum(1)
    loss=(coordinate*visible).sum()/visible.sum().clamp_min(1)*20
    if logits is not None:
        h,w=logits.shape[-2:]; dtype=logits.dtype; device=logits.device
        gx=(torch.arange(w,device=device,dtype=dtype)+.5)/w
        gy=(torch.arange(h,device=device,dtype=dtype)+.5)/h
        sq=(gx[None,None,:]-xy[:,0,None,None])**2+(gy[None,:,None]-xy[:,1,None,None])**2
        target=torch.exp(-sq/(2*(cfg['sigma']/cfg['size'])**2)).flatten(1)
        target=target/target.sum(1,keepdim=True).clamp_min(1e-20)
        ce=-(target*logits.flatten(1).log_softmax(1)).sum(1)
        loss=loss+(ce*visible).sum()/visible.sum().clamp_min(1)
    return loss+F.binary_cross_entropy_with_logits(vis,visible)*.1

@torch.inference_mode()
def predict(model,loader,device,size,tta=False):
    model.eval(); preds=[]; visibilities=[]
    for image,xy,visible,inverse in loader:
        image=image.to(device); xyhat,vis,_=model(image); vs=vis.sigmoid()
        if tta:
            xflip,vflip,_=model(image.flip(-1)); xflip[:,0]=1-1/size-xflip[:,0]
            xyhat=(xyhat+xflip)/2; vs=(vs+vflip.sigmoid())/2
        netxy=xyhat.cpu()*size
        hom=torch.cat([netxy,torch.ones(len(netxy),1)],1)
        raw=torch.bmm(inverse,hom.unsqueeze(-1)).squeeze(-1)
        raw=raw[:,:2]/raw[:,2:3]

        preds.append(raw.numpy()); visibilities.append(vs.cpu().numpy())
    return np.concatenate(preds),np.concatenate(visibilities)

def checkpoint_fingerprint(cfg,fold,df):
    identity=dict(pipeline=PIPELINE_VERSION,cfg=cfg,fold=fold,seed=SEED,
        images=df[['image','group','x','y','fold']].to_dict('records'))
    if cfg['accumulate']>1:
        identity['accumulation_rule']='sample_weighted_partial_group'
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

def test_pixels(base):
    pixels={}; rows=[]
    for path in sorted((base/'test').glob('*.jpg')):

        rgb=np.array(Image.open(path).convert('RGB')); h,w=rgb.shape[:2]
        cw=round(w*min(1,768/max(w,h))); ch=round(h*min(1,768/max(w,h)))
        im=cv2.resize(rgb,(cw,ch),interpolation=cv2.INTER_AREA)
        n,_,stats,_=cv2.connectedComponentsWithStats((im.max(2)>20).astype('uint8'),8)
        if n>1:
            x,y,bw,bh,_=stats[1+np.argmax(stats[1:,4])]; m=round(.015*max(cw,ch))
            box=(max(0,x-m),max(0,y-m),min(cw,x+bw+m),min(ch,y+bh+m))
        else: box=(0,0,cw,ch)
        pixels[path.name]=(im,box)
        rows.append(dict(image=path.name,width=w,height=h,cw=cw,ch=ch,x=0.,y=0.,visible=False))
    return pd.DataFrame(rows),pixels
def decode_heatmap(logits,mode):
    h,w=logits.shape[-2:]
    z=logits.flatten(1)
    if mode=='sharp':z=z/.5
    elif mode=='local9':
        peak=z.argmax(1); py=peak//w;px=peak%w
        yy=torch.arange(h,device=z.device)[None,:,None];xx=torch.arange(w,device=z.device)[None,None,:]
        mask=((yy-py[:,None,None]).abs()<=4)&((xx-px[:,None,None]).abs()<=4)
        z=z.masked_fill(~mask.flatten(1),-torch.inf)
    else:assert mode=='global'
    prob=z.softmax(1).reshape(-1,h,w)
    gx=(torch.arange(w,device=z.device,dtype=z.dtype)+.5)/w
    gy=(torch.arange(h,device=z.device,dtype=z.dtype)+.5)/h
    return torch.stack([(prob.sum(1)*gx).sum(1),(prob.sum(2)*gy).sum(1)],1)

@torch.inference_mode()
def predict_decoders(model,loader,device,size):
    model.eval(); result={mode:[] for mode in MODES}
    for image,_,_,inverse in loader:
        _,_,logits=model(image.to(device));assert logits is not None
        for mode in MODES:
            xy=decode_heatmap(logits,mode).cpu()*size
            hom=torch.cat([xy,torch.ones(len(xy),1)],1)
            raw=torch.bmm(inverse,hom.unsqueeze(-1)).squeeze(-1)
            result[mode].append((raw[:,:2]/raw[:,2:3]).numpy())
    return {mode:np.concatenate(value) for mode,value in result.items()}
def train_fold(name,cfg,fold,df,pixels,out,device):
    dest=out/name/f'fold{fold}'; dest.mkdir(parents=True,exist_ok=True)
    fingerprint=checkpoint_fingerprint(cfg,fold,df)
    if (dest/'complete.json').exists():
        old=json.loads((dest/'complete.json').read_text()); assert old['fingerprint']==fingerprint
        return
    seed_all(SEED+fold+cfg.get("seed_offset",0))
    tr=df[df.fold!=fold]; va=df[df.fold==fold]
    train=DataLoader(Fundus(tr,pixels,cfg,True),batch_size=cfg['batch'],shuffle=True,num_workers=0)
    valid=DataLoader(Fundus(va,pixels,cfg),batch_size=cfg['batch'],num_workers=0)
    model=Locator(cfg['width'],cfg['model']).to(device)
    if hasattr(model,'encoder'):
        encoder_ids={id(p) for p in model.encoder.parameters()}
        groups=[dict(params=model.encoder.parameters(),lr=cfg['encoder_lr']),
                dict(params=[p for p in model.parameters() if id(p) not in encoder_ids],lr=cfg['lr'])]
        opt=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda t:.01+.99*.5*(1+math.cos(math.pi*t/cfg['epochs'])))
    else:
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'],eta_min=cfg['lr']*.01)
    history=[]; start=0; average=None; n_average=0

    if (dest/'last.pt').exists():
        state=torch.load(dest/'last.pt',map_location='cpu',weights_only=False)
        assert state['fingerprint']==fingerprint
        model.load_state_dict(state['model']); opt.load_state_dict(state['opt']); sched.load_state_dict(state['sched'])
        start=state['epoch']+1; history=state['history']; average=state['average']; n_average=state['n_average']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start,cfg['epochs']):
        begin=time.time(); model.train(); total=0.; count=0
        opt.zero_grad(set_to_none=True)
        for step,(image,xy,visible,_) in enumerate(train):
            image=image.to(device); xy=xy.to(device); visible=visible.to(device)
            loss=loss_fn(model(image),xy,visible,cfg)
            assert torch.isfinite(loss),f'Nonfinite loss in {name} fold {fold}'
            group_start=(step//cfg['accumulate'])*cfg['batch']*cfg['accumulate']
            group_samples=min(cfg['batch']*cfg['accumulate'],len(train.dataset)-group_start)
            (loss*len(image)/group_samples).backward()
            if (step+1)%cfg['accumulate']==0 or step+1==len(train):
                nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); opt.zero_grad(set_to_none=True)
            total+=float(loss.detach())*len(image); count+=len(image)
        sched.step()

        if epoch>=cfg['epochs']-10:
            weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if average is None: average=weights
            else:
                for k in average: average[k].add_((weights[k]-average[k])/(n_average+1))
            n_average+=1
        record=dict(epoch=epoch+1,train_loss=total/count,lr=opt.param_groups[0]['lr'],seconds=time.time()-begin)
        if (epoch+1)%5==0 or epoch==0:
            p,_=predict(model,valid,device,cfg['size'])
            record['val_rmse']=rmse(va[['x','y']].values,p)
        history.append(record)
        state=dict(fingerprint=fingerprint,model={k:v.detach().cpu() for k,v in model.state_dict().items()},opt=opt.state_dict(),sched=sched.state_dict(),
                   epoch=epoch,history=history,average=average,n_average=n_average,
                   python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        torch.save(state,dest/'last.tmp'); os.replace(dest/'last.tmp',dest/'last.pt')
        pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
        if 'val_rmse' in record: print(name,fold,record,flush=True)
    assert n_average==10
    model.load_state_dict(average)
    p,v=predict(model,valid,device,cfg['size']); pt,vt=predict(model,valid,device,cfg['size'],True)
    result=va[['image','fold','x','y','group']].copy()
    result['pred_x']=p[:,0]; result['pred_y']=p[:,1]; result['tta_x']=pt[:,0]; result['tta_y']=pt[:,1]
    result['visibility']=v; result.to_csv(dest/'oof.csv',index=False)
    torch.save(dict(model=average,cfg=cfg,fingerprint=fingerprint,fold=fold),dest/'model.pt')
    summary=dict(fingerprint=fingerprint,rmse=rmse(va[['x','y']],p),tta_rmse=rmse(va[['x','y']],pt),
                 cfg=cfg,fold=fold,seconds=float(sum(r['seconds'] for r in history)))
    (dest/'complete.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',name,fold,summary,flush=True)
    del model,opt,state; torch.cuda.empty_cache()

def freeze_batch_norm(module):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.BatchNorm2d):
            frozen=FrozenBatchNorm2d(child.num_features,eps=child.eps)
            with torch.no_grad():
                for attr in ['weight','bias','running_mean','running_var']:
                    getattr(frozen,attr).copy_(getattr(child,attr))
            setattr(module,name,frozen)
        else:freeze_batch_norm(child)

class ResNetLocator(nn.Module):
    def __init__(self,model):
        super().__init__();self.model='heatmap'
        backbone='resnet34' if '34' in model else 'resnet18'
        self.encoder=(resnet34 if backbone=='resnet34' else resnet18)(weights=None)
        if 'imagenet' in model:
            weights = ResNet34_Weights.IMAGENET1K_V1 if backbone == 'resnet34' else ResNet18_Weights.IMAGENET1K_V1
            self.encoder.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
        self.encoder.fc=nn.Identity();freeze_batch_norm(self.encoder)
        self.d4=block(768,128);self.d3=block(256,64);self.d2=block(128,32)
        self.head=nn.Conv2d(32,1,1);self.visibility=nn.Linear(512,1)
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(self,x):
        x=(x-self.rgb_mean)/self.rgb_std;e=self.encoder
        x=e.maxpool(e.relu(e.bn1(e.conv1(x))))
        a=e.layer1(x);b=e.layer2(a);c=e.layer3(b);z=e.layer4(c)
        u=self.d4(torch.cat([F.interpolate(z,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        u=self.d3(torch.cat([F.interpolate(u,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        u=self.d2(torch.cat([F.interpolate(u,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        logits=self.head(u).squeeze(1)
        return decode_heatmap(logits,'global'),self.visibility(z.mean((2,3))).squeeze(1),logits

def Locator(width=24, model='heatmap'):
    if model.startswith('resnet'):
        return ResNetLocator(model)
    if model == 'heatmap_dropout':
        net = ScratchLocator(width, 'heatmap')
        for name in ['e1', 'e2', 'e3', 'e4', 'center', 'd3', 'd2']:
            getattr(net, name).add_module('dropout', nn.Dropout2d(.1))
        return net
    return ScratchLocator(width, model)
