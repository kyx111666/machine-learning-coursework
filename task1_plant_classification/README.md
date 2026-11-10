# 任务 1：植物幼苗分类

使用 500 张竞赛训练图完成五类植物幼苗分类，预测 378 张测试图。正式训练只使用竞赛图片，网络由公开预训练权重初始化。

最终方案包含三步：ConvNeXt-Tiny 五折训练、DINOv2-Base 五折部分微调、DINOv2-Base 全数据三种子重训。所有类别的概率顺序为：Black-grass、Common wheat、Loose Silky-bent、Scentless Mayweed、Sugar beet。

## 代码结构

| 文件 | 用途 |
| --- | --- |
| `train_convnext.py` | 320 分辨率的 ConvNeXt-Tiny，分层五折、早停与八视图 TTA |
| `train_dinov2.py` | 392 分辨率的 DINOv2-Base，五折部分微调，并与 ConvNeXt 等权融合 |
| `train_ensemble.py` | 全部 500 张训练图上训练三个 DINOv2 种子，生成最终融合提交 |
| `data.py` | 数据读取、共享折划分、前景裁剪、概率和提交格式校验 |
| `ablation.py` | 报告使用的训练方式、增强、分辨率、超参数及模型对照实验 |
| `notebooks/` | 可独立上传 Kaggle 的 Notebook，内嵌对应 Python 源码 |
| `tests/` | 使用合成图片及小模型检查数据流、融合、训练与恢复行为 |

## 数据与环境

从 [2026task1 竞赛](https://www.kaggle.com/competitions/2026task1) 获取数据，保留原文件名与目录。官方提交模板的实际文件名为 `submission-for-task2.csv`。

```text
data/
└── 2026task1/
    ├── train/
    │   ├── Black-grass/          # 100 张 PNG
    │   ├── Common wheat/         # 100 张 PNG
    │   ├── Loose Silky-bent/     # 100 张 PNG
    │   ├── Scentless Mayweed/    # 100 张 PNG
    │   └── Sugar beet/           # 100 张 PNG
    ├── test/                    # 378 张 PNG
    └── submission-for-task2.csv
```

使用 Python 3.10 或更新版本、匹配的 PyTorch/torchvision 和 CUDA GPU。首次运行需要联网下载预训练权重；DINOv2 固定使用源码中的 Hub 提交。

在本目录执行：

```bash
python -m pip install -r requirements.txt
python train_convnext.py --data data --out outputs
python train_dinov2.py --data data --out outputs
python train_ensemble.py --data data --out outputs
```

三步共享同一个输出目录，分别生成 `convnext_probabilities.npz`、`dinov2_cv_probabilities.npz` 和最终的 `submission.csv`。提交行顺序严格跟随官方模板。

已有同配方、同五折划分的五个 ConvNeXt checkpoint 时，可在第一步增加 `--weights checkpoint_directory`，读取 `convnext_fold1.pth` 至 `convnext_fold5.pth` 重新计算概率。DINOv2 两步会保存逐轮训练状态及已完成的预测，在相同输入和配置下可从相同输出目录继续运行。

## Kaggle Notebook

按顺序运行：

1. `notebooks/train_convnext.ipynb`：添加竞赛数据，开启 GPU 和 Internet。
2. `notebooks/train_dinov2.ipynb`：添加竞赛数据和第一步保存的 `convnext_probabilities.npz`。
3. `notebooks/train_ensemble.ipynb`：添加竞赛数据和前两步的两个概率文件。

概率文件可通过前一步 Notebook Output 或自己创建的 Dataset 添加到 Input。每步只挂一个所需文件的来源，避免重复命名。最终下载 `/kaggle/working/submission.csv`。

Notebook 由仓库根目录的 `python tools/build_notebooks.py` 生成；均不含运行输出或预先打包的特征和预测。

## 模型与融合

ConvNeXt 使用 torchvision 的 ImageNet-1k 预训练权重、320 输入、batch 12、AdamW（学习率 1e-4，weight decay 1e-4）、label smoothing 0.05，最多 20 轮，按验证 Macro-F1 选 checkpoint，连续五轮未改善时早停。预测平均四个 90° 旋转及其水平翻转，共八个视图。

DINOv2 使用带 register tokens 的 ViT-B/14。拼接 CLS 和 patch 均值特征后，经 LayerNorm 和线性层输出五类。冻结前八个模块，仅更新最后四个模块、末端归一化及分类头。输入 392，全程 FP32，固定 12 轮，前两轮只训练分类头；主干学习率 1e-5，分类头 3e-4，batch 2，梯度累积 8，EMA decay 0.95。推理平均整图、整图翻转、前景裁剪和裁剪翻转四个视图。

设 C 为 ConvNeXt 五折平均概率，D 为 DINOv2 五折平均概率，F 为全数据三种子（2026、3407、9187）DINOv2 平均概率：

```text
五折融合 = 0.5 × C + 0.5 × D
最终预测 = 0.5 × 五折融合 + 0.5 × F
```

五折统一使用 `StratifiedKFold(5, shuffle=True, random_state=2026)`。每张训练图恰好在一个验证折中被预测；全数据重训没有独立留出集，因此不报告它自身的 OOF 分数。

## 报告结果

以下为课程报告中记录的原实验成绩，指标为准确率（本题等于 micro-F1）。

| 方法 | OOF 准确率 | Kaggle Public |
| --- | ---: | ---: |
| ConvNeXt-Tiny 五折 | 0.938 | 0.9629 |
| DINOv2-Base 五折 | 0.956 | — |
| ConvNeXt + DINOv2 五折融合 | 0.960 | 0.9708 |
| 五折融合 + 全数据三种子重训 | — | **0.9735** |
| 补充实验的 timm DINOv2-Base | 0.966 | 0.9656 |

补充实验中的 timm DINOv2 使用不同的分类头和训练细节，不等同于最终模型的严格复现。多次使用同一组折进行方案比较，OOF 只能作为探索性验证；重新训练也可能因随机性和软件环境得到不同结果。

## 消融实验

```bash
python ablation.py --list
python ablation.py --data data --out results --exp base aug_none aug_flip aug_flip_rot
python ablation.py --data data --out results --exp res_224 res_448
python ablation.py --data data --out results --exp dinov2_base_partial
```

每个配置执行五折，保存每轮损失、验证指标和逐视图概率，可用于分析增强、TTA、分辨率与模型差异。消融基线为 timm ConvNeXt-Tiny、固定 12 轮，与最终训练入口的 torchvision 基线配置有区别。提高分辨率、随机初始化及模型适配配置同时调整了其他参数，需按完整配置比较。

在本目录运行检查：

```bash
python -m unittest discover -s tests -v
```

发布整理已用合成数据检查训练边界、固定融合、输入校验和断点恢复；未重新执行完整的 Kaggle GPU 训练。
