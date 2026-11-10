# 任务 4：眼底血管分割

在官方 20 张有标注眼底图像上训练，预测 20 张测试图的血管掩码。评价指标为 Dice，越高越好；提交文件为 `Id,Predicted` 两列，掩码按官方脚本转为 512×512 的游程编码。

## 最终采用的两套方案

| 方案 | Public Dice | Private Dice | 正式入口 | 生成的提交文件 |
| --- | ---: | ---: | --- | --- |
| 单 Gamma 三分支融合 | **0.86187** | **0.86424** | `train.py --route gamma` | `outputs/gamma/selected/submission.csv` |
| 双 RGB 种子与绿通道融合 | **0.86150** | **0.86406** | `train.py --route seed_blend` | `outputs/seed_blend/selected/submission.csv` |

成绩对应最终采用的原始比赛提交。两套方案均为随机初始化，只使用官方训练图像和标注，不需要预训练权重、历史 Notebook 输出或外部数据。这里的正式入口从头训练各自所需的 15 个折模型；训练环境及随机计算差异可能使重新训练后的数值与原比赛运行有所不同。

## 代码结构

| 文件 | 用途 |
| --- | --- |
| `train.py` | 两套固定方案的训练、五折推理、概率融合与提交入口 |
| `runtime.py` | 官方数据、图像级五折、裁块增强、U-Net、基础训练器及官方掩码/RLE 协议 |
| `training.py` | 独立种子训练器，记录成功更新数、AMP 跳过数和完成状态 |
| `predictions.py` | 与权重和数据绑定的概率缓存、固定后处理及逐图指标 |
| `gamma.py` | 逐通道 Gamma 增强和分阶段配对实验 |
| `experiments.py` | 16 项筛选、消融、超参数对照及 Gamma 配对实验入口 |
| `notebooks/` | 内嵌全部源码的独立 Kaggle Notebook |
| `tests/` | 不训练模型的增强、融合、后处理和提交检查 |

## 数据与运行

从 [2026task-4 竞赛](https://www.kaggle.com/competitions/2026task-4) 获取官方数据，保留如下目录和文件名：

```text
data/
└── segmentation/
    ├── train/
    │   ├── image/                 # 20 张 584×565 RGB 图像
    │   └── label/                 # 对应的血管标签
    ├── test/
    │   └── image/                 # 20 张测试图像
    ├── sample_submission.csv      # Id,Predicted
    └── segmentation_to_csv.py     # 官方转换脚本
```

代码按实际官方标签的白色血管前景读取，并检查二值标签、图像尺寸、重复图像、文件指纹和提交顺序。训练图按数值 ID 排序，使用 `KFold(5, shuffle=True, random_state=3407)`，每折 16 张训练、4 张验证，裁块不会跨原图划分。

使用 Python 3.10 或更新版本及匹配的 PyTorch/torchvision。训练入口要求在 Kaggle GPU 环境执行。最直接的方法是上传以下 Notebook，只添加官方数据并启用 GPU：

- `notebooks/train_gamma.ipynb`：单 Gamma 最终方案。
- `notebooks/train_seed_blend.ipynb`：双 RGB 种子最终方案。
- `notebooks/experiments_screening.ipynb`：模型、超参数与消融筛选。
- `notebooks/experiments_gamma.ipynb`：Gamma 配对实验及确认。

Notebook 内嵌源码，无需添加本仓库或旧 Notebook Output。Kaggle 预装 PyTorch/torchvision；其余依赖若尚未安装，需要临时联网安装。模型本身不下载任何权重。

也可将代码放到 Kaggle 工作目录，在本目录执行：

```bash
python -m pip install -r requirements.txt
python train.py --data /kaggle/input --out outputs --route gamma
python train.py --data /kaggle/input --out outputs --route seed_blend
```

同一输出目录复用配置与数据一致的已完成折；未完成折从头重训，不从中断的优化器状态恢复。修改源码、数据或参数后使用新的输出目录。保留完整 Notebook Output，其中包含模型、概率、配置和训练日志。

## 模型与训练

RGB 分支使用残差 U-Net；绿通道分支使用普通 U-Net。编码器通道为 24、48、96、192、384，解码器逐级双线性上采样，经 1×1 卷积和跳连拼接后恢复分辨率。卷积块含两层 3×3 卷积、BatchNorm 和 SiLU；输出为单通道血管 logit。

绿通道输入由原绿通道、CLAHE 绿通道和局部对比度通道组成。训练均匀抽取 192×192 裁块，采用 0.8、1、1.25 尺度、随机旋转、水平/垂直翻转、亮度和噪声增强；图像线性插值、标签最近邻插值，输入归一化到 [−1,1]。Gamma 方案仅在第二个 RGB 分支的训练裁块上，以 50% 概率分别对 R/G/B 施加独立的 log-uniform[0.5,2] 指数，验证和测试仍输入原 RGB。

损失为等权 BCE 与 Dice，Dice 平滑项为 1。优化器为 AdamW，学习率 8×10⁻⁴、权重衰减 10⁻⁴、batch 12，预热后余弦衰减，梯度范数裁剪为 5。EMA 平均浮点模型参数，BatchNorm 统计量直接复制。每 250 步使用验证图整图推理，以无 TTA、阈值 0.5、无连通块过滤的官方网格代理 Dice 保存最佳检查点。

单 Gamma 方案的三个分支均使用 `training.py` 的成功更新计数器，每折完成 4,000 次有效更新，AMP 跳过的更新会补足。双 RGB 种子方案的 RGB1 和 green 保留原始 `runtime.py` 基础训练器的 4,000 批次协议，RGB2 使用成功更新计数器；两种训练器各自保留原始检查点逻辑。RGB1、green 的种子为 20261031，RGB2 为 20261102，裁块随机流还包含折编号。

## 固定融合与提交协议

设 R1、R2 为两个 RGB 分支的概率，G 为绿通道概率，Rγ 为 Gamma 训练的第二个 RGB 分支：

```text
双 RGB 种子方案 = 0.75 × (0.5 × R1 + 0.5 × R2) + 0.25 × G
单 Gamma 方案   = 0.375 × R1 + 0.375 × Rγ + 0.25 × G
```

每个分支对训练图使用对应折外模型，对测试图等权平均五折。每个折模型使用原图、水平翻转、垂直翻转及双翻转四种输入；原图反射补边至 16 的倍数，输出翻转回原方向后裁掉补边，平均 sigmoid 概率。两套公式保留原始 float32 运算顺序。

融合后先在 584×565 概率图上用 **0.425** 阈值二值化，再按官方脚本双线性缩放到 **512×512**，取非零像素，删除小于 **8** 像素的 8 连通分量。RLE 使用 C 序、一基起点；程序核对样例 ID 顺序、RLE 往返，并将生成的 PNG 逐张交给官方 `turn_to_str` 比对。

原双 RGB 种子方案的 OOF 代理 Dice 为 **0.852198**；单 Gamma 为 **0.853706**。单 Gamma 的同轮新训练参考为 **0.852389**，配对增益 **0.001318**，五折均改善。模型、检查点与后处理曾在这些图像上反复比较，这些 OOF 数值用于描述开发结果，与 Public/Private 成绩分开报告。

正式入口固定使用已选方案，不重新按 OOF 搜索分支或阈值。每套方案保存 `metrics.json`、`oof_per_image.csv`、分量及融合概率、15 个折模型和最终 CSV。

## 有效性消融、超参数与模型比较

```bash
python experiments.py --data /kaggle/input --out experiment_outputs --group screening
python experiments.py --data /kaggle/input --out experiment_outputs --group gamma
```

筛选组保持原实验设计：第 0 折比较 16 种配置，每项 1,000 步；短名单前三项在第 1、2 折各训练 2,000 步。覆盖普通/残差 U-Net、上下文模块、RGB/原绿通道/绿通道特征、宽度、学习率、裁块大小、Dice 权重、增强与多尺度裁块。另在同一检查点上比较原参数/EMA 与有无翻转 TTA，保存 `screening.csv`、`ablation.csv`、`paired_ema_tta.csv` 和确认表。

Gamma 配对组从官方数据重训同初始化的原 RGB2/Gamma RGB2，先检查裁块标签完全一致，再按第 0 折、第 0/2/4 折单 RGB、三折完整融合、完整五折依次检查。单 RGB 三折需平均增益至少 0.001 且至少两折改善；完整融合需至少 0.0005，五折还需至少三折改善、新增第 1/3 折平均增益为正且至少一折改善。通过后才预测测试图，最多训练 20 个折模型；拒绝时保存比较结果而不生成候选提交。

筛选比较和正式 4,000 步训练使用不同预算，分数分别记录。Gamma 配对实验会重新作出采用判断；它与直接训练固定最终方案的 `train_gamma.ipynb` 是两个入口。

## 不训练模型的检查

```bash
python -m unittest discover -s tests -v
```

检查覆盖 RLE、原生阈值后官方缩放、8 连通过滤、融合运算、配置、Gamma 标签同步与确认门槛；不实例化模型、不创建优化器。四个 Notebook 可在仓库根目录用 `python tools/build_task4_notebooks.py` 重新生成。
