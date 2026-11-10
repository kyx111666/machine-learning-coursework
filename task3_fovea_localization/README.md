# 任务 3：眼底中心凹定位

使用官方 80 张训练图及中心凹坐标，预测 20 张测试图的原图像素坐标。评价指标为坐标 RMSE，数值越小越好；官方提交把每张图的 X、Y 分为两行，共 40 行。

## 最终选用的两套方案

| 最终选用方案 | Public RMSE | Private RMSE | 生成的提交文件 |
| --- | ---: | ---: | --- |
| 粗到细级联、独立 ResNet34、两套划分 Dropout200 的四分量融合 | **11.04242** | **19.53134** | `submissions/submission_four_component.csv` |
| 粗到细级联与第一套划分 Dropout200 的 70/30 融合 | **11.38818** | **20.61907** | `submissions/submission_cascade_scratch.csv` |

两套方案共享粗到细级联和第一套划分的 Dropout200。四分量方案进一步加入独立 ResNet34 与第二套划分的 Dropout200。全部模型只使用官方训练图像；预训练参数取自 ImageNet。

## 代码结构

| 文件 | 内容 |
| --- | --- |
| `train.py` | 粗定位、精定位及固定级联的训练入口 |
| `recipes.py` | 实际完成的 25 组粗到细实验配置、最终成员与解码设置 |
| `data.py` | 官方数据读取、分组五折、视野几何、图像金字塔和标签同步增强 |
| `models.py`、`training.py` | timm 编码器热图网络、训练损失、EMA 和仿射 TTA |
| `stages.py`、`inference.py`、`cascade.py` | 分折训练、保存权重重推理、精定位窗口重定心和级联集成 |
| `classic.py` | 从零热图/坐标回归与 TorchVision ResNet 的独立训练协议 |
| `train_components.py` | Dropout200 两套划分、独立 ResNet34、相关消融及模型对照 |
| `blend.py`、`submission.py` | 两套固定坐标融合、输入校验和官方提交格式 |
| `notebooks/` | 自包含的 Kaggle 训练及实验入口 |
| `tests/` | 几何变换、解码、模型接口、提交格式及合成数据流程检查 |

## 数据与环境

从 [2026task3 竞赛](https://www.kaggle.com/competitions/2026task3) 获取官方数据，保留原文件名：

```text
data/
└── detection/
    ├── train/                              # 80 张 JPG
    ├── test/                               # 20 张 JPG
    ├── fovea_localization_train_GT.csv      # data,Fovea_X,Fovea_Y
    └── sample_submission.csv               # ImageID,value，40 行
```

使用 Python 3.10 或更新版本、匹配的 PyTorch/torchvision、CUDA GPU。首次运行预训练网络需要联网下载 ImageNet 权重。命令均在本目录执行：

```bash
python -m pip install -r requirements.txt
python train.py --data data --out outputs --group final
python train_components.py --data data --out outputs --group final
python blend.py --data data --results outputs --recipe four_component --out submissions
python blend.py --data data --results outputs --recipe cascade_scratch --out submissions
```

第一步训练 6 个粗定位配置和 4 个精定位配置，每个配置五折，生成 `outputs/cascade_predictions.npz`。第二步训练 3 个独立分量，每个分量五折，生成 `outputs/components/<分量名>/predictions_sharp.npz`。后两步分别生成两份最终提交。

只运行指定分量时：

```bash
python train_components.py --data data --out outputs --run dropout_split1
python train_components.py --data data --out outputs --run dropout_split2
python train_components.py --data data --out outputs --run resnet34_transfer
```

在相同输出目录继续运行会复用已完成的折。独立分量还保存逐轮训练状态，可从轮间 checkpoint 恢复；粗到细网络在单折中断后重跑该折。修改配置或输入数据时使用新的输出目录。

## 模型与训练方法

### 粗到细级联

粗定位使用 timm 的 ImageNet ResNet18/34 多尺度特征和热图解码器。输入主要为 256×256，输出步长为 4 的热图；损失为归一化高斯目标交叉熵加期望坐标 L1，使用 AdamW、预热加余弦调度和 EMA。每折取固定训练终点的 EMA 权重，验证指标只用于记录。

最终粗定位等权平均 6 个配置：ResNet34 两个种子、ResNet18 两个种子、冻结 BatchNorm 统计量的 ResNet34、加入 Dropout/随机擦除及较强几何增强的 ResNet34。它们均训练 1,000 step，编码器学习率为 3e-4，解码器为 1e-3。

精定位围绕粗定位的 OOF 坐标或测试集五折平均坐标截取局部窗口。训练时围绕真实标注施加宽范围随机偏移。最终精定位等权平均 4 个配置：640 像素窗口的两个种子、896 像素窗口、640 窗口且输入为 384 的配置，每折固定训练 600 step。模型在最终粗定位集成中心上重新推理，使用原图 15 像素宽度的高斯先验、翻转 TTA 和一次精定位。

视野等比裁剪、图像金字塔和仿射逆变换共同保证输出回到原图像素坐标。

### 独立融合分量

- **Dropout200**：随机初始化的热图 U-Net，输入 256，通道宽度 24，Dropout2d 0.1，AdamW 学习率 3e-4，训练 200 epoch，平均最后 10 轮权重。推理温度为 0.5，不使用翻转 TTA。
- **两套分组划分**：第一、第二套 Dropout200 使用不同的固定分组五折，分别由种子 2026、7891 的划分冻结而来；它们不是同一划分下的两个模型种子。
- **独立 ResNet34**：TorchVision `IMAGENET1K_V1` 初始化，冻结 BatchNorm 统计量，输入 256，sigma 8，编码器学习率 1e-4、解码器 1e-3，训练 100 epoch，平均最后 10 轮权重，温度 0.5，无翻转 TTA。

两套划分都把像素重复的训练图放在同一折；80 张训练图对应 79 个不同像素组。独立分量与粗到细网络保留各自的预处理、坐标约定及训练实现。

## 最终融合公式

设 C 为粗到细级联坐标，R 为独立 ResNet34 坐标，D1、D2 为两套划分的 Dropout200 坐标：

```text
四分量融合 = 0.35 × C + 0.35 × R + 0.15 × D1 + 0.15 × D2
70/30 融合 = 0.70 × C + 0.30 × D1
```

融合使用 float64 坐标算术。程序检查所有分量的数据指纹、图像顺序、分组划分、有限值和原图坐标边界，按官方模板的 40 行顺序输出。

四分量融合与 70/30 融合的原实验 OOF RMSE 分别为 **14.426955** 和 **14.598080**。各分量对每张训练图的预测均来自未训练过该图及其重复图像组的折模型；四分量方案混合了两套划分。模型与融合配置经过反复 OOF 比较，这些分数用于描述选定方案，不作为独立测试分数。

## 有效性消融、超参数与模型选择

```bash
# 粗到细：实际完成的 25 组五折实验
python train.py --data data --out outputs --group experiments

# 从零基线：热图、坐标回归、增强、学习率、高斯宽度、分辨率等对照
python train_components.py --data data --out outputs --group baseline

# 固定预算 × Dropout 的对照
python train_components.py --data data --out outputs --group duration

# ResNet18 初始化/学习率与 ResNet34 的独立模型比较
python train_components.py --data data --out outputs --group transfer
```

| 类别 | 覆盖内容 |
| --- | --- |
| 有效性消融 | 随机初始化、几何增强、光度增强、视野裁剪、坐标 L1 损失、Dropout、CLAHE、热图与直接回归 |
| 超参数 | 输入分辨率、训练 step/epoch、学习率、高斯目标宽度、精定位窗口、偏移范围、先验宽度、解码温度 |
| 模型选择 | ResNet18/34、从零 U-Net、不同种子、冻结 BatchNorm、正则化配置、粗定位与粗到细级联 |

每个运行组保存配置、逐轮或逐 step 训练记录、逐折预测、OOF 和测试集预测及指标表。`--list` 查看可运行的实验名，`--run` 选择单个配置。不同因素同时变化的配置按完整训练方案比较。

## Kaggle Notebook

- `notebooks/train_cascade.ipynb`：正式粗到细级联。
- `notebooks/train_components.ipynb`：正式三个独立分量。
- `notebooks/experiments_cascade.ipynb`：粗到细消融、超参数和模型对照。
- `notebooks/experiments_components.ipynb`：从零及迁移学习的相关对照。

各 Notebook 只需添加官方竞赛数据。输出保存于 `/kaggle/working/task3/`；将正式训练的 `runs/`、`components/` 和 `cascade_predictions.npz` 汇入同一个本地 `outputs/` 后，运行上述两条融合命令。Notebook 由仓库根目录的 `python tools/build_task3_notebooks.py` 生成。

## 检查命令

```bash
python -m unittest discover -s tests -v
python tests/smoke_training.py
```
