# 任务 2：灰度人脸表情分类

将 48×48 灰度人脸图像分为 Angry、Fear、Happy、Sad、Surprise、Neutral 六类，标签依次为 0–5。训练集 28,275 张，测试集 7,065 张，评价指标为准确率。

## 最终选用的两套方案

| 最终选用方案 | Kaggle 分数 | 融合入口 | 生成的提交文件 |
| --- | ---: | --- | --- |
| 双种子 ResNet32 与五个预训练家族融合 | **0.743** | `blend.py --recipe seeds` | `submissions/submission_seeds.csv` |
| 上述模型再加入从零训练的 VGG | **0.743** | `blend.py --recipe vgg` | `submissions/submission_vgg.csv` |

从零训练的 ResNet32 和 VGG 保持 48×48 输入。预训练模型使用 ImageNet-1k 初始化，将输入放大至 112×112；不使用额外的人脸图片。ResNet18、EfficientNet-B0、DeiT-Tiny、ViT-Small、ConvNeXt-Tiny 是五个预训练家族。

## 代码结构

| 文件 | 内容 |
| --- | --- |
| `train_resnet32.py` | 从零 ResNet32，分组五折，两个固定随机种子 |
| `train_pretrained.py` | 五个预训练家族、从零 VGG，以及最终配方的消融实验 |
| `train_holdout.py` | 最终融合使用的单个 ResNet32：固定分组留出划分、最佳 checkpoint、翻转 TTA |
| `ablation_initial.py` | 初始配方下的消融、学习率与分辨率实验，以及初始 ResNet18 对照 |
| `train_model_comparison.py` | 已完成的 ResNet50 迁移学习对照实验 |
| `blend.py` | 固定权重融合、五折完整性与图像/ID/划分校验 |
| `cli.py` | 训练程序共享的运行组与输入输出参数 |
| `notebooks/` | 可独立上传 Kaggle 的训练及实验 Notebook |
| `tests/` | 概率融合、分组校验、模型接口与合成数据流程检查 |

## 数据与环境

从 [2026task2 竞赛](https://www.kaggle.com/competitions/2026task2) 获取官方数据。六类目录可使用英文表情名；程序递归识别解压后的训练与测试目录。

```text
input/
└── 2026task2/
    ├── train/
    │   ├── Angry/
    │   ├── Fear/
    │   ├── Happy/
    │   ├── Sad/
    │   ├── Surprise/
    │   └── Neutral/
    ├── test/
    └── sample_submission.csv   # 数据提供时保留；列名为 ID,Emotion
```

如果官方输入仅包含图片，程序按完整文件名生成 `ID,Emotion` 提交格式。所有图像均检查为 48×48；图片损坏、训练类别缺失、测试数量错误时停止运行。

使用 Python 3.10 或更新版本、匹配的 PyTorch/torchvision 和 CUDA GPU。安装依赖：

```bash
python -m pip install -r requirements.txt
```

下列命令均在本目录执行。`--input` 指向包含 `2026task2` 的上层目录；必要时使用 `--data-root` 指定解压后的竞赛目录。所有正式训练结果放在 `outputs/<运行组>/`。

## 训练正式模型

```bash
# 单模型：固定分组留出划分，供两套最终方案共同使用
python train_holdout.py --input input --out outputs

# ResNet32：五折、两个模型种子
python train_resnet32.py --run cv_resnet32 --input input --out outputs
python train_resnet32.py --run cv_resnet32_s2027 --input input --out outputs

# 五个预训练家族，其中 ResNet18 有两个种子
python train_pretrained.py --run cv_resnet18 --input input --out outputs
python train_pretrained.py --run cv_resnet18_s2027 --input input --out outputs
python train_pretrained.py --run cv_effnet_b0 --input input --out outputs
python train_pretrained.py --run cv_deit_tiny --input input --out outputs
python train_pretrained.py --run cv_vit_small --input input --out outputs
python train_pretrained.py --run cv_convnext_tiny --input input --out outputs

# 第二套最终方案额外需要 VGG
python train_pretrained.py --run cv_vgg_scratch48 --input input --out outputs
```

每个五折运行组保存 `plan.json`、`train_index.csv`、`status.json`、OOF/测试概率、逐轮训练记录和 checkpoint。中断后在相同输出目录重新运行即可继续；跨 Kaggle 会话时，使用 `--resume` 指定保存的运行组目录，或把之前的完整 Output 加到 Input。用于恢复的完整 Output 包含权重，结果 ZIP 仅用于分析。

单模型 ResNet32 使用 20 折分层分组划分构造约 70% 训练、15% 验证、15% 留出集；只在训练部分更新参数和计算归一化统计，在验证部分选择 checkpoint，不使用留出集调参。它按验证准确率选择 60 轮中的最佳权重，并使用水平翻转 TTA。

五折模型使用最后一轮 EMA 权重。所有五折运行组共用 `StratifiedGroupKFold(5, shuffle=True, random_state=2026)`，每个种子的测试概率为五个折模型的平均。相同或互为水平镜像的图片归为一组，防止跨折重复。模型种子为 2026 和 2027，分组划分种子始终为 2026。低对比度图片保留在五折评估中，标签冲突样本保留原标签。

## 模型配置

| 模型 | 输入 | 轮数 | 学习率 | 预训练 | OOF 准确率 |
| --- | ---: | ---: | ---: | --- | ---: |
| ResNet32，种子 2026 | 48 | 60 | 1e-3 | 无 | 67.85% |
| ResNet32，种子 2027 | 48 | 60 | 1e-3 | 无 | 67.94% |
| ResNet18，种子 2026 | 112 | 100 | 1e-3 | ImageNet-1k | 68.56% |
| ResNet18，种子 2027 | 112 | 40 | 1e-3 | ImageNet-1k | 67.86% |
| EfficientNet-B0 | 112 | 40 | 1e-3 | ImageNet-1k | 67.69% |
| DeiT-Tiny | 112 | 30 | 3e-4 | ImageNet-1k | 68.44% |
| ViT-Small | 112 | 40 | 3e-4 | ImageNet-1k | 69.09% |
| ConvNeXt-Tiny | 112 | 30 | 2e-4 | ImageNet-1k | 69.12% |
| VGG | 48 | 80 | 1e-3 | 无 | 69.41% |

表中为课程实验记录的五折结果，推理使用 EMA 和 10 视图 TTA。10 视图由中心及上下左右各平移 2 像素，再分别水平翻转构成。

训练增强、Mixup、标签平滑、权重衰减、预热和 EMA 参数分别保存在各运行组的配方中。ResNet32 与预训练家族使用各自的增强和优化器实现；ResNet32 采用 Mixup 0.2，最终预训练配方关闭 Mixup。

## 最终融合

同一家族先平均种子和折模型；从零训练的部分占总权重的一半，五个预训练家族平分另一半。

| 概率成员 | 双种子融合 `seeds` | 加入 VGG 的融合 `vgg` |
| --- | ---: | ---: |
| 单模型 ResNet32（固定留出划分） | 0.25 | 0.125 |
| 五折 ResNet32，种子 2026 | 0.125 | 0.0625 |
| 五折 ResNet32，种子 2027 | 0.125 | 0.0625 |
| 五折 VGG | — | 0.25 |
| 五折 ResNet18，种子 2026 | 0.05 | 0.05 |
| 五折 ResNet18，种子 2027 | 0.05 | 0.05 |
| 五折 EfficientNet-B0 | 0.10 | 0.10 |
| 五折 DeiT-Tiny | 0.10 | 0.10 |
| 五折 ViT-Small | 0.10 | 0.10 |
| 五折 ConvNeXt-Tiny | 0.10 | 0.10 |

完成全部所需运行组后：

```bash
python blend.py --results outputs --recipe seeds --out submissions
python blend.py --results outputs --recipe vgg --out submissions
```

融合程序拒绝缺折、图像顺序不同、划分不同、ID 错配或无效概率。`submission_seeds.csv` 与 `submission_vgg.csv` 分别对应两套最终采用的方法，原实验的公开榜均为 **0.743**。

不包含单模型留出预测时，两套五折融合规则的全量 OOF 准确率分别为 **72.10%** 和 **72.31%**。单模型曾在自身验证集上选择 checkpoint，因此完整提交不具有覆盖全部训练图的独立 OOF；两种口径在代码和结果字段中分别表示。

## 有效性消融、超参数与模型选择

```bash
# 初始配方：第 0 折，每组 10 轮
python ablation_initial.py --run ablation --input input --out outputs/initial

# 初始 ResNet18 对照：五折、25 轮
python ablation_initial.py --run cv_resnet18 --input input --out outputs/initial

# 最终配方：第 0 折，每组 30 轮
python train_pretrained.py --run final_ablation --input input --out outputs

# 已完成的 ResNet50 对照：五折、20 轮，分类头学习率为主干的 10 倍
python train_model_comparison.py --input input --out outputs
```

| 实验类别 | 覆盖内容 |
| --- | --- |
| 有效性消融 | 去掉预训练、只训分类头、去掉增强、Mixup、Random Erasing、标签平滑、EMA、TTA |
| 超参数选择 | 学习率、输入分辨率、权重衰减、优化器、随机种子；初始与最终配方分别比较 |
| 模型选择 | 上表各模型与 ResNet50、初始 ResNet18；五折结果、训练曲线、TTA/EMA 和融合收益 |

初始配方排除了 11 张近纯色图，最终五折配方保留全部训练图，二者的划分不同。各配方内使用同一验证折进行组件比较；初始与最终配方的绝对分数不作为同一验证集上的逐图配对比较。

## Kaggle Notebook

`notebooks/train_pretrained.ipynb` 在最后一格修改 `RUN`，即可运行各预训练家族或 VGG；`train_resnet32.ipynb` 可分别选择两个种子。`train_single_resnet32.ipynb` 对应单模型留出方案。另有初始消融、最终消融和模型对比 Notebook。

只需添加官方 `2026task2` 数据；预训练模型开启 GPU 和 Internet，从零模型开启 GPU 即可。各 Notebook 输出 `task2/<运行组>/`。收集到本地时保留运行组目录名，放入统一 `outputs/` 目录，再运行融合命令。

Notebook 由仓库根目录的 `python tools/build_task2_notebooks.py` 生成。

## 检查命令

```bash
python -m unittest discover -s tests -v
python tests/smoke_training.py
```
