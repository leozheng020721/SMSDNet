# SEO 原尺度裁块与辅助监督

新配置为 `config_seo_crop_aux.yaml`，实验名为 `seo_segformer_native512_aux`。旧配置 `config_seo.yaml` 保留原来的整图缩放流程。

## 图像处理

训练从原图随机裁出 `512 x 512`，保留原始像素尺度，再进行翻转、90 度旋转和亮度对比度增强。小于裁块的图像先补边，补出的标签为 `255`，所有损失忽略这些位置。每个 epoch 每张训练图随机采样一个块，不代表该轮已经遍历该图全部区域。

验证和推理保留原图尺寸，以 `512 x 512` 窗口、`384` 步长覆盖整张图。重叠位置的 logits 取平均后再分类，边缘补齐后去除补边。只把小批量图块送入设备，整图及拼接结果保留在 CPU。验证固定使用原图标签计算 BER、IoU、F1 和 Kappa。

配置入口：`dataset.spatial_mode: crop`、`dataset.input_size: 512`、`dataset.tile_stride: 384`。训练使用 `training.batch_size: 6`，验证与推理的图块批次仍为 `dataset.tile_batch_size: 4`；原尺寸图像仍逐张加载。这些设置会随新 checkpoint 保存并在推理时恢复。

SEO 使用标准 RGB 图像的 `/255` 和 mean/std 归一化，阴影标签由原始掩膜的黑色区域映射为 `1`。`normalization.div_value` 是历史反射率参数，不用于 SEO 图像。

## 跨尺度十字交叉注意力

`config_seo_cca.yaml` 在当前颜色分支、空间门控和三路辅助监督之上，增加跨尺度十字交叉注意力。配置开关为：

```yaml
model:
  multicolor:
    cross_scale_attention:
      enabled: true
      attention_channels: 64
```

四个尺度先完成 RGB/HSV/Lab 门控融合。融合后的第三阶段特征 F3（输入的 1/16）生成 Q，第四阶段 F4（1/32）先双线性对齐 F3，再生成 K/V。每个位置只查询同行、同列的信息，二者联合做 softmax，中心位置只计一次。增强后的 F3 和其他三个尺度一起进入原解码器。这是基于 [CCNet](https://arxiv.org/abs/1811.11721) 的跨尺度扩展，不是三个颜色分支各自加注意力，也不是完整二维全局注意力。

新增残差为 `F3_out = F3 + gamma * delta_F3`，可学习的 `gamma` 初始为 0，迁移旧权重时初始输出与旧模型一致。第一步通常先更新 gamma，随后注意力投影开始获得梯度。RGB/HSV/Lab 的辅助头仍监督各自融合前的特征；普通推理也会执行新增注意力。此版本尚未加入颜色分歧 D 或校准门 G。

新配置沿用当前 batch size 6、MixUp、主损失和辅助损失，学习率为 1e-5 到 1e-6 的无重启余弦衰减，适合从已有辅助分支权重开始新实验。下列命令同时迁移匹配的主模型、颜色分支、空间门控和三路辅助头，注意力模块重新初始化；不恢复旧优化器、调度器、epoch 或最佳 BER：

```powershell
$env:NO_ALBUMENTATIONS_UPDATE = "1"
Set-Location E:/MyPythonProject/segformer
& "C:/Users/111/.conda/envs/seg/python.exe" "E:/MyPythonProject/segformer/train.py" --config "E:/MyPythonProject/segformer/config_seo_cca.yaml" --init-from "E:/MyPythonProject/segformer/HSVLAB辅助分支/seo_segformer_native512_aux_cosine_mixup_best.pth" --init-auxiliary-heads
```

`--init-auxiliary-heads` 只与 `--init-from` 一起使用；不加这个开关时，旧的迁移规则仍保留辅助头重新初始化的行为。增加结构属于新实验，不能用旧权重直接 `--resume`。新实验中断后，使用同一新配置及新实验的 last checkpoint 普通 `--resume`。

新结果保存在 `E:/MyPythonProject/segformer/checkpoints/seo_segformer_native512_aux_cca_cosine_mixup_best.pth`、同前缀的 `_last.pth` 和 `_history.csv`，不覆盖已有辅助分支实验。训练完成后推理：

```powershell
& "C:/Users/111/.conda/envs/seg/python.exe" "E:/MyPythonProject/segformer/inference.py" --config "E:/MyPythonProject/segformer/config_seo_cca.yaml" --checkpoint "E:/MyPythonProject/segformer/checkpoints/seo_segformer_native512_aux_cca_cosine_mixup_best.pth" --split val --output "E:/MyPythonProject/segformer/predictions_cca"
```

旧配置默认不启用新增模块，旧模型仍按原结构推理。新权重保存完整注意力配置，推理会从 checkpoint 恢复结构。上述迁移实验可检查模块是否有潜在收益；严谨结构消融还应设置相同初始化来源、相同训练预算的无注意力对照。若从 ImageNet 开始重跑，应为有、无注意力的两组同时设定训练学习率和新的实验名，避免把从头训练与低学习率迁移直接比较。

### 注意力版本从头训练

`config_seo_cca_scratch.yaml` 用于不加载之前阴影检测 checkpoint 的新训练。RGB 编码器仍使用 ImageNet 预训练初始化；颜色分支、门控、辅助头、注意力和解码器重新初始化，优化器、epoch 和最佳指标重新开始。这里的“从头”指从新的任务训练开始，不是把 RGB 编码器也改为完全随机初始化。

配置保留相同的 seed=42、20% 地块验证划分、batch size 6、MixUp 和辅助损失。初始学习率与此前从头训练配置一致，为 6e-5，通过无重启余弦退火在 50 轮内衰减至 1e-6。运行时不加 `--resume`、`--init-from` 或 `--init-auxiliary-heads`：

```powershell
$env:NO_ALBUMENTATIONS_UPDATE = "1"
Set-Location E:/MyPythonProject/segformer
& "C:/Users/111/.conda/envs/seg/python.exe" "E:/MyPythonProject/segformer/train.py" --config "E:/MyPythonProject/segformer/config_seo_cca_scratch.yaml"
```

结果使用独立前缀 `checkpoints/seo_segformer_native512_aux_cca_scratch_mixup`。迁移训练配置 `config_seo_cca.yaml` 保留用于单独实验；两者训练起点不同，不能把指标差异全部归因于注意力模块。

## 损失的来源与作用

官方 NVlabs SegFormer 的[基础模型配置](https://github.com/NVlabs/SegFormer/blob/master/local_configs/_base_/models/segformer.py)和 [B3 ADE20K 配置](https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B3/segformer.b3.512x512.ade.160k.py)使用 softmax 交叉熵 `CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0)`，这两处配置没有 `auxiliary_head`。

本项目原来的 `2 * Focal + Dice` 是当前阴影分割实现中的组合，不是直接照搬官方 SegFormer 的损失。新增辅助监督也属于本项目多颜色分支的改动。

本次保留主损失，只增加独立分支监督：

```text
L_main  = 2 * Focal(P_final, Y) + Dice(P_final, Y)
L_aux   = (CE(P_rgb, Y) + CE(P_hsv, Y) + CE(P_lab, Y)) / 3
L_total = L_main + 0.4 * L_aux
```

- `P_final` 是 RGB、HSV、Lab 特征融合后的最终分割 logits；Focal 强调难分类像素，Dice 优化预测区域与标签的重叠。
- `P_rgb`、`P_hsv`、`P_lab` 分别来自各分支融合前的四级特征。每个分支增加轻量预测头，将辅助 logits 上采样到标签尺寸，再与同一标签 `Y` 计算 softmax 交叉熵。
- 辅助损失让三条分支分别学习可用于阴影分类的特征，避免仅依赖最终融合输出的间接梯度；RGB 辅助头同样参与监督。
- 三个辅助 CE 先取平均，再乘 `loss.auxiliary_weight: 0.4`。`0.4` 是可调的初始设置，需要通过验证或消融决定，不能据此预先声称 BER 改善。

`model.multicolor.auxiliary_supervision: true` 控制是否构建辅助头，`aux_decoder_channels: 16` 控制辅助头宽度。训练通过 `forward_with_aux` 得到主输出和三路辅助输出；普通 `forward` 只计算主输出，推理不会执行辅助头，因此没有辅助头的前向计算开销。

## MixUp 训练

当前推荐的 `config_seo_cosine_mixup.yaml` 在训练时启用 MixUp：`training.mixup.enabled: true`、`alpha: 0.2`、`probability: 0.5`。每个训练批次以 `0.5` 的概率混合配对图像，使用同一个混合系数分别对两组标签的损失加权：

```text
x_mix      = lambda * x_A + (1 - lambda) * x_B
L_mix_main = lambda * L_main(output, Y_A) + (1 - lambda) * L_main(output, Y_B)
L_mix_aux  = lambda * L_aux(output, Y_A) + (1 - lambda) * L_aux(output, Y_B)
L_mix      = L_mix_main + 0.4 * L_mix_aux
```

模型对 `x_mix` 前向计算一次，主输出和 RGB、HSV、Lab 三路辅助输出都计算上述加权损失。标签保持原来的整数类别，不把两张 mask 直接平均后再阈值化；忽略标签 `255` 仍由每组标签对应的损失处理。验证和推理不混图。

`alpha=0.2` 和应用概率 `0.5` 是初始试验值，需要与相同 batch size 和学习率下的无 MixUp 对照比较，不能保证 BER 下降。本配置保持 `mixed_precision: false`，此次启用的是 MixUp。

## 学习率与当前模型续训

之前的 `CosineAnnealingWarmRestarts` 沿用旧工程设置，没有针对 SEO 完成消融验证，也不是原版 SegFormer 的学习率策略。官方 [B3 ADE20K 配置](https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B3/segformer.b3.512x512.ade.160k.py)使用按迭代更新的 Poly 衰减，`power=1.0`，并以 `1500` 次迭代进行线性 warmup。

`config_seo_crop_aux.yaml` 已改用无重启的 `CosineAnnealingLR`：从头训练的初始学习率仍为 `6e-5`，`T_max=50`，最低学习率为 `1e-6`。`6e-5` 与官方 B3 的 AdamW 基础学习率相同，但官方任务、数据、参数组及预热设置不同，不能据此认定它对 SEO 最优，也不能只凭这个数值断定当前 BER 的原因。本次续训降为 `1e-5`，不再周期性地把学习率拉回高值。

当前推荐使用 batch size `6`、低学习率、无重启余弦衰减和 MixUp 继续训练，配置为 `config_seo_cosine_mixup.yaml`。先在旧训练终端按 `Ctrl+C`，确认旧进程退出，再执行：

```powershell
conda activate seg
python train.py --config config_seo_cosine_mixup.yaml --resume checkpoints/seo_segformer_native512_aux_best.pth --reset-scheduler
```

文件修改不会热更新仍在运行的 Warm Restarts 进程。上述命令恢复完整模型（含三路辅助头）、优化器动量、epoch 和最佳 BER，再由 `--reset-scheduler` 显式将学习率设为 `1e-5`，建立无重启的余弦衰减，最低为 `1e-6`。它保留已学权重，不是从头训练；`--init-from` 默认重新初始化辅助头，配合 `--init-auxiliary-heads` 才会迁移匹配的辅助头。

总训练目标仍为 `50` 轮。载入当前已完成第 `10` 轮的 best checkpoint 后，从第 `11` 轮继续，用剩余 `40` 轮作为新的衰减长度。MixUp 新实验名为 `seo_segformer_native512_aux_cosine_mixup`，history、best 和 last 使用此独立前缀，旧实验文件不覆盖。

保留无 MixUp 的余弦对照，训练 batch size 同样为 `6`、学习率同样为 `1e-5` 到 `1e-6`，实验名为 `seo_segformer_native512_aux_cosine`：

```powershell
python train.py --config config_seo_cosine_finetune.yaml --resume checkpoints/seo_segformer_native512_aux_best.pth --reset-scheduler
```

后续在同一余弦计划下恢复中断训练，只需普通 `--resume`，不要重复添加 `--reset-scheduler`。默认 `--resume` 恢复 checkpoint 中的优化器学习率和调度器状态，单独修改配置中的学习率不会覆盖已经恢复的学习率。

## 启动新实验

如需完全从头训练，可从 ImageNet 预训练的 RGB 编码器开始，不加载旧实验的权重、优化器或 epoch。HSV/Lab 编码器、融合模块、主解码器和辅助头重新初始化。以下基础配置不启用 MixUp：

```powershell
conda activate seg
python train.py --config config_seo_crop_aux.yaml
```

`seg` 环境已检查可导入 PyTorch、segmentation-models-pytorch 和 Albumentations，支持 `mit_b3`；ImageNet 预训练缓存位于 `C:/Users/111/.cache/torch/hub/checkpoints/mit_b3.pth`。`rgb` 环境也已完成代码测试，可作为已有的备用环境。上述命令供用户启动；本次代码修改没有自动开始长时间训练。

从头训练的命令不加 `--resume` 或 `--init-from`。如果已有同名 history，应先为这个全新实验设置另一个 `experiment_name`；不要把旧 history 追加到从头训练的记录中。已有当前模型应使用上一节的续训命令。

以后若需要迁移其他兼容 checkpoint，可单独使用 `--init-from`：它迁移兼容的主模型和颜色分支权重，辅助头默认重新初始化；加 `--init-auxiliary-heads` 可迁移匹配的辅助头。两种方式的优化器、调度器及 epoch 均重新开始。若要严谨比较“裁块”和“辅助监督”各自的贡献，应固定数据划分和相同初始化，重新训练对应消融实验。

## 推理与指标比较

使用 MixUp 新实验产生的 checkpoint：

```powershell
python E:/MyPythonProject/segformer/inference.py --config E:/MyPythonProject/segformer/config_seo_cosine_mixup.yaml --checkpoint E:/MyPythonProject/segformer/HSVLAB辅助分支/seo_segformer_native512_aux_cosine_mixup_best.pth --mode dataset --split val --output E:/MyPythonProject/segformer/predictions_aux
```

单图模式同样执行原尺寸滑窗：

```powershell
python E:/MyPythonProject/segformer/inference.py --config E:/MyPythonProject/segformer/config_seo_cosine_mixup.yaml --checkpoint E:/MyPythonProject/segformer/HSVLAB辅助分支/seo_segformer_native512_aux_cosine_mixup_best.pth --mode single --image E:/Dataset/SEO/example.png --output E:/MyPythonProject/segformer/predictions_aux
```

将 `--image` 指向实际图像；SEO 数据集入口支持 `train` 和 `val`，没有独立 `test` 划分时会明确报错。ONNX 脚本 `aisd_inference.py` 也按 checkpoint 的空间模式执行滑窗，保持窗口归一化和拼接方式一致。

新指标在原图分辨率上统计，旧训练日志的 BER 则来自缩成 `512 x 512` 的标签，两者不能直接作为同一评价协议下的改进幅度。若要比较旧模型和新模型，应在相同验证图像、标签分辨率和推理协议下重新评价。滑窗覆盖整图增加了验证和推理耗时；裁块训练本身仍以小块控制显存。

## 查看验证名单和保存预测图片

先只导出名单，不启动模型或占用 GPU：

```powershell
python E:/MyPythonProject/segformer/inference.py --config E:/MyPythonProject/segformer/config_seo_cosine_mixup.yaml --checkpoint E:/MyPythonProject/segformer/HSVLAB辅助分支/seo_segformer_native512_aux_cosine_mixup_best.pth --split val --output E:/MyPythonProject/segformer/predictions_aux --export-only
```

名单位于指定输出目录中的 `val_manifest.csv`，包含地块名、原图路径、标签路径和计划输出路径。之前导出的名单仍在 `E:/MyPythonProject/segformer/seo_val_best/val_manifest.csv`。CSV 中列出的输出图片会在实际推理时生成。加 `--count 5` 只选择固定顺序的前五张，名单也仅包含这五张；预览请使用单独的输出目录，保留完整名单。

正式推理使用同一命令，去掉 `--export-only`。脚本从 SEO checkpoint 恢复训练时的随机种子和验证比例，使用当前数据目录重建划分；如果训练后目录增删过地块或配对图片，不能仅靠随机种子保证恢复历史名单。

结果按地块和原图文件名保存，例如 `E:/MyPythonProject/segformer/predictions_aux/JAX_100/JAX_100_19_rgb_vis.png`。每张图有三份结果：

- `_vis.png`：原图、真实标签和预测结果的三列对比图。
- `_shadow.png`：原尺寸黑白阴影图，白色为预测阴影，黑色为背景。
- `_pred.png`：原始类别编号，背景为 0、阴影为 1，直接查看时接近全黑。

`_shadow.png` 的黑白含义与 SEO 原始标签相反，不应套用读取原标签的 `mask < 127` 来重新评价预测；指标仍由拼接后的类别编号在全原图上计算。当前推理沿用这次训练的标签和指标口径，没有更改有效像素过滤规则。训练进程退出后再运行 GPU 推理，避免同时占用显存。

## 相同 SEO 划分的基础 SegFormer

`E:/MyPythonProject/segformer/model.py` 保留了项目原有的 `segmentation_models_pytorch.Segformer` 基础实现。`variant: baseline` 会直接使用该模型；编码器仍为 MiT-B3，输入为 RGB，没有本项目新增的颜色分支、门控或辅助预测头。这是 SMP 实现的架构基线，不是 NVIDIA 官方代码和训练配方的逐项复现。

新增配置 `E:/MyPythonProject/segformer/config_seo_baseline.yaml` 使用相同的数据目录、seed=42、20% 地块验证划分、原尺度随机 512 裁块、stride=384 的完整原图滑窗评估和归一化。数据目录内容保持不变时，训练/验证原图名单与当前改进模型相同。

该配置使用交叉熵，辅助损失权重为 0，关闭 MixUp；这与官方 B3 ADE20K 配置的交叉熵目标一致。学习率采用本项目的无重启余弦策略，初始 6e-5、最低 1e-6、50 轮；从头训练的初始学习率与后期微调使用的 1e-5 用途不同。官方配置的 Poly、warmup 和参数组并未在此完整复现。参考：<https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B3/segformer.b3.512x512.ade.160k.py>。

在旧训练/推理进程退出后运行：

```powershell
$env:NO_ALBUMENTATIONS_UPDATE = "1"
Set-Location E:/MyPythonProject/segformer
& C:/Users/111/.conda/envs/seg/python.exe E:/MyPythonProject/segformer/train.py --config E:/MyPythonProject/segformer/config_seo_baseline.yaml
```

该命令从 ImageNet 预训练编码器开始一个新实验，不加载多色模型 checkpoint。不要为此命令添加旧模型的 `--resume` 或 `--init-from`。基础模型使用独立实验名 `seo_segformer_rgb_baseline_native512_ce`，最佳权重保存为 `E:/MyPythonProject/segformer/checkpoints/seo_segformer_rgb_baseline_native512_ce_best.pth`，最新权重和 history 同前缀。

由于此基线同时关闭了辅助监督、Focal/Dice 和 MixUp，且当前改进模型经过多阶段续训，二者结果可以作为同一数据划分上的基线比较，不能将差值全部归因于新增结构。若要做严格结构消融，应让两个模型从相同来源的初始化出发，使用相同主损失、增强、学习率策略和训练预算重新训练。

## 多模型推理结果目录

每个模型直接在项目根目录下使用一个结果文件夹：

```text
E:/MyPythonProject/segformer/
  predictions_segformer/
    val_manifest.csv
    JAX_101/
      JAX_101_13_rgb_vis.png
      JAX_101_13_rgb_shadow.png
      JAX_101_13_rgb_pred.png
  predictions_aux/
    val_manifest.csv
    JAX_101/
```

`predictions_segformer` 保存原版结果，`predictions_aux` 保存辅助分支版结果，内部保留同样的地块和原图名称。后续模型沿用 `predictions_模型名`，同一模型的新实验可加 `_v2`。前五张预览另用 `predictions_aux_preview` 等目录。划分种子、验证比例和模型设置保留在配置及日志中，无需逐层建立文件夹。

原版基础模型推理：

```powershell
$env:NO_ALBUMENTATIONS_UPDATE = "1"
& "C:/Users/111/.conda/envs/seg/python.exe" "E:/MyPythonProject/segformer/inference.py" --config "E:/MyPythonProject/segformer/config_seo_baseline.yaml" --checkpoint "E:/MyPythonProject/segformer/checkpoints/seo_segformer_rgb_baseline_native512_ce_best.pth" --split val --output "E:/MyPythonProject/segformer/predictions_segformer"
```

辅助分支模型推理：

```powershell
$env:NO_ALBUMENTATIONS_UPDATE = "1"
& "C:/Users/111/.conda/envs/seg/python.exe" "E:/MyPythonProject/segformer/inference.py" --config "E:/MyPythonProject/segformer/config_seo_cosine_mixup.yaml" --checkpoint "E:/MyPythonProject/segformer/HSVLAB辅助分支/seo_segformer_native512_aux_cosine_mixup_best.pth" --split val --output "E:/MyPythonProject/segformer/predictions_aux"
```

模型权重继续保存在各自现有路径，结果目录只收集名单和预测图片。比较时核对两份 `val_manifest.csv` 的原图与标签名单；目录名称本身不决定划分，也不自动保证其他结构采用相同预处理与评估协议。
