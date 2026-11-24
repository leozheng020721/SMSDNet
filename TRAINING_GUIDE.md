# 训练与断点续训指南

## 当前推荐：MixUp 续训

```powershell
conda activate seg
python train.py --config config_seo_cosine_mixup.yaml --resume checkpoints/seo_segformer_native512_aux_best.pth --reset-scheduler
```

先在仍运行旧训练的终端按 `Ctrl+C`，确认进程退出，再执行上述命令。修改 Python 或 YAML 文件不会热更新已启动的训练进程。

此命令保留当前 best 的完整模型（含三路辅助头）、优化器动量、epoch 和最佳 BER。`--reset-scheduler` 显式将学习率改为 `1e-5`，建立无重启的 `CosineAnnealingLR`，最低为 `1e-6`。总目标仍为 `50` 轮；当前 best 已完成 `10` 轮，因此从第 `11` 轮继续，用剩余 `40` 轮作为新的衰减长度。

训练 batch size 为 `6`，验证与推理的滑窗图块批次仍为 `4`。配置启用 MixUp，`alpha=0.2`、应用概率 `0.5`；`mixed_precision` 保持 `false`。新实验以前缀 `seo_segformer_native512_aux_cosine_mixup` 保存独立 history、best、last，旧实验文件不覆盖。

无 MixUp 的余弦对照使用同样的 batch size `6` 和学习率计划，独立前缀为 `seo_segformer_native512_aux_cosine`：

```powershell
python train.py --config config_seo_cosine_finetune.yaml --resume checkpoints/seo_segformer_native512_aux_best.pth --reset-scheduler
```

## MixUp 如何监督

```text
x_mix = lambda * x_A + (1 - lambda) * x_B
L_mix = lambda * L(output, Y_A) + (1 - lambda) * L(output, Y_B)
L     = L_main + 0.4 * L_aux
```

模型对混合图像前向计算一次，主输出和 RGB、HSV、Lab 三路辅助输出都分别对两组整数标签计算损失，再按 `lambda` 加权。mask 不直接平均后阈值化，忽略标签 `255` 分别由对应标签的损失处理。验证与推理不使用 MixUp。`alpha=0.2` 和概率 `0.5` 是初始试验值，不保证 BER 下降，需与无 MixUp 对照评价。

## 按原计划断点续训

若只是恢复中断训练，使用同一学习率计划和配置，通常从最新的 `last` checkpoint 继续：

```powershell
python train.py --config config_seo_cosine_mixup.yaml --resume checkpoints/seo_segformer_native512_aux_cosine_mixup_last.pth
```

默认 `--resume` 恢复完整模型（含辅助头）、优化器状态、学习率、调度器状态、训练进度和最佳 BER。它不会用配置里的学习率覆盖 checkpoint 中已经恢复的学习率。选择 `best` 则从取得最低 BER 的那一轮继续，可能早于 `last` 的进度。

`--reset-scheduler` 需要与 `--resume` 一起使用，仅在主动改变学习率计划时添加。随后再次恢复同一新实验时使用普通 `--resume`，不要重复重置调度器。

若只想迁移兼容权重并让优化器和训练轮次重新开始，才使用 `--init-from`；该入口会重新初始化辅助头，不适合本次保留当前辅助头的续训。

## 从头训练

```powershell
conda activate seg
python train.py --config config_seo_crop_aux.yaml
```

此基础配置不启用 MixUp，从 ImageNet 预训练编码器开始新实验，训练 batch size `6`。初始学习率为 `6e-5`，通过无重启余弦退火在 `50` 轮衰减至 `1e-6`。已有同名 history 时，应为全新实验设置新的 `experiment_name`。KappaSet 仍可通过 `--config config.yaml` 选择原数据配置。

## 策略来源与保存规则

之前的 Warm Restarts 沿用旧工程，没有 SEO 消融依据。官方 [NVlabs SegFormer B3 ADE20K 配置](https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B3/segformer.b3.512x512.ade.160k.py)使用 Poly 衰减和线性 warmup：`power=1.0`、`warmup_iters=1500`，按迭代更新。不能把旧 Warm Restarts 或本次余弦退火称为官方原版策略。

`6e-5` 与官方 B3 的 AdamW 基础学习率相同，但任务、数据、参数组和预热设置不同，不能据此认定它是 SEO 的最优学习率，也不能仅凭该数值确定 BER 停滞的原因。本次从已学权重继续训练，采用 `1e-5` 到 `1e-6` 的无重启余弦计划。

训练在 `checkpoints/` 下按实验名保存文件：`best.pth` 仅在有效 BER 刷新最低值时更新，`last.pth` 每轮保存最新状态，`history.csv` 记录每轮实际使用的学习率、损失和指标。原尺寸评价与旧版缩放到 `512 x 512` 的评价协议不同，不能直接用两份日志计算改进幅度。
