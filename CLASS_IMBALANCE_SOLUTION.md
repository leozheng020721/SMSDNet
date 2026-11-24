# KappaSet云影检测 - 类别不平衡解决方案

## 🔍 问题分析

从训练日志看到的问题：
```
Epoch 5-10:
- Shadow IoU: 0.000
- Semi IoU: 0.000  
- Cloud IoU: 0.000 或 0.028
- Clear IoU: 0.389
- mIoU: 停滞在 0.097
```

**根本原因**：类别不平衡导致模型退化
- 背景(Clear)占33.5%，Cloud占33.5%，Semi占25.7%
- **Shadow仅占7.3%** ← 最重要但最稀少的类
- 使用均等权重的Loss导致模型忽略少数类

## ✅ 已实施的改进

### 1. 数据集类别分布分析

运行脚本获得真实分布：
```bash
python compute_class_weights.py
```

**实际分布**（1000样本采样）：
- Clear: 33.53%
- Cloud Shadow: **7.28%** ⚠️ 最稀少
- Semi-transparent: 25.69%
- Cloud: 33.50%

### 2. 新增损失函数配置

在`config.yaml`中添加了损失函数配置：

```yaml
loss:
  type: "weighted_ce"  # 使用加权CrossEntropy
  class_weights: [0.3, 5.0, 1.5, 1.0]  # Shadow权重最高(5.0)
```

**权重设计理念**：
- **Clear: 0.3** - 最常见类，降低权重
- **Shadow: 5.0** - 核心目标，权重最高（16.7倍于Clear！）
- **Semi: 1.5** - 中等稀少，中等权重
- **Cloud: 1.0** - 常见类，基准权重

### 3. 支持三种损失函数

修改`train.py`支持：

**A. 加权CrossEntropy (推荐) ⭐**
```yaml
loss:
  type: "weighted_ce"
  class_weights: [0.3, 5.0, 1.5, 1.0]
```

**B. Focal Loss**
```yaml
loss:
  type: "focal"
  focal_alpha: 0.25
  focal_gamma: 2.0
```

**C. 标准CrossEntropy (不推荐)**
```yaml
loss:
  type: "ce"
```

## 🚀 重新训练指南

### 方案1：从头开始（推荐）

使用新的损失权重从头训练：

```cmd
cd d:\segformer
conda activate kappa_swin
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python train.py
```

### 方案2：从现有checkpoint微调

保留已学到的特征，只调整损失函数：

```cmd
python train.py --resume checkpoints/KappaSet_SegFormer_MiT-B3_RGB_Baseline_best.pth
```

> ⚠️ 注意：resume时优化器状态会被保留，可能需要降低学习率

## 📊 预期改进效果

**Before（原始配置）**：
- Shadow IoU: 0.000
- mIoU: 0.097

**After（加权损失）预期**：
- Shadow IoU: 0.40 - 0.60  ← 大幅提升
- mIoU: 0.50 - 0.65
- 训练初期可能其他类IoU略降，但Shadow会快速上升

## 🔧 高级调优

如果Shadow IoU仍然不理想，可以尝试：

### 1. 进一步增加Shadow权重
```yaml
class_weights: [0.2, 8.0, 1.0, 0.8]  # Shadow权重提升到8.0
```

### 2. 使用Focal Loss
```yaml
loss:
  type: "focal"
  focal_alpha: 0.25
  focal_gamma: 2.0
```

### 3. 降低学习率（微调阶段）
```yaml
training:
  learning_rate: 2.0e-5  # 从6e-5降低到2e-5
```

## 📈 监控指标

训练时重点关注：
1. **Shadow IoU** - 核心指标
2. **mIoU** - 整体性能  
3. **Class-wise Loss** - 各类损失是否平衡

理想的训练曲线：
- Epoch 1-5: Shadow IoU快速上升（0 → 0.3）
- Epoch 5-20: 继续提升（0.3 → 0.5）
- Epoch 20+: 微调阶段（0.5 → 0.6+）

---

**现在就开始重新训练吧！** 🎯
