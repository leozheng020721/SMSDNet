# KappaSet云影检测基线系统

基于**SegFormer (MiT-B3)** 架构的高精度RGB云影检测系统，针对**NVIDIA RTX 5070 Ti (Blackwell架构)** 深度优化。

---

## 🚀 项目概述

本项目实现了针对KappaSet数据集的云影检测基线系统，在仅使用RGB波段（无近红外/短波红外）的限制下，通过Transformer的全局注意力机制实现高精度的云影分割。

### 核心特性

- **🏗️ SOTA架构**: 采用SegFormer-B3，利用Self-Attention解决云影检测的空间依赖问题
- **⚡ 硬件优化**: 针对RTX 5070 Ti Blackwell架构（12GB GDDR7）定制显存管理策略
- **🔬 混合精度**: FP16自动混合精度训练，充分利用第5代Tensor Cores
- **📊 完整流水线**: 从数据加载、训练到推理的端到端实现

### 检测类别

| 类别ID | 类别名称 | 描述 | 颜色标识 |
|--------|----------|------|----------|
| 0 | Clear | 晴空背景 | 绿色 |
| 1 | Cloud Shadow | 云的投影阴影 | 紫色 |
| 2 | Semi-transparent | 半透明云/薄卷云 | 橙色 |
| 3 | Cloud | 厚云 | 白色 |

---

## 📋 系统要求

### 硬件要求

- **GPU**: NVIDIA RTX 5070 Ti (或其他支持CUDA 12.8的Blackwell/Ada Lovelace架构GPU)
- **显存**: 至少12GB VRAM
- **CPU**: 多核处理器 (推荐8核以上，用于数据加载)
- **内存**: 32GB RAM (推荐)
- **存储**: 至少100GB可用空间 (用于存放KappaSet数据集)

### 软件要求

- **操作系统**: Windows 10/11 或 Linux
- **Python**: 3.11+
- **CUDA**: 12.8
- **驱动**: NVIDIA驱动版本 ≥ 570.x (支持Blackwell架构)

---

## 🛠️ 环境配置

### 1. 创建虚拟环境

在**Conda Prompt**（或PowerShell）中执行以下命令：

```bash
# 创建虚拟环境
conda create -n cloud_detect_env python=3.11 -y

# 激活环境
conda activate cloud_detect_env
```

### 2. 配置显存管理

为避免显存碎片化导致的OOM错误，需设置环境变量：

**Windows (CMD/Conda Prompt):**
```bash
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**Windows (PowerShell):**
```powershell
$Env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
```

**Linux/macOS:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

> **💡 提示**: 建议将此环境变量添加到系统环境变量或conda环境激活脚本中。

### 3. 安装PyTorch Nightly (CUDA 12.8)

由于RTX 5070 Ti属于最新硬件，必须使用PyTorch Nightly版本：

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 4. 验证安装

```bash
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

预期输出示例：
```
Torch: 2.10.0.dev20251104+cu128
CUDA: True
Device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

### 5. 安装依赖库

```bash
# 计算机视觉与深度学习库
pip install opencv-python-headless albumentations matplotlib scikit-learn tqdm pyyaml

# 语义分割模型库
pip install segmentation-models-pytorch

# Timm (SegFormer依赖的backbone库)
pip install timm

# ONNX Runtime (可选，用于模型部署)
pip install onnxruntime-gpu
```

---

## 📦 数据集准备

### 下载KappaSet

KappaSet数据集可从以下渠道获取：
- 官方仓库: [链接待补充]
- Kaggle竞赛: [链接待补充]

### 数据集结构

解压后的数据集应遵循以下目录结构：

```
D:/datasets/KappaSet/
├── train_images/
│   ├── tile_0001.tif (或 .png)
│   ├── tile_0002.tif
│   └── ...
├── train_masks/
│   ├── tile_0001.tif
│   ├── tile_0002.tif
│   └── ...
├── val_images/
│   └── ...
├── val_masks/
│   └── ...
├── test_images/
│   └── ...
└── test_masks/
    └── ...
```

> **⚠️ 注意**: 如果您的数据集路径不同，请修改 `config.yaml` 中的 `dataset.root_dir` 参数。

### 数据格式说明

- **图像格式**: 16-bit GeoTIFF 或 PNG
- **掩膜格式**: 8-bit 单通道图像
- **分辨率**: 512x512 (或根据实际数据集调整 `config.yaml` 中的 `input_size`)

---

## 🎯 使用指南

### 配置文件

所有超参数集中在 `config.yaml` 中管理。关键配置项：

```yaml
dataset:
  root_dir: "D:/datasets/KappaSet"  # 数据集路径
  input_size: 512                   # 输入图像大小

training:
  batch_size: 8                     # 批量大小（12GB显存可用8，若OOM降至4）
  epochs: 50                        # 训练轮数
  learning_rate: 6.0e-5             # 学习率
  mixed_precision: true             # 混合精度训练
```

### 训练模型

```bash
python train.py
```

训练过程中，模型会自动保存到 `checkpoints/` 目录，最佳模型文件名为：
```
checkpoints/KappaSet_SegFormer_MiT-B3_RGB_Baseline_best.pth
```

### 推理预测

#### 单张图像推理

```bash
python inference.py \
    --mode single \
    --checkpoint checkpoints/KappaSet_SegFormer_MiT-B3_RGB_Baseline_best.pth \
    --image path/to/your/image.tif \
    --output predictions/
```

#### 批量推理（整个验证集）

```bash
python inference.py \
    --mode dataset \
    --checkpoint checkpoints/KappaSet_SegFormer_MiT-B3_RGB_Baseline_best.pth \
    --split val \
    --output predictions/
```

---

## 📊 性能优化指南

### 显存不足 (OOM) 解决方案

如果出现 `CUDA out of memory` 错误：

1. **降低Batch Size**: 修改 `config.yaml` 中的 `batch_size` 从 8 降至 4
2. **使用更小的编码器**: 将 `model.encoder` 从 `mit_b3` 改为 `mit_b2`
3. **确认环境变量**: 检查 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` 是否生效

### 训练速度优化

1. **调整DataLoader Workers**: 根据CPU核心数调整 `num_workers` (推荐设为CPU核心数的一半)
2. **启用cudnn benchmark**: 在 `config.yaml` 中确保 `cudnn_benchmark: true`
3. **减少数据增强**: 注释 `dataset.py` 中部分耗时的增强操作（如GaussianBlur）

### 预期性能指标

在RTX 5070 Ti上的训练速度（Batch Size=8）：
- **训练速度**: ~2-3秒/迭代
- **显存占用**: ~10-11GB
- **收敛时间**: 约50轮 (2-3小时)

预期精度（验证集）：
- **mIoU**: 75-80%
- **Cloud IoU**: 85-90%
- **Shadow IoU**: 60-70% (RGB限制下的瓶颈)

---

## 🔧 故障排查

### PyTorch无法识别GPU

**症状**: `torch.cuda.is_available()` 返回 `False`

**解决方案**:
1. 检查NVIDIA驱动版本: `nvidia-smi`
2. 确认CUDA版本匹配: `nvcc --version`
3. 重新安装PyTorch Nightly

### 数据加载错误

**症状**: `FileNotFoundError: 在 xxx 未找到图像文件`

**解决方案**:
1. 检查 `config.yaml` 中的 `dataset.root_dir` 路径
2. 确认数据集目录结构是否符合要求
3. 检查图像文件扩展名（`.tif` 或 `.png`），必要时修改 `dataset.py` 中的 `glob` 模式

### 标签值异常

**症状**: 训练时出现意外的类别ID

**解决方案**:
检查数据集的标签编码是否符合以下映射：
- 0: Undefined → 映射为255 (Ignore)
- 1: Clear → 映射为0
- 2: Cloud Shadow → 映射为1
- 3: Semi-transparent → 映射为2
- 4: Cloud → 映射为3
- 5: Missing → 映射为255 (Ignore)

---

## 📂 项目结构

```
segformer/
├── config.yaml           # 配置文件
├── dataset.py            # 数据加载模块
├── model.py              # 模型定义
├── train.py              # 训练脚本
├── inference.py          # 推理脚本
├── utils.py              # 工具函数
├── README.md             # 本文档
├── checkpoints/          # 模型权重保存目录 (自动创建)
└── predictions/          # 推理结果保存目录 (自动创建)
```

---

## 📖 技术原理

### 为什么选择SegFormer？

1. **全局感受野**: Self-Attention机制可在第一层建立全局像素关联，解决云与影距离较远的检测难题
2. **多尺度特征**: Mix Transformer编码器生成1/4, 1/8, 1/16, 1/32多尺度特征，适合遥感图像的精细分割
3. **轻量级解码器**: 全MLP解码器将计算资源集中在特征提取上，提高效率

### RGB数据的局限性突破

在缺乏NIR/SWIR波段的情况下：
- **水体 vs 阴影混淆**: 在RGB波段几乎无法区分
- **解决方案**: 利用Transformer学习"云-影"的空间位置关系（几何依赖）
- **进一步优化**: 可考虑引入多时相数据或DEM（数字高程模型）辅助

---

## 🤝 贡献与反馈

如有问题或改进建议，欢迎通过以下方式反馈：
- 提交Issue
- Pull Request
- 邮件联系: [您的邮箱]

---

## 📄 许可证

本项目仅供学术研究使用。数据集版权归KappaSet原作者所有。

---

## 🙏 致谢

- **SegFormer论文**: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
- **Segmentation Models PyTorch**: [qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- **KappaSet数据集**: [相关论文/仓库链接]

---

**最后更新**: 2025-11-28  
**作者**: KappaSet云影检测项目组
