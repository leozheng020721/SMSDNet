"""
计算KappaSet数据集的类别分布，用于设置损失函数的类别权重
"""
import yaml
import torch
import numpy as np
from tqdm import tqdm
from dataset import KappaSetDataset, get_transforms

# 加载配置
with open("config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("=== Computing Class Distribution ===\n")

# 加载训练集
train_ds = KappaSetDataset(
    config['dataset']['root_dir'], 
    split='train', 
    config=config,
    transform=None  # 不做增强，只统计原始数据
)

print(f"Total training samples: {len(train_ds)}\n")

# 统计各类像素数
class_counts = np.zeros(config['dataset']['num_classes'], dtype=np.int64)
total_pixels = 0
valid_pixels = 0

print("Analyzing samples...")
# 采样分析（全量分析太慢，采样1000个样本）
sample_size = min(1000, len(train_ds))
for i in tqdm(range(sample_size), desc="Processing"):
    try:
        _, mask = train_ds[i]
        
        # 过滤ignore像素
        valid_mask = (mask != 255)
        valid_pixels += valid_mask.sum().item()
        total_pixels += mask.numel()
        
        # 统计各类
        for cls in range(config['dataset']['num_classes']):
            class_counts[cls] += (mask == cls).sum().item()
    except:
        continue

print("\n" + "="*60)
print("Class Distribution Analysis")
print("="*60)

class_names = ['Clear', 'Cloud Shadow', 'Semi-transparent', 'Cloud']
total_valid = class_counts.sum()

for i, name in enumerate(class_names):
    count = class_counts[i]
    percentage = (count / total_valid * 100) if total_valid > 0 else 0
    print(f"{name:20s}: {count:12,} pixels ({percentage:6.2f}%)")

print(f"\nTotal valid pixels: {total_valid:,}")
print(f"Ignored pixels: {total_pixels - valid_pixels:,} ({((total_pixels - valid_pixels) / total_pixels * 100):.2f}%)")

# 计算类别权重（逆频率）
print("\n" + "="*60)
print("Recommended Class Weights")
print("="*60)

# 方法1: 逆频率权重（标准化）
frequencies = class_counts / total_valid
inv_freq_weights = 1.0 / (frequencies + 1e-6)
inv_freq_weights = inv_freq_weights / inv_freq_weights.sum() * len(class_names)

print("\n1. Inverse Frequency Weights (balanced):")
for i, name in enumerate(class_names):
    print(f"  {name:20s}: {inv_freq_weights[i]:.4f}")

# 方法2: 针对Shadow优化的权重
shadow_weights = inv_freq_weights.copy()
shadow_weights[1] *= 3.0  # Shadow权重额外放大3倍
shadow_weights = shadow_weights / shadow_weights.sum() * len(class_names)

print("\n2. Shadow-Focused Weights (3x multiplier on Shadow):")
for i, name in enumerate(class_names):
    print(f"  {name:20s}: {shadow_weights[i]:.4f}")

# 方法3: 极端强调Shadow
extreme_weights = np.array([0.5, 5.0, 1.0, 1.0])  # 手动设置
extreme_weights = extreme_weights / extreme_weights.sum() * len(class_names)

print("\n3. Extreme Shadow Emphasis (manual tuning):")
for i, name in enumerate(class_names):
    print(f"  {name:20s}: {extreme_weights[i]:.4f}")

# 保存推荐权重到文件
print("\n" + "="*60)
print("Saving weights to class_weights.txt...")

with open("class_weights.txt", "w") as f:
    f.write("# KappaSet Class Weights for Loss Function\n")
    f.write("# Copy one of these lines to config.yaml\n\n")
    
    f.write("# Method 1: Inverse Frequency (Balanced)\n")
    f.write(f"class_weights: {inv_freq_weights.tolist()}\n\n")
    
    f.write("# Method 2: Shadow-Focused (Recommended)\n")
    f.write(f"class_weights: {shadow_weights.tolist()}\n\n")
    
    f.write("# Method 3: Extreme Shadow Emphasis\n")
    f.write(f"class_weights: {extreme_weights.tolist()}\n")

print("✓ Weights saved to class_weights.txt")
print("\nRecommendation: Use Method 2 (Shadow-Focused) for best results!")
