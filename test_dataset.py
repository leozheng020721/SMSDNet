"""
测试NetCDF文件读取的robustness
"""
import yaml
from dataset import KappaSetDataset, get_transforms

# 加载配置
with open("config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("Testing dataset with updated error handling...")
print(f"num_workers: {config['num_workers']}")

# 创建验证集
val_ds = KappaSetDataset(
    config['dataset']['root_dir'], 
    split='val', 
    config=config,
    transform=get_transforms(config, 'val')
)

print(f"\nValidation samples: {len(val_ds)}")

# 测试读取前10个样本
print("\nTesting first 10 samples...")
for i in range(min(10, len(val_ds))):
    try:
        img, mask = val_ds[i]
        print(f"  Sample {i}: OK (shape: {img.shape}, mask unique: {mask.unique().tolist()})")
    except Exception as e:
        print(f"  Sample {i}: FAILED - {str(e)}")

# 测试之前出错的文件索引附近
print("\nTesting samples around potential problem area...")
test_indices = [100, 200, 300, 400, 500]
for i in test_indices:
    if i < len(val_ds):
        try:
            img, mask = val_ds[i]
            print(f"  Sample {i}: OK")
        except Exception as e:
            print(f"  Sample {i}: FAILED - {str(e)}")

print("\n✓ Dataset robustness test complete!")
