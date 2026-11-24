import cv2
import numpy as np
import glob
import os

results_dir = "inference_results"
pred_files = glob.glob(os.path.join(results_dir, "*_pred.png"))

print(f"Found {len(pred_files)} prediction files.")

class_counts = {}

for f in pred_files:
    # 读取Mask (灰度模式)
    mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"Error reading {f}")
        continue
        
    unique, counts = np.unique(mask, return_counts=True)
    
    print(f"\nFile: {os.path.basename(f)}")
    print(f"  Shape: {mask.shape}")
    print(f"  Unique values: {unique}")
    
    for val, count in zip(unique, counts):
        class_name = "Unknown"
        if val == 0: class_name = "Clear"
        elif val == 1: class_name = "Shadow"
        elif val == 2: class_name = "Semi"
        elif val == 3: class_name = "Cloud"
        elif val == 255: class_name = "Ignore"
        
        print(f"  {class_name} ({val}): {count} pixels ({count/mask.size*100:.2f}%)")
        
        if val not in class_counts:
            class_counts[val] = 0
        class_counts[val] += count

print("\n" + "="*30)
print("Overall Distribution")
print("="*30)
total_pixels = sum(class_counts.values())
for val, count in class_counts.items():
    class_name = "Unknown"
    if val == 0: class_name = "Clear"
    elif val == 1: class_name = "Shadow"
    elif val == 2: class_name = "Semi"
    elif val == 3: class_name = "Cloud"
    elif val == 255: class_name = "Ignore"
    
    print(f"{class_name} ({val}): {count} pixels ({count/total_pixels*100:.2f}%)")
