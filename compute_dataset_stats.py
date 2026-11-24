import netCDF4
import numpy as np
import glob
import os
from tqdm import tqdm

def compute_stats(root_dir, max_reflectance=0.3, num_samples=1000):
    print(f"Searching for NetCDF files in {root_dir}...")
    files = glob.glob(os.path.join(root_dir, "kappaset", "**", "*.nc"), recursive=True)
    
    if not files:
        print("No files found!")
        return

    # Randomly sample files
    np.random.shuffle(files)
    files = files[:num_samples]
    print(f"Computing stats on {len(files)} files...")
    
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0
    
    for f in tqdm(files):
        try:
            with netCDF4.Dataset(f, 'r') as nc:
                # Read RGB
                b02 = nc.variables['B02'][:]
                b03 = nc.variables['B03'][:]
                b04 = nc.variables['B04'][:]
                
                # Stack RGB
                img = np.stack([b04, b03, b02], axis=-1).astype(np.float32)
                
                # Apply Stretch
                img = img / max_reflectance
                img = np.clip(img, 0.0, 1.0)
                
                # Accumulate
                pixel_sum += img.sum(axis=(0, 1))
                pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
                pixel_count += img.shape[0] * img.shape[1]
                
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
            
    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
    
    print("\n" + "="*40)
    print("Dataset Statistics (after / 0.3 stretch)")
    print("="*40)
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    print(f"Mean (formatted): [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"Std (formatted):  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

if __name__ == "__main__":
    compute_stats("D:/datasets/KappaSet")
