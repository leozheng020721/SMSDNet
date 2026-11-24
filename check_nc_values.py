import netCDF4
import numpy as np
import glob
import os

# 查找一个NetCDF文件
nc_files = glob.glob("D:/datasets/KappaSet/kappaset/May/*.nc")
if not nc_files:
    print("No NetCDF files found in May folder. Trying recursive search...")
    nc_files = glob.glob("D:/datasets/KappaSet/**/*.nc", recursive=True)

if not nc_files:
    print("No NetCDF files found anywhere!")
    exit(1)

target_file = nc_files[0]
print(f"Analyzing file: {target_file}")

try:
    with netCDF4.Dataset(target_file, 'r') as nc:
        print("\nVariables:", nc.variables.keys())
        
        for band in ['B02', 'B03', 'B04', 'Label']:
            if band in nc.variables:
                data = nc.variables[band][:]
                print(f"\nBand {band}:")
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                print(f"  Min: {data.min()}")
                print(f"  Max: {data.max()}")
                print(f"  Mean: {data.mean():.4f}")
                
                # 统计非零值
                non_zero = data[data > 0]
                if len(non_zero) > 0:
                    print(f"  Non-zero Min: {non_zero.min()}")
                    print(f"  Non-zero Mean: {non_zero.mean():.4f}")
                else:
                    print("  ALL ZEROS!")
                    
                # 统计0值的比例
                zero_count = (data == 0).sum()
                print(f"  Zero count: {zero_count} ({zero_count/data.size*100:.2f}%)")

except Exception as e:
    print(f"Error reading file: {e}")
