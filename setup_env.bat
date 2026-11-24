@echo off
REM KappaSet云影检测 - 环境配置脚本 (Windows)

echo ============================================
echo KappaSet 云影检测系统 - 环境配置
echo ============================================
echo.

echo [1/6] 创建Conda虚拟环境 (Python 3.11)...
call conda create -n cloud_detect_env python=3.11 -y

echo [2/6] 激活虚拟环境...
call conda activate cloud_detect_env

echo [3/6] 配置显存管理...
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo [4/6] 安装 PyTorch Nightly (CUDA 12.8)...
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo [5/6] 验证安装...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo [6/6] 安装依赖库...
pip install opencv-python-headless albumentations matplotlib scikit-learn tqdm pyyaml
pip install segmentation-models-pytorch timm

echo.
echo ============================================
echo 配置完成！
echo ============================================
echo 下一步: python train.py
pause
