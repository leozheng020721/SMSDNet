# Linux 服务器训练

目标服务器：`root@10.68.135.34`，SSH 端口 `22`，工作目录 `/data/data_set`。
服务器有两张 Quadro RTX 6000，每张 24 GB 显存，驱动为 535.183.01；使用下文 PyTorch 2.5.1 + CUDA 12.1 安装命令。
以下命令用于准备和运行，不表示已经在服务器安装或开始训练。

## 上传并解压

本机打包输出为 `remote_bundle/segformer_code.tar`、`remote_bundle/seo_data.tar` 和 `remote_bundle/SHA256SUMS`。
代码包顶层是 `segformer/`，数据包顶层是 `SEO/`，两个包解压到同一父目录。
这里的 `/data/data_set/SEO` 和 `/data/data_set/segformer` 都是远程服务器路径。
本机原数据和代码不移动，生成的包位于 `E:\MyPythonProject\segformer\remote_bundle`。
PNG 和 NIR TIFF 已经压缩，使用不再做二次压缩的 `.tar`，数据约 194 GiB。
数据包包含完整 20,275 幅 RGB、Min/Max shadow 和 uncertainty 掩膜，以及 19,584 对严格通过的 RGB-grid NIR/valid。
原 RGB+Max 配对基底保留，训练集合仍为 14,183 幅、验证集合为 3,236 幅。
候选或 PAN-only NIR 不参与打包；服务器预检会核对包内记录的划分数量和有序样本哈希。

在本机 PowerShell 执行：

```powershell
Set-Location E:\MyPythonProject\segformer
ssh -p 22 root@10.68.135.34 "mkdir -p /data/data_set"
scp -P 22 .\remote_bundle\segformer_code.tar .\remote_bundle\seo_data.tar .\remote_bundle\SHA256SUMS root@10.68.135.34:/data/data_set/
ssh -p 22 root@10.68.135.34
```

数据包较大，如果上传中断，可以使用 Windows 自带 SFTP 续传同一个包。
先在 PowerShell 连接，再在出现的 `sftp>` 提示符中逐行输入后续命令：

```text
sftp -P 22 root@10.68.135.34
lcd E:/MyPythonProject/segformer/remote_bundle
cd /data/data_set
reput seo_data.tar
put segformer_code.tar
put SHA256SUMS
bye
```

`reput` 会继续已存在的部分文件；只对本次同一份归档续传，完成后仍须做下面的 SHA-256 校验。
参见 [OpenSSH SFTP 手册](https://man.openbsd.org/sftp#reput)。

在服务器执行：

```bash
cd /data/data_set
df -h /data/data_set
sha256sum -c SHA256SUMS
tar -xf segformer_code.tar -C /data/data_set
tar -xf seo_data.tar -C /data/data_set
cd /data/data_set/segformer
```

先确认校验全部显示 `OK`，并确认磁盘同时容纳两个 tar、解压数据和后续 checkpoint。
目录应为：

```text
/data/data_set/
  segformer_code.tar
  seo_data.tar
  SHA256SUMS
  segformer/
    train.py
    preflight_remote.py
    config_seo_cca_nir_server.yaml
    requirements-server.txt
    checkpoints/
      seo_segformer_native512_aux_cca_nir_mixup_best.pth
  SEO/
    pansharpened_he/
    shadows_max/shadow_max_compressed/
    shadows_min/shadows/
    nir_aligned_v1/
      registration_with_candidates.csv
```

所有训练和预检命令均从 `/data/data_set/segformer` 执行。服务器配置的 RGB 数据根为 `../SEO`，NIR 根为 `../SEO/nir_aligned_v1`。
数据包中的 NIR 清单使用相对 NIR 根目录的路径；不要换回含 `E:/` 或反斜杠绝对路径的原始 Windows 清单。
数据包只需训练所用的 RGB、Min/Max 标签、Min uncertainty 和选定的 NIR/valid；不需要原始 PAN/MSI 或下载分卷。

## 创建环境并检查 GPU

以下使用已有的 Conda/Miniconda；若服务器尚无 Conda，先准备 Python 3.11 环境。

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uname -m
conda create -n seo-server python=3.11 -y
conda activate seo-server
python -m pip install --upgrade pip
```

本机测试通过的框架组合是 `torch==2.5.1+cu121` 与 `torchvision==0.20.1+cu121`。
确认是兼容 CUDA 12.1 wheel 的 RTX 6000 / Quadro RTX 6000 / RTX 6000 Ada、且驱动支持时，使用：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

命令来自 [PyTorch 官方旧版本安装表](https://pytorch.org/get-started/previous-versions/#v251)。本项目不需要 torchaudio。

若完整型号是 RTX PRO 6000 Blackwell 或 RTX 50 系列，不要执行上面的 cu121 安装行。
PyTorch 官方从 2.7 引入 Blackwell 和 CUDA 12.8 wheel 支持；应按具体 GPU、驱动和架构选择受支持的框架组合。
例如官方提供以下配对，但本项目尚未在这组 Linux 环境下实测，必须完成后续预检：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

两组框架安装命令按硬件选择一组执行，不能把 cu121 当作所有 RTX 6000 代际的通用方案。
参见 [PyTorch 2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/) 和 [官方安装表](https://pytorch.org/get-started/previous-versions/#v271)。

安装应用依赖：

```bash
python -m pip install -r requirements-server.txt
python -m pip check
export NO_ALBUMENTATIONS_UPDATE=1
export MPLBACKEND=Agg
```

`requirements-server.txt` 按本机已通过测试的应用版本固定，不锁死 Torch、CUDA 或 Triton。
本项目当前 SEO 流程可用 Pillow 读取单通道 float32 NIR TIFF，不要求安装 rasterio/GDAL 或 KappaSet 的 netCDF4。
Linux wheel、GPU 驱动和框架新组合仍需服务器预检确认，不能直接复制 Windows Conda 环境目录。

## 预检与训练

先运行预检：

```bash
cd /data/data_set/segformer
python preflight_remote.py --config config_seo_cca_nir_server.yaml --check-gpu --checkpoint checkpoints/seo_segformer_native512_aux_cca_nir_mixup_best.pth
```

确认数据/清单路径、标签形状、实际可用 train/val 样本及 GPU 检查通过后再开始训练。
配置保留 batch size 6、workers 2、prefetch 1、原学习率及损失设置。
`training.mixed_precision: true` 启用 CUDA FP16 autocast 和动态 GradScaler；损失、敏感注意力运算和滑窗拼接保留 FP32，CPU 自动使用 FP32。
启动日志应显示 `Precision: CUDA FP16 mixed precision (AMP + GradScaler)`，checkpoint 保存并恢复梯度缩放状态。
当前代码使用单张 GPU，两张卡不会自动合并显存；训练命令通过 `CUDA_VISIBLE_DEVICES=0` 指定第 0 张卡，需要第 1 张时改为 `1`。
监督协议为 Min/Max 一致区域并排除 Min uncertainty，缺监督文件沿用 `missing_policy: skip`；NIR 候选默认不参与。
该协议沿用本地 AOI 分组验证划分，不等同原论文固定九 AOI 的测试集。

建议用代码包中的旧 Max 协议最佳权重初始化新的 Min/Max 实验：

```bash
tmux new -s seo-minmax
conda activate seo-server
cd /data/data_set/segformer
export NO_ALBUMENTATIONS_UPDATE=1
export MPLBACKEND=Agg
mkdir -p logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u train.py --config config_seo_cca_nir_server.yaml --init-from checkpoints/seo_segformer_native512_aux_cca_nir_mixup_best.pth --init-auxiliary-heads 2>&1 | tee logs/seo_minmax_$(date +%Y%m%d_%H%M%S).log
```

`--init-from` 复用模型权重并启动新的优化器/调度器和轮次；指定辅助头迁移，形状不匹配的参数按现有迁移规则处理。
该命令会关闭构建时的 ImageNet 权重下载，使用打包的 checkpoint 初始化。
按 `Ctrl+B` 后按 `D` 退出 tmux 界面，训练继续；重新连接：

```bash
ssh -p 22 root@10.68.135.34
tmux attach -t seo-minmax
```

若 `tmux` 命令不存在，先在服务器安装 tmux 后再启动会话。
新实验输出位于：

```text
checkpoints/seo_segformer_native512_aux_cca_nir_mixup_minmax_best.pth
checkpoints/seo_segformer_native512_aux_cca_nir_mixup_minmax_last.pth
checkpoints/seo_segformer_native512_aux_cca_nir_mixup_minmax_history.csv
```

若这些新实验历史文件已存在，不要重复执行初始化命令覆盖实验；使用同协议断点续训，或为独立实验设置新名称。

## 中断后续训

只有已经生成上述 **Min/Max 新实验** 的 `last.pth` 后，才能按原配置恢复：

```bash
cd /data/data_set/segformer
CUDA_VISIBLE_DEVICES=0 python -u train.py --config config_seo_cca_nir_server.yaml --resume checkpoints/seo_segformer_native512_aux_cca_nir_mixup_minmax_last.pth
```

在 tmux 内执行恢复命令，可继续保留终端断开后的任务。恢复位置是最后一个完整保存轮次的下一轮，中断轮尚未保存的批次需要重跑。
同一 Min/Max 协议的旧 FP32 checkpoint 可以接着使用 FP16 训练，若没有梯度缩放状态则自动初始化；本次代码更新不要求从头训练。
不要对旧 Max 协议的 `last.pth` 使用新的 Min/Max 配置做 `--resume`，也不要把 `--init-from` 和 `--resume` 同时使用。
监督协议、归一化、划分、MixUp、损失或学习率策略变化属于新实验，不能伪装成同配置续训。
SSH 密码仅在终端提示中输入，不写入配置、脚本或分享的训练日志。
