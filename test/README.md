# 21 世纪 RGB + NIR 阴影推理

入口：`infer_century_nir.py`。代码与结果都保存在本项目 `test` 文件夹，原始影像只读。

## 已核实的波段

输入为 `E:\Dataset\21century\registration_output\rectified` 下的 `1`、`4` 两组。
第 1 组 5 个区域、10 幅 TIFF；第 4 组 6 个区域、12 幅 TIFF，共 22 幅。
实际 TIFF 描述与 ENVI 头文件均明确标为 `BLUE, GREEN, RED, NIR`，数据为 uint16。
读取 RGB 时使用第 3、2、1 波段；NIR 使用第 4 波段。

源全色锐化 TIFF 的标签记录了四波段等权 `WEIGHTED_BROVEY` 融合。
`rectify_pairs.py` 对全部四个波段使用同一个透视矩阵，正射后的 NIR 已与本文件 RGB 对齐。
JPEG 预览只有三通道；推理读取保留四波段的 TIFF。

## 权重和预处理

默认权重为 `checkpoints/seo_segformer_native512_aux_cca_nir_mixup_minmax_best.pth`。
模型结构、ImageNet 均值/标准差、NIR 分位数均从 checkpoint 自带配置读取，严格加载参数。
模型在推理时明确接收真实 `nir` 与布尔 `nir_valid`，保留 HSV、Lab、CCA 和 NIR 模块。
不需要推理影像的 Min/Max 标签；Min/Max 是该权重训练及验证时采用的监督协议。

RGB 沿用本批 21 世纪影像已有流程：`clip(DN / 4095, 0, 1)` 后做 ImageNet 标准化。
4095 是这批数据使用的 DN 上限，不能因为容器为 uint16 就改成除以 65535。
SEO 训练 RGB 来自 `pansharpened_he`；固定 DN 映射不等同于该数据集的直方图均衡，
因此这些结果属于跨数据源预测，应检查暗地物和浅阴影。

NIR 按每幅影像有效像素的 1%/99% 分位数裁剪后映射到 `[-1, 1]`，与训练定义一致。
使用分块 uint16 直方图精确计算整幅分位数，不逐窗口拉伸。
内部有效掩膜和四波段非全零条件共同确定有效区；NIR 单独为零不代表无效。
无效处 NIR 为零、`nir_valid=False`；常量 NIR 按训练行为归一化为零。

使用 512×512 窗口、384 步长，最后一个窗口靠齐影像边缘。
重叠处先均匀平均原始 logits，再计算阴影概率，默认阈值 0.5，平局归非阴影。
条带拼接和 128 MiB GDAL 缓存控制内存，最大输入 7740×11035 也不整图载入浮点副本。
CUDA 默认使用 FP16 autocast，拼接与概率计算为 FP32；CPU 自动使用 FP32。

## 执行

本机已有的 `rgb` 环境包含 PyTorch 2.5.1+cu121、SMP 0.4.0 和 Rasterio 1.4.3。
`seg` 环境缺少 Rasterio，因此直接使用 `rgb` 环境，无需另建环境。

在 PowerShell 中运行全部 22 幅：

```powershell
& 'C:\Users\111\.conda\envs\rgb\python.exe' `
  'E:\MyPythonProject\segformer\test\infer_century_nir.py' `
  --groups 1 4 --save-probability
```

只试第一幅可加 `--limit 1`。显存不足时可改 `--batch-size 1`。
默认输出目录为 `E:\MyPythonProject\segformer\test\results_century_nir_minmax`。
默认输出单页、不压缩 TIFF，使用独立有效区文件，方便旧版 ENVI 打开。
需要压缩时可显式增加 `--compression deflate`。
完整且参数签名一致的已有结果会跳过；修改权重或参数后需要新输出目录或 `--overwrite`。

仅运行第 4 组 `region_01`，使用同一权重关闭 NIR，保存对照概率图：

```powershell
& 'C:\Users\111\.conda\envs\rgb\python.exe' 'E:\MyPythonProject\segformer\test\infer_century_nir.py' --groups 4 --regions region_01 --disable-nir --save-probability
```

`--disable-nir` 只将送入模型的 `nir_valid` 置为 False，并将 NIR 输入置零，
RGB、影像有效区、窗口和拼接方式保持不变，不重新训练，也不修改权重。
默认另存到 `test/results_century_nir_disabled/4/region_01`，不会覆盖原来的真实 NIR 结果。
去掉 `--disable-nir` 即为正常 RGB+NIR 推理，默认仍写入 `test/results_century_nir_minmax`。
元数据签名记录 `nir_mode`，防止将两种模式的结果误当成相同结果复用。
这是当前权重的敏感性检查；正式评估增加 NIR 的收益仍需分别训练有/无 NIR 模型。

## 输出

保留 `1/region_XX` 与 `4/region_XX` 的层级，每幅源图生成：

- `image_N_shadow_mask.tif`：uint8，0=非阴影、255=阴影。
- `image_N_valid_mask.tif`：uint8，0=无效区、255=有效区。无效区不可当作非阴影真值。
- `image_N_shadow_probability.tif`：使用 `--save-probability` 时生成，float32，范围 0–1，无效区 -1。
- `image_N_preview.jpg`：左侧固定 DN 映射 RGB，右侧红色覆盖预测阴影，便于快速检查。
- `image_N_prediction.json`：模型 SHA-256、输入、波段、NIR 分位数、参数和统计信息。

根目录 `run_summary.json` 汇总全部影像。
输出尺寸、像素变换和 CRS 与对应输入一致；输入本身没有地图 CRS 的，输出也不虚构 CRS。
这些是模型预测，不是人工真值，也没有计算 21 世纪数据上的 BER。

## 检查代码

```powershell
& 'C:\Users\111\.conda\envs\rgb\python.exe' -m unittest discover `
  -s 'E:\MyPythonProject\segformer\test' -p 'test_century_inference.py' -v
```

测试检查真实四波段 TIFF 的通道顺序、有效区、NIR 精确分位数、常量/零值 NIR、
不规则尺寸及小图拼接、平均 logits 而非平均概率，以及输出尺寸和坐标。

独立核验全部输出并重新生成预览汇总：

```powershell
& 'C:\Users\111\.conda\envs\rgb\python.exe' `
  'E:\MyPythonProject\segformer\test\verify_century_outputs.py'
& 'C:\Users\111\.conda\envs\rgb\python.exe' `
  'E:\MyPythonProject\segformer\test\make_preview_sheet.py'
```

核验报告为结果根目录的 `verification_report.json`，逐块检查尺寸、有效区、
掩膜编码、概率范围、概率与掩膜的一致性，以及原始输入文件是否保持不变。
`preview_contact_sheet.jpg` 展示六幅代表性预览；加 `--all` 可生成全部 22 幅的汇总。

本次预览中，第 4 组部分屋顶区域也被预测为阴影，存在疑似误检，需要人工复核。
文件完整性与数值一致性核验不等于预测准确率验证。

## 阴影轮廓切块

`export_shadow_outline_tiles.py` 使用已有 mask，在 RGB 原图上只绘制红色阴影轮廓，
不填充阴影内部，不重新运行模型。默认处理第 4 组 `region_01` 的两幅影像：

```powershell
& 'C:\Users\111\.conda\envs\rgb\python.exe' `
  'E:\MyPythonProject\segformer\test\export_shadow_outline_tiles.py'
```

结果位于 `test/shadow_outline_tiles/4/region_01/image_1` 和 `image_2`。
每幅 54 张无重叠的 1024x1024 RGB PNG，共 108 张；最后一行下方补黑。
文件名记录行列号和原图左上角像素坐标，完整索引在 `tile_index.json`。
轮廓默认向阴影内部绘制 2 像素宽，排除与无效区相邻的边界。
切块计算使用邻域，避免块边缘产生伪轮廓；RGB 映射与前面的推理预览一致。
`outline_samples.jpg` 汇总四张样例，PNG 文件本身不叠加标题或文字。
