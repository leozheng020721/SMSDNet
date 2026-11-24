# Native-resolution folder inference

Use `folder_inference.py` for unlabeled image folders. It does not select a
training/validation split. Sources are sorted by their relative paths.

## Region 01: two matching output folders

`region_01` has two original images. Each 9216 x 5701 image produces 216
non-overlapping 512 x 512 tiles: 432 tile pairs, or 864 PNG files in total.

```powershell
& "C:/Users/111/.conda/envs/seg/python.exe" "E:/MyPythonProject/segformer/folder_inference.py" --config "E:/MyPythonProject/segformer/config_seo_cca_scratch.yaml" --checkpoint "E:/MyPythonProject/segformer/checkpoints/seo_segformer_native512_aux_cca_scratch_mixup_best.pth" --input "E:/Dataset/21century/registration_output/rectified/4/region_01" --output "E:/MyPythonProject/segformer/predictions_region1" --tile-size 512 --stride 384 --tile-batch-size 2 --export-layout paired
```

The `paired` layout saves exactly two PNGs for each tile:

```text
predictions_region1/
  masks/
    image_1_rectified_r00000_c00000.png
    image_2_rectified_r00000_c00000.png
  marked/
    image_1_rectified_r00000_c00000.png
    image_2_rectified_r00000_c00000.png
  tile_manifest.csv
```

Names match exactly between the two folders. `r` is the original row/y
coordinate and `c` is the original column/x coordinate. The original image
identifier prevents the two source images from sharing a name.

`masks` contains black/white predictions (white = shadow). `marked` contains
the corresponding RGB tiles with red outlines around predicted shadow
regions. Outlines do not fill the shadow interiors. Both outputs are 512 x
512, with black padding where necessary. No separate RGB or filled overlay
PNGs are written in this layout. The CSV records coordinates and source paths.

Add `--count 1` to process just the first original image (216 pairs / 432
PNGs), and use a separate output directory for that preview. A scan without
model loading uses `--scan-only --export-layout paired` with the same input
and output paths; config/checkpoint are optional for scanning.

## Preview one original image

Run in PowerShell from `E:/MyPythonProject/segformer`:

```powershell
& "C:/Users/111/.conda/envs/seg/python.exe" "E:/MyPythonProject/segformer/folder_inference.py" --config "E:/MyPythonProject/segformer/config_seo_cca_scratch.yaml" --checkpoint "E:/MyPythonProject/segformer/checkpoints/seo_segformer_native512_aux_cca_scratch_mixup_best.pth" --input "E:/Dataset/21century/registration_output/rectified/4" --output "E:/MyPythonProject/segformer/predictions_rectified4_preview" --count 1 --tile-size 512 --stride 384 --tile-batch-size 2
```

`--count 1` selects one original image, including all of its export tiles.
The first source is `region_01/image_1_rectified`; its 9216 x 5701 pixels
produce 18 x 12 = 216 export tile groups.

## Count or process all images

The following scan writes manifests and counts without loading the model:

```powershell
& "C:/Users/111/.conda/envs/seg/python.exe" "E:/MyPythonProject/segformer/folder_inference.py" --input "E:/Dataset/21century/registration_output/rectified/4" --output "E:/MyPythonProject/segformer/rectified4_scan" --tile-size 512 --scan-only
```

After reviewing the preview, process all 12 original images with:

```powershell
& "C:/Users/111/.conda/envs/seg/python.exe" "E:/MyPythonProject/segformer/folder_inference.py" --config "E:/MyPythonProject/segformer/config_seo_cca_scratch.yaml" --checkpoint "E:/MyPythonProject/segformer/checkpoints/seo_segformer_native512_aux_cca_scratch_mixup_best.pth" --input "E:/Dataset/21century/registration_output/rectified/4" --output "E:/MyPythonProject/segformer/predictions_rectified4" --tile-size 512 --stride 384 --tile-batch-size 2
```

## Output and input conventions

- Export tiles do not overlap. They preserve the original pixel scale.
- Right and bottom edge tiles are padded with black to exactly 512 x 512.
- Model windows overlap at stride 384; overlapping logits are averaged
  before the final class decision and non-overlapping export.
- The default `triplet` layout has `_rgb.png`, `_shadow.png`, and `_overlay.png` files
  in its region folder. White (255) in `_shadow.png` means predicted shadow;
  black (0) means background or invalid/padded pixels. The overlay marks
  predicted shadow in red.
- Coordinates and valid extents are recorded in the tile manifest so the
  exported tiles can be traced back to their original image.
- Corresponding TIFF and ENVI copies are counted once. Registration preview
  JPGs and geometry-valid masks are excluded from the image list.
- These four-band uint16 ENVI images are stored as blue, green, red, NIR.
  The model uses RGB bands only, in red, green, blue order. A fixed stretch
  from sampled valid-pixel 2nd/98th percentiles is calculated for each original
  image and reused for every window; NIR is not an alpha channel.
- Validity masks describe the usable registered image area. They are not
  shadow labels, and this folder cannot provide BER without shadow labels.
- The saved full checkpoint is loaded strictly, including the cross-scale
  attention module. No pretrained encoder download or training is started.

With the current folder contents, the export count is:

| Region | Original width x height | Tiles per image | Both images |
| --- | --- | --- | --- |
| region_01 | 9216 x 5701 | 216 | 432 |
| region_02 | 5795 x 4628 | 120 | 240 |
| region_03 | 7740 x 11035 | 352 | 704 |
| region_04 | 6009 x 4117 | 108 | 216 |
| region_05 | 10154 x 4112 | 180 | 360 |
| region_06 | 7695 x 2901 | 96 | 192 |
| Total | 12 original images | | 2144 |

The default preserves the complete rectangular grid, including four tiles
with no valid source pixels. Excluding those four would leave 2140 groups.
Three PNGs per group means 6432 PNGs for the full folder, plus manifests.
