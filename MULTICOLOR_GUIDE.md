# SegFormer with lightweight HSV/Lab branches

For SEO native-resolution crops and RGB/HSV/Lab auxiliary supervision, use
`config_seo_crop_aux.yaml` and read [the SEO guide](SEO_CROP_AUX_GUIDE.md).
The existing configurations below retain the original resize-only ablation.

The base variant adds two independent lightweight color branches and spatial
gates at the four MiT encoder stages. Cross-scale criss-cross attention is now
optional; `config_seo_cca.yaml` enables it after fusion, before the decoder.
Disagreement calibration is not implemented. The original CBAM/CoordAtt
implementation remains available through the original configuration.

## Architecture

- RGB uses the existing MiT encoder and ImageNet normalization.
- The model restores RGB to [0, 1] and derives HSV and CIE Lab internally. All
  geometric and photometric image augmentations therefore remain aligned across
  the three color representations. The dataset still returns `(image, mask)`.
- HSV channels are normalized to [0, 1]. Lab uses L/100, (a+128)/255 and
  (b+128)/255. Both representations then use mean=0.5 and std=0.5.
- Each auxiliary stage pools spatially, applies a learnable 1x1 projection, and
  uses GELU. Default auxiliary widths are [8, 16, 32, 64], a deliberately smaller
  starting point than the proposal's C_i/4 widths.
- Each scale predicts separate one-channel spatial gates for HSV and Lab. Their
  projected features are injected into the matching RGB feature by a gated
  residual. The negative initial gate bias limits the initial residual size.
- The existing All-MLP decoder and segmentation head produce the final logits.

When `model.multicolor.cross_scale_attention.enabled` is true, fused stage 3
queries aligned fused stage 4 keys/values. Each query attends jointly to its
row and column, with the center counted once. A learned residual scale starts
at zero. This is a cross-scale adaptation of
[CCNet's criss-cross attention](https://arxiv.org/abs/1811.11721), rather than
three independent branch attention modules. Native branch auxiliary heads
keep their existing inputs. Old configurations omit this setting and retain
their original parameter layouts and behavior.

Pooling is a reasonable inexpensive auxiliary encoder, but pooling alone cannot
learn useful feature projections. The 1x1 layers keep the branches learnable.
Average pooling is the default because it summarizes local color statistics;
`model.multicolor.pooling: "max"` is also supported for a controlled comparison.
Pooling can discard small shadow boundaries. Any accuracy or BER improvement
must be established experimentally against the RGB baseline.

## Train

Activate the existing environment and start the new model:

```powershell
Set-Location E:\MyPythonProject\segformer
conda activate rgb
python train.py --config config_multicolor.yaml
```

Train the RGB-only comparison with identical data, loss, optimizer and schedule:

```powershell
python train.py --config config_multicolor_baseline.yaml
```

The new configurations keep the existing Focal + Dice loss to isolate the
architecture change. BER, IoU, F1 and Kappa are printed in the terminal. The
checkpoint with the lowest validation BER is saved as:

```text
checkpoints/aisd_segformer_hsv_lab_pool_gates_best.pth
checkpoints/aisd_segformer_rgb_baseline_best.pth
```

The new modules require training. Existing CBAM or RGB-only full checkpoints
cannot be loaded strictly into this architecture. ImageNet initialization still
applies to the RGB encoder. The first training run may download the encoder
weights if they are not cached; offline runs can set `encoder_weights: null` to
train without that initialization.

For an optional compatible RGB-weight transfer, use `--init-from` with an
existing checkpoint instead of `--resume`. This starts a new experiment and
does not restore optimizer state or the old best BER. The legacy experiments
randomly mixed Train412 and Val51 before splitting, so their checkpoints may
already have seen images from the new validation split. Use ImageNet
initialization for both new runs when making a fair architecture comparison;
an old AISD checkpoint warm start is appropriate only for transfer/debugging,
not as the basis for that controlled comparison. Untrained auxiliary branches
cannot demonstrate an improvement.

## Data splits

The new configurations select `dataset.split_mode: "predefined"`:

```text
AISD/Train412/shadow/   AISD/Train412/mask/
AISD/Val51/shadow/      AISD/Val51/mask/
Test51/shadow/         Test51/mask/
```

Images and masks are paired by filename stem. The fixed-split reader accepts
TIFF, PNG, JPG/JPEG and BMP files. Missing masks and duplicate stems within a
split raise an error. Relative split directories are resolved under `root_dir`;
`test_dir` can be absolute. Edit these paths if your data is elsewhere.

Configurations without `split_mode` retain the legacy random 90/10 train/val
split, with sorted discovery for a stable order. A test split always requires an
explicit `test_dir`; validation images are never silently reused as test data.

## Test

Run the held-out AISD test split with the matching architecture configuration:

```powershell
python inference.py --config config_multicolor.yaml --checkpoint checkpoints/aisd_segformer_hsv_lab_pool_gates_best.pth --split test --output multicolor_test_results
```

The terminal reports BER (%), shadow-class IoU, F1 and Cohen's Kappa. Compare
models using the same split, image size, preprocessing and metric implementation.
Resume training with `--resume` and the matching architecture configuration.
