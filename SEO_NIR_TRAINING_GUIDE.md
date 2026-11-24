# SegFormer with an aligned NIR auxiliary branch

Use `config_seo_cca_nir.yaml` for the existing RGB/HSV/Lab model with cross-scale
attention and an additional independent NIR branch. RGB remains the three-channel
input to the pretrained MiT-B3 encoder. NIR uses a one-channel pooled encoder with
the same four auxiliary widths, masked gated fusion at each scale, and the same
auxiliary prediction head design as HSV. The existing cross-scale attention stays
after fusion.

## Data and supervision

The default configuration selects strict successful RGB-grid registrations from
`E:/Dataset/SEO/nir_aligned_v1/registration_with_candidates.csv`. PAN-only results,
rejected registrations and unverified candidates are excluded. Set
`dataset.nir.include_candidates: true` only to explicitly include `candidate_rgb`
records in a new experiment. AOI groups are split on the original paired RGB data
before filtering for NIR, retaining the original seed-42 validation groups.
The new default uses Min/Max agreement supervision:

```yaml
dataset:
  supervision:
    mode: minmax_agreement
    min_root: shadows_min/shadows
    max_root: shadows_max/shadow_max_compressed
    ignore_uncertainty: true
    missing_policy: skip
```

The mask roots are relative to `dataset.root_dir`. Corresponding Min/Max masks
already use the RGB image grid; no new image registration is required. Check
filenames and spatial sizes before use. Raw shadow-mask values are 0 for shadow
and 255 for non-shadow. The target uses the Min mask, maps shadow to class 1, and
sets label 255 wherever Min and Max disagree or the Min uncertainty mask is
nonzero. That uncertainty mask already includes unseen and vegetated regions.
Do not require uncertainty values to equal 255: the nonzero rule handles both
0/1 and 0/255 masks. The inspected Min PNGs use a palette: raw palette indices
are 0/1, but grayscale decoding yields 0/255. Use cv2 grayscale reading or Pillow
convert('L') before comparing Min and Max; raw palette indices are not labels.
The same ignored labels apply to training, validation, all branch losses and
evaluation metrics. Label 255 here is the generated ignore target, distinct from
raw-mask value 255 meaning non-shadow.

The AOI split and strict RGB-grid NIR filtering are unchanged. This is not a full
reproduction of the paper's geographic training/test split. The previous
Max-only labeled-data intersection contained 17,444 images: 14,208 training and
3,236 validation images. A complete archive-membership audit found 25 of those
training keys missing both the Min shadow and uncertainty files; all 3,236
validation keys were present. With the explicit `missing_policy: skip` setting,
the expected new counts are 14,183 training and 3,236 validation images, or
17,419 total. These estimates use archive filenames, not a full real-data shape
and pixel audit.

Missing-file handling defaults to `error` unless `skip` is explicitly configured.
Under `skip`, filtering occurs after the original AOI split and strict NIR
filtering. Missing Min/uncertainty pairs are reported with counts and example
keys; AOIs are not reassigned to replace them. The Min `shadow_masks` and
`uncertainty_masks` directories must both exist. An unextracted archive or missing
whole mask directory still raises an error instead of silently skipping the
entire dataset.

Extract the downloaded Min shadow archive in PowerShell:

```powershell
New-Item -ItemType Directory -Path 'E:\Dataset\SEO\shadows_min' -Force -ErrorAction Stop | Out-Null
tar -xzf 'E:\Dataset\shadow-eo\shadows_min.tar.gz' -C 'E:\Dataset\SEO\shadows_min'
if ($LASTEXITCODE -ne 0) { throw 'Min shadow archive extraction failed.' }
```

The archive retains its `shadows` directory, so the resulting mask directories
are `E:\Dataset\SEO\shadows_min\shadows\shadow_masks` and
`E:\Dataset\SEO\shadows_min\shadows\uncertainty_masks`. These commands have not
been executed as part of this configuration change.

NIR TIFF values are raw floating-point DN, not 8-bit RGB. Valid full-image pixels
are clipped/scaled using the 1st and 99th percentiles to [-1, 1]. NaN, TIFF-invalid
pixels and invalid PNG-mask pixels become zero with a false validity mask. RGB,
label, NIR and validity undergo the same spatial transforms. RGB brightness and
ImageNet normalization do not change NIR. Small-image padding is invalid for NIR.
Training crops with less than 5% valid NIR can make up to four additional attempts
and use the best available crop. Validation uses original-size images without
random crops.

The loss formula remains unchanged; `Y` includes the new ignored regions:

```text
L_main = 2 * Focal(P_final, Y) + Dice(P_final, Y)
L_aux  = (CE(P_rgb, Y) + CE(P_hsv, Y) + CE(P_lab, Y)
          + CE(P_nir, Y restricted to valid NIR pixels)) / 4
L      = L_main + 0.4 * L_aux
```

All branches ignore label 255. Missing NIR does not remove RGB supervision. An
all-invalid NIR crop contributes zero NIR CE and zero NIR fusion residual.
MixUp uses one permutation and lambda for RGB and NIR, with validity-weighted NIR
mixing and separate source masks for each target's NIR CE. Full-image validation
passes matching NIR tiles and validity masks through the same sliding windows.
Validation loss uses the final prediction only, as in the previous experiment.

## Training

Run in PowerShell from the project directory:

```powershell
Set-Location 'E:\MyPythonProject\segformer'
$env:NO_ALBUMENTATIONS_UPDATE = '1'
& 'C:\Users\111\.conda\envs\seg\python.exe' train.py --config config_seo_cca_nir.yaml --init-from checkpoints/seo_segformer_native512_aux_cca_nir_mixup_best.pth --init-auxiliary-heads
```

This starts a new Min/Max supervision stage from the previous NIR best model,
including compatible auxiliary prediction heads. It creates a fresh optimizer
and scheduler and recomputes the best validation score under the new protocol.
It does not resume the old Max-only training state. Native 512-pixel crops, batch
size 6, 50 epochs, AdamW, initial LR 6e-5 and cosine decay remain configured.
Both local and server NIR configs enable `training.mixed_precision: true`.
CUDA training and validation use FP16 autocast with dynamic gradient scaling
for training. Loss reductions, sensitive attention operations and CPU tile
stitching remain FP32. CPU runs fall back to FP32. The startup log reports the
active precision, and checkpoints save and restore the scaler state. A same-
protocol FP32 checkpoint without scaler state can continue with a fresh scaler.
The TIFF reader can use Pillow in the existing `seg` environment; rasterio is
optional. Omitting `--init-from` starts a new model from the configured encoder
initialization rather than the previous trained NIR weights.

Outputs use the separate prefix `checkpoints/seo_segformer_native512_aux_cca_nir_mixup_minmax`:
`_best.pth`, `_last.pth`, and `_history.csv`. History includes `train_aux_nir`.
Only resume a checkpoint from the same Min/Max supervision protocol:

```powershell
& 'C:\Users\111\.conda\envs\seg\python.exe' train.py --config config_seo_cca_nir.yaml --resume checkpoints/seo_segformer_native512_aux_cca_nir_mixup_minmax_last.pth
```

Do not use `--resume` with the old Max-only `_last.pth`; transfer weights using
`--init-from` when changing supervision protocols.
Checkpoints are saved after each completed epoch; an interrupted epoch restarts
from its beginning. Model, optimizer and scheduler states are restored together.
NIR normalization uses float32 in-place operations to avoid full-image float64
temporaries. `prefetch_factor: 1` limits queued batches per worker. These memory
options do not alter the sampling geometry or model settings.
Finite-value mask checks operate on 128-row blocks and reuse the validity PNG
buffer. A missing optional rasterio package is handled before Pillow reading,
so a later memory failure is reported without an unrelated import-error context.
After restoration, training releases the loaded CPU checkpoint containers before
starting the next epoch. These changes reduce per-process demand; the machine
must still have enough free system commit for CUDA, Python and worker processes.
To initialize from an older RGB/HSV/Lab checkpoint, use `--init-from` rather than
`--resume`; add `--init-auxiliary-heads` to transfer matching old prediction heads.
Keep the same selected image subset when comparing against a no-NIR baseline.

## Evaluating previous weights

The explicit runtime supervision configuration takes priority when loading a
checkpoint for inference. Evaluate old NIR weights with the new Min/Max labels
without updating those weights:

```powershell
& 'C:\Users\111\.conda\envs\seg\python.exe' inference.py --config config_seo_cca_nir.yaml --checkpoint checkpoints/seo_segformer_native512_aux_cca_nir_mixup_best.pth --output seo_val_minmax
```

This requires the extracted Min masks. Its metrics use a different supervision
protocol from the old Max-only history and must be labeled accordingly.

## Verification

Under the previous Max-only supervision, the RTX 3060 completed a real MiT-B3
6x512x512 training batch with MixUp, finite four-branch losses and an updated NIR
encoder. Peak allocated GPU memory was 8.59 GiB (8.92 GiB reserved). A complete
validation image also passed through RGB/NIR sliding-window inference. That was
a smoke check without checkpoint output, not a trained-model accuracy evaluation.

For the new Min/Max protocol, read-only checks covered all archive filenames and
one real sample, `JAX_101_13`: its RGB, Min, Max and uncertainty masks all use
1549x1408 pixels. Its palette indices are 0/1 and grayscale values are 0/255.
The later remote-bundle check read both JAX_100_13 and JAX_101_13 through the real
RGB/Min/Max/NIR loader: aligned shapes and finite normalized NIR both passed.
Full-sample shape and pixel checks, archive extraction and formal training have
not been performed. The CPU regression suite passed 121 tests with one optional
ONNX dependency skip (122 total). It covers agreement and nonzero uncertainty,
missing-pair reports and unchanged AOI membership, ignored labels through RGB/NIR
geometry, masked losses and MixUp gradients, checkpoint protocol guards, and a
synthetic MiT-B0 Min/Max+NIR optimizer step followed by sliding-window validation.
These checks verify the new code path, not a new trained-model BER.

The Min-side uncertainty and agreement rule follows the
[official dataset implementation](https://github.com/centreborelli/shadow-eo-detection/blob/main/dataset.py).
