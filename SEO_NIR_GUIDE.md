# S-EO NIR alignment

Run from this repository in PowerShell:

```powershell
& 'C:\Users\111\.conda\envs\rgb\python.exe' .\align_seo_nir.py --workers 2 --preview-count 6
```

The default input root is `E:/Dataset/SEO`. The script matches visible MSI
bands to the current `pansharpened_he` RGB images and resamples NIR once onto
that RGB pixel grid. PAN dimensions are checked; NIR is not pansharpened.
Original `msi`, `crops`, and RGB files are read without modification.

Default output preserves the source AOI folders and image identifiers:

```text
E:/Dataset/SEO/nir_aligned_v1/
  JAX_100/
    JAX_100_13_nir.tif
    JAX_100_13_valid.png
  registration.csv
  registration.jsonl
  processing_config.json
  summary.json
  _previews/
```

NIR TIFFs contain one float32 band in original raw DN units, with lossless
compression, NaN nodata, and an internal validity mask. The matching PNG has
255 for valid pixels and 0 for invalid pixels. Outputs describe RGB pixel
alignment; they do not claim an orthorectified map grid or calibrated reflectance.

CSV/JSONL records contain transforms, quality measurements, validity coverage,
and per-image status. `coverage_class=partial` means less than 95 percent of
the RGB canvas has valid NIR. Genuine partial overlap and black padding in
the target RGB can both reduce this global fraction. Use the validity mask
when selecting usable training pixels. Feature-match errors are registration
checks, not a guarantee of exact alignment everywhere.

Repeat the same command to resume. Existing successful outputs are skipped
when input fingerprints and output sizes match. Configuration and source-code
SHA-256 must match `processing_config.json`; choose a new output directory
after changing the algorithm or options. Failed records are retried and remain
visible in the manifests. Do not replace rejected registration with plain
resize or fill missing coverage with fabricated NIR measurements.

During training, apply identical crops, flips, rotations, and geometric warps
to RGB, NIR, validity, and labels. Normalize NIR separately using valid training
pixels; raw DN is not an 8-bit RGB channel. Mask invalid values before they
reach the network and preserve the mask when using any numeric fill value.

## PAN-based supplementation in the same folder

`recover_seo_nir_pan.py` processes only failed keys from the original
`registration.jsonl`. It writes missing products directly to the same AOI
folders in `E:/Dataset/SEO/nir_aligned_v1`, using unchanged `_nir.tif` and
`_valid.png` names. An existing original NIR product is never overwritten.

Run or resume recovery with one fresh worker per image:

```powershell
& 'C:\Users\111\.conda\envs\rgb\python.exe' .\recover_seo_nir_pan_isolated.py --workers 1
```

Each image runs in a new process so Windows releases its native allocations
on exit, limiting system-memory accumulation across images. Only process
lifetime changes; the production algorithm and existing configuration/source
hashes are reused.

The recovery uses native-PAN-resolution SIFT, an additional green/CLAHE
candidate and LoFTR against PAN. Every accepted mapping must pass the spatial
holdout and valid-coverage checks. Raw NIR is resampled only once. RGB grid
verification and composition are checked separately.

After supplementation, use `registration_all.csv` as the complete index:

- `original_rgb_ready` and `rgb_ready`, with `current_rgb_ready=True`, may be
  paired with the current RGB images. `rgb_training_manifest.csv` contains
  only these rows.
- `pan_only` is registered to PAN but has not been confirmed against current
  RGB. Its TIFF has `REFERENCE_GRID=PAN` and `CURRENT_RGB_READY=false`.
- `rejected` records have no fabricated output. `error` records are eligible
  for retry and are not treated as quality rejections.

The original `registration.*`, `processing_config.json`, and `summary.json`
remain the historical first-pass records. Recovery has separate
`pan_recovery_config.json`, `pan_recovery_registration.jsonl`,
`pan_recovery_registration.csv`, and `pan_recovery_summary.json`.
The wrapper resumes from the same recovery records and configuration. Files
and metadata must match before a completed result is reused; errors are retried.

`_pan_recovery_previews` contains percentile-stretched 8-bit PNGs for viewing.
Training continues to use the float32 raw-DN TIFF, not these previews.

## Relaxed, explicitly unverified candidates

The user-authorized relaxed pass is a separate pipeline. It selects only keys
whose latest strict PAN recovery status is `rejected`, and leaves every original
or strictly recovered product and historical ledger unchanged.

```powershell
& 'C:\Users\111\.conda\envs\rgb\python.exe' .\recover_seo_nir_candidates.py --workers 1
```

The default output is still `E:/Dataset/SEO/nir_aligned_v1/<AOI>/<stem>_nir.tif`
with the matching `_valid.png`. Every image runs in a fresh process. The pass
uses geometrically constrained affine and homography candidates, keeps the basic
finite/invertible/nonfolded footprint and valid-coverage checks, and relaxes the
statistical support requirements. Spatial holdout and feature-area coverage are
not acceptance gates. Reported fit residuals are training residuals, not an
independent estimate of registration accuracy.

All new products have `REGISTRATION_STATUS=UNVERIFIED_CANDIDATE`,
`REGISTRATION_VERIFIED=false`, and `CURRENT_RGB_READY=false`. A verified PAN to
RGB bridge permits output on the current RGB grid, but does not verify the
upstream MSI to PAN candidate. `RGB_GRID_VERIFIED=true` refers only to that
bridge. A validity mask indicates valid resampling support, not correct alignment.
It covers NIR support only; experiments must also intersect it with the usable
RGB region and the label-validity mask. RGB black padding is not removed by the
NIR validity mask automatically.

Use these new manifests to inspect or explicitly opt into candidate data:

- `pan_candidate_registration.jsonl` and `.csv`: this pass, its matrices,
  relaxed fit diagnostics, failures, and candidate statuses.
- `registration_with_candidates.csv`: complete original/strict/candidate index.
- `rgb_candidate_manifest.csv`: only `candidate_rgb` products, all unverified.
- `pan_candidate_config.json`, `pan_candidate_baseline.json`, and
  `pan_candidate_summary.json`: frozen configuration, preservation baseline,
  and final counts. Run-time errors remain retryable on the same command.
- `_pan_candidate_previews`: stretched PNGs for viewing raw-DN candidate NIR.

`candidate_pan` has only a PAN reference and must not be treated as a current-RGB
pair. The original `rgb_training_manifest.csv` remains the accepted-data list;
candidates are never silently added. `registration_all.csv` remains the strict
pass's historical index. Its older auditor does not include candidate products;
use `verify_seo_nir_candidates.py` for the combined directory after this pass.

Before processing, the pass freezes historical-ledger hashes and protected
product size/mtime metadata. Resuming requires the same cohort, sources,
configuration, and baseline. Existing original or foreign products cannot be
overwritten. Inputs remain read-only, NIR remains float32 raw DN, and invalid
support remains NaN plus an explicit mask.
