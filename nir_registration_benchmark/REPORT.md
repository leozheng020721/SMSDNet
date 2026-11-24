# NIR registration method comparison

## Scope

The target is the existing S-EO RGB pixel grid, not merely a raster of the same size.
No original dataset file or production `nir_aligned_v1` product was modified.

The fixed sample contains 48 previously rejected matching cases and 9 previously
accepted controls. Each of JAX, OMA and UCSD contributes 16 rejected cases and
3 controls. All failed acquisition-index groups are represented. This is a
balanced diagnostic sample, not a population-proportional estimate for all 638 cases.

## Protocol

- Direct method: pretrained outdoor LoFTR, Kornia 0.7.3, maximum input edge 640,
  confidence candidates 0.5 and 0.2.
- PAN method: six SIFT/ORB multiscale candidates and LoFTR MSI-visible to PAN;
  PAN to current RGB is verified separately, then the transforms are composed.
- The raw NIR channel shares the MSI-visible image coordinates. Matching never
  changes NIR radiometry.
- Both routes use the same spatial-block holdout protocol: 4x4 cells, a 4-native-MSI
  pixel guard, at least 15 training inliers, two-dimensional support and at least
  5 percent feature hull coverage. Holdout median <=0.75 and P90 <=3 native-MSI
  pixels; at least 65 percent of held-out points within 1.5 pixels. NIR-valid
  raster coverage must be at least 10 percent.
- Dense matches and spatial holdout are internal consistency evidence, not
  independent ground truth. Candidate selection also uses these diagnostics.
- Structural scores are compared over shared valid regions.
- Final execution used one process per sample and a 32 MB GDAL cache. Earlier
  memory-interrupted `formal_v1` results are superseded by `formal_v2`.

## Main result

| Method | Rejected cohort accepted | Control cohort accepted |
| --- | ---: | ---: |
| Direct LoFTR | 0 / 48 | 9 / 9 |
| PAN multiscale SIFT/ORB | 0 / 48 | 9 / 9 |
| PAN LoFTR | 0 / 48 | 9 / 9 |
| Either PAN variant | 0 / 48 | 9 / 9 |

All 57 cases completed without runtime errors. The final manifest contains 57
unique keys and every preview exists. All per-case source fingerprints agree.
PAN/current-RGB grid alignment was verified for 52/57 samples; these 52 were
consistent with a shared grid. No unverified grid was assumed to be identity.

## Native-resolution follow-up

A separate 6-rejected/3-control probe used native PAN resolution, cubic-upscaled
MSI, official-style ORB cross-check matching and SIFT mutual-ratio matching.
No rejected case passed the complete route to existing RGB. All three controls
passed the SIFT and mutual-ratio ORB variants.

In UCSD_291/UCSD_291_3, native-resolution SIFT alone did verify MSI to PAN:
331 correspondences, 220 training inliers, spatial holdout median 0.169 and P90
0.447 native-MSI pixels. Its NIR validity mask supports 58.3004 percent of the
PAN canvas. The existing RGB has only 7.9217 percent nonblack pixels, and its
PAN-to-RGB spatial verification was insufficient. This demonstrates why failure
of the complete RGB route must not be labeled as failure of MSI-to-PAN itself.

The four-way orientation LoFTR follow-up on three near-candidate rejected cases
also produced no additional spatially verified full registration.

Native-PAN-resolution SIFT5000 mutual-ratio matching was then extended to all
57 main-test samples, using unchanged validation thresholds:

| Stage | Rejected cohort accepted | Control cohort accepted |
| --- | ---: | ---: |
| MSI to PAN alone | 1 / 48 | 9 / 9 |
| PAN to existing RGB bridge verified | 43 / 48 | 9 / 9 |
| Complete route to existing RGB | 0 / 48 | 9 / 9 |

The sole rejected-cohort PAN-only acceptance is UCSD_291/UCSD_291_3, described
above. All 57 samples completed without execution errors. Runtime was 159.9 s
for this SIFT-only extension. The result separates inability to verify the
existing RGB product from inability to register MSI/NIR to PAN.

## Interpretation

The same MSI-to-PAN transform should be applied to MSI R/G/B/NIR bands.
This geometric design favors PAN as the common reference. It does not recover
the official historical transform automatically, establish adequate crop overlap,
or repair a malformed RGB product. The experiment does not show a recovery-rate
advantage for a method on the complete existing-RGB task, and zero automatic
acceptances in this sample does not prove all 638 are impossible to register.

Keep the 19,570 existing accepted pairs for training. Investigate original crop
geometry, RPC initialization or historical transforms for rejected acquisitions;
when RGB generation is faulty, regenerate RGB and NIR together on a verified
PAN grid instead of attaching a new NIR image to that RGB without verification.

Recommended shared-band preprocessing design: use PAN as the reference, recover
or estimate a validated MSI-to-PAN mapping, and apply that same mapping to all
MSI bands. The choice reflects geometric consistency and the demonstrated
PAN-only case, not a claimed end-to-end recovery-rate advantage over LoFTR.

## Artifacts

- `formal_v2/comparison.csv`: per-case/per-method acceptance and diagnostics.
- `formal_v2/summary.json`: complete main-test counts.
- `formal_v2/cases/`: individual configurations, diagnostics and previews.
- `pan_fullres_probe/pan_only_qa.json`: staged native-resolution probe results.
- `pan_fullres_sift57/`: all-57 native-resolution SIFT staged results.
- `input_review/`: native MSI, PAN and current RGB contact sheets.
