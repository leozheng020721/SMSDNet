# SEO label audit: six held-out scenes

This is a small, deliberately selected diagnostic sample, not an unbiased estimate of the entire dataset. No model weights, training settings, source imagery, or labels were changed.

## Split and selection

The checkpoint's `GroupShuffleSplit(random_state=42, test_size=0.2)` was reproduced on all 17,700 sorted RGB/Max pairs before NIR filtering. It produces 3,278 validation pairs, and 3,236 remain after accepted RGB-grid NIR filtering and complete Min/Max supervision. This matches the logged validation size. Two different validation AOIs were selected per city, with at least 90% valid NIR coverage. All six selected registrations have `original_rgb_ready` status and measured NIR coverage above 99%.

The actual training target is shadow where the Max mask is zero, with Min/Max disagreement or any nonzero Min-side uncertainty assigned the ignore index 255. Source mask value 255 denotes nonshadow; target value 255 denotes ignore.

## Exclusion of original Max shadow pixels

Every percentage below uses original Max shadow pixels as its denominator. The disagreement and uncertainty columns overlap; union is their combined exclusion. Uncertainty-only excludes the overlap.

| Sample | Disagreement | Uncertainty | Union | Uncertainty-only |
| --- | ---: | ---: | ---: | ---: |
| JAX_255/JAX_255_24 | 84.46% | 4.33% | 84.48% | 0.02% |
| JAX_231/JAX_231_24 | 80.20% | 9.04% | 80.78% | 0.58% |
| OMA_260/OMA_260_0 | 75.37% | 13.57% | 75.54% | 0.17% |
| OMA_194/OMA_194_0 | 76.56% | 13.13% | 76.67% | 0.10% |
| UCSD_189/UCSD_189_0 | 80.87% | 2.29% | 81.11% | 0.23% |
| UCSD_169/UCSD_169_0 | 81.59% | 2.99% | 82.13% | 0.54% |

Min and Max shadows are not spatially nested in these samples: both Min-only and Max-only shadow pixels exist. Therefore a simple difference of their total areas does not equal their disagreement area. The supervised pixels comprise 59.50% to 85.22% of each image, and shadow is only 1.45% to 4.34% of retained pixels. This is the intended conservative agreement protocol, not proof that the files are corrupt or that labels cause the target-domain model error. It does mean that substantial boundary regions have no direct segmentation supervision.

## Concrete visual examples

See `label_detail_crops.jpg` and its original-image crop coordinates in `label_detail_crops.json`. The middle panels show Min boundaries in red and Max boundaries in cyan. The right panels show retained shadow supervision in red and ignored pixels in yellow.

- JAX_255/JAX_255_24: crop x=500, y=150, width=850, height=800. The large polygonal airport-building shadow has visibly different Min/Max boundaries. Most thin roof shadows are ignored under agreement.
- JAX_231/JAX_231_24: crop x=250, y=250, width=950, height=750. On original row y=400, the first strong dark interval near the left hangar edge starts at x=347 when grayscale <40, the Min shadow starts at x=357, and Max starts at x=346. This is a local visible-edge mismatch of roughly 10 pixels for Min, not a measured global image translation. At row y=380, a strong dark interval is x=349..633, while Min covers x=358..560 and Max extends to x=691. Both labels can differ from a visible brightness edge.

Brightness thresholds and Canny edges are not manual shadow ground truth. Material texture, penumbra, and nearby roof or wall edges can change these proxies. Their role here is locating examples for inspection, not certifying label accuracy.

## Vegetation masks

No separately named vegetation-mask directory exists in the local SEO root, Min-label root, or Max-label root that were inspected. These directory listings are saved in `sample_manifest.json`. The current `dataset.py` supervision path loads Min, Max, and Min-side uncertainty, with no separately loaded vegetation mask. This does not establish whether some vegetation information is embedded in uncertainty or available in an unextracted/external archive.

## Reproduce

`python test/diagnose_dataset_model/label_sample_audit.py --audit-existing` rereads only the six selected samples and updates their statistics and visual diagnostics, without rescanning candidate images.
