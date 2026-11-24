# Original MS / PAN / fused color check

Scope: group 7, the exact new 1024 x 1024 footprints of showcase M01 and M02,
both source dates (2024-12-31 and 2025-12-10). Source files were read only.
No model inference, pansharpening change, or source-data modification was made.

## Finding

The strong blue color in the selected 2024-12-31 examples is already present
in the original multispectral data before pansharpening. This was verified
using original MS pixels, not only interpolated or fused pixels, and a common
linear display range for all RGB channels. The original TIFF tag explicitly
states BLU(0)/GRN(1)/RED(2)/NIR(3); RGB display uses bands 3, 2, 1 (one based).

| Area / date | Blue subset native B, G, R DN medians | Native blue B/R median | Nonblue B/R median | Native blue subset fraction |
| --- | --- | --- | --- | --- |
| M01 / 2024-12-31 | 102, 16, 12 | 6.379 | 0.529 | 36.69% |
| M02 / 2024-12-31 | 141, 35, 23 | 5.694 | 0.773 | 56.23% |
| M01 / 2025-12-10 | 74, 12, 34 | 2.162 | 0.689 | 2.27% |
| M02 / 2025-12-10 | 82, 26, 47.5 | 1.720 | 0.794 | 0.19% |

The descriptive blue subset is B > 1.5*G, B > 1.5*R, and B > 10 DN inside
the valid source footprint. It is not a calibrated spectral class, and the
fractions describe these selected examples only. The median of B/R differs
from the ratio of the separately computed channel medians.

The single-band PAN contains brightness, not an independent blue channel.
For all four source windows, the stored fused values exactly matched the
tested equal-weight four-band Brovey calculation on all valid pixels:

    intensity = (B + G + R + NIR) / 4
    gain = clip(PAN / intensity, 0, 8)
    fused_band = round(clip(upsampled_MS_band * gain, 0, 4095))

Thus fusion does not independently boost blue relative to red; all bands
share the same gain before rounding/clipping. Cubic interpolation, rounding,
clipping, and subsequent channel display stretches can still alter the
displayed appearance. The fusion does change local brightness substantially:
in M01 / 2024-12-31, median gain over the darkest MS-intensity quartile was
0.357, and 29.34% of those pixels had every fused visible DN at most 5. In
M02 / 2024-12-31, the selected blue-pixel gain median was 1.051, so the blue
parts of that example were not uniformly darkened. Avoid claiming that all
blue pixels are caused by PAN or all are made darker by fusion.

## Artifacts

Each `*_comparison.jpg` shows, from left to right:

1. Original native-resolution MS RGB, nearest-neighbor enlarged for display.
2. Original PAN in grayscale, using its own p98 range.
3. Fused RGB with exactly the same scalar display range as the original MS.
4. Fused RGB using its own scalar p98 display range.

For MS and fused RGB, all three channels share the same scalar range, zero
lower bound, gamma 1. The yellow polygon locates the selected rectified
1024 x 1024 footprint in the unrectified original source. Its axis-aligned
bounding box is shown, so source image dimensions differ from 1024 x 1024.
The PNGs have no polygon annotation; the native MS PNG retains native size.
Black invalid areas in the second-date controls are outside available valid
source content; the numeric subsets exclude invalid areas.

`raw_color_report.json` records source paths and file sizes, original TIFF
metadata, exact inverse-homography coordinates, native MS/source windows,
display scales, and per-case radiometric statistics. `check_raw_color.py`
reproduces the audit using the `rgb` conda environment.

## Limits

The TIFFs contain integer DN values, scales 1 and offsets 0; no physical
calibration coefficients were found in their metadata. B/R DN ratios do not
equal calibrated surface-reflectance ratios. The audit establishes that the
selected blue appearance predates fusion, but does not establish a sensor
fault, atmospheric cause, or whether all shadowed terrain should look blue.
The other groups and all mountain pixels were not exhaustively audited.
