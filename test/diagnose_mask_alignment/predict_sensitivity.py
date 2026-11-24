"""Read-only local sensitivity check for the reported 1024-pixel crop."""

from pathlib import Path
import json
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from infer_century_nir import load_model, prepare_tile, tile_starts


def main():
    started = time.perf_counter()
    metadata_path = HERE.parent / 'results_century_nir_minmax/4/region_01/image_1_prediction.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    settings = metadata['signature']
    crop_x, crop_y, size = 5120, 3072, 1024
    tile_size, stride, batch_size = 512, 384, 4
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    model, _, _ = load_model(metadata['checkpoint']['path'], device)
    sums = {mode: np.zeros((2, size, size), dtype=np.float32)
            for mode in ('real', 'no_nir', 'hflip')}
    coverage = np.zeros((size, size), dtype=np.float32)
    evaluated = []
    with rasterio.Env(GDAL_CACHEMAX=128 * 1024 * 1024), rasterio.open(settings['source']) as source:
        rows = tile_starts(source.height, tile_size, stride)
        columns = tile_starts(source.width, tile_size, stride)
        for top in rows:
            if top >= crop_y + size or top + tile_size <= crop_y:
                continue
            for offset in range(0, len(columns), batch_size):
                positions = columns[offset:offset + batch_size]
                if not any(left < crop_x + size and left + tile_size > crop_x for left in positions):
                    continue
                tiles = [prepare_tile(source, left, top, tile_size, settings['mean'], settings['std'],
                                      settings['dn_max'], metadata['statistics']['nir_low'],
                                      metadata['statistics']['nir_high']) for left in positions]
                rgb = torch.from_numpy(np.stack([tile[0] for tile in tiles])).to(device)
                nir = torch.from_numpy(np.stack([tile[1] for tile in tiles])).to(device)
                nir_valid = torch.from_numpy(np.stack([tile[2] for tile in tiles])).to(device)
                for mode in sums:
                    image_input, nir_input, validity_input = rgb, nir, nir_valid
                    if mode == 'no_nir':
                        nir_input, validity_input = torch.zeros_like(nir), torch.zeros_like(nir_valid)
                    elif mode == 'hflip':
                        image_input = rgb.flip(-1)
                        nir_input, validity_input = nir.flip(-1), nir_valid.flip(-1)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                        logits = model(image_input, nir=nir_input, nir_valid=validity_input)
                        if mode == 'hflip':
                            logits = logits.flip(-1)
                    logits = logits.float().cpu().numpy()
                    for index, left in enumerate(positions):
                        x0, x1 = max(left, crop_x), min(left + tile_size, crop_x + size)
                        y0, y1 = max(top, crop_y), min(top + tile_size, crop_y + size)
                        if x0 >= x1 or y0 >= y1:
                            continue
                        src_y, src_x = slice(y0 - top, y1 - top), slice(x0 - left, x1 - left)
                        dst_y, dst_x = slice(y0 - crop_y, y1 - crop_y), slice(x0 - crop_x, x1 - crop_x)
                        valid = tiles[index][3][src_y, src_x]
                        sums[mode][:, dst_y, dst_x] += logits[index, :, src_y, src_x] * valid
                        if mode == 'real':
                            coverage[dst_y, dst_x] += valid
                evaluated.append({'top': top, 'columns': positions})
        arrays = {}
        valid = coverage > 0
        for mode, total in sums.items():
            np.divide(total, coverage[None], out=total, where=valid[None])
            probability = 1.0 / (1.0 + np.exp(np.clip(total[0] - total[1], -80, 80)))
            probability[~valid] = -1
            arrays[mode] = probability.astype(np.float32)
        with rasterio.open(metadata['outputs']['probability']) as saved:
            arrays['saved'] = saved.read(1, window=Window(crop_x, crop_y, size, size))
        arrays['rgb'] = source.read([3, 2, 1], window=Window(crop_x, crop_y, size, size))
        arrays['valid'] = valid
    report = {'source': settings['source'], 'crop': {'x': crop_x, 'y': crop_y, 'size': size},
              'modes': {}, 'batch_groups': evaluated,
              'note': 'Input sensitivity of the same trained model, not a retrained RGB-only ablation.'}
    for mode in ('real', 'no_nir', 'hflip'):
        difference = np.abs(arrays[mode][valid] - arrays['saved'][valid])
        report['modes'][mode] = {
            'mean_absolute_difference_vs_saved': float(difference.mean()),
            'max_absolute_difference_vs_saved': float(difference.max()),
            'binary_agreement_vs_saved': float(np.mean((arrays[mode][valid] > .5) == (arrays['saved'][valid] > .5))),
            'shadow_fraction': float(np.mean(arrays[mode][valid] > .5)),
        }
    report['elapsed_seconds'] = time.perf_counter() - started
    np.savez_compressed(HERE / 'sensitivity_arrays.npz', **arrays)
    (HERE / 'sensitivity_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
