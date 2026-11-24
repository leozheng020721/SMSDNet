"""Arrange existing prediction previews into an unmodified quantitative contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/results_century_nir_minmax')
    parser.add_argument('--all', action='store_true', help='Include all 22 previews instead of six representatives')
    args = parser.parse_args()
    selected = []
    for group in ('1', '4'):
        paths = sorted((args.output / group).glob('region_*/image_*_prediction.json'))
        if not paths:
            raise FileNotFoundError(f'No prediction metadata in group {group}')
        if not args.all:
            paths = [paths[index] for index in sorted({0, len(paths) // 2, len(paths) - 1})]
        selected.extend(paths)
    columns, cell_width, cell_height = 2, 650, 340
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * cell_width, rows * cell_height + 48), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), 'Century groups 1 and 4 | RGB (left) / predicted shadow in red (right)', fill='black')
    draw.text((12, 25), 'SEO NIR checkpoint; cross-domain predictions, not ground truth', fill='black')
    for index, path in enumerate(selected):
        record = json.loads(path.read_text(encoding='utf-8'))
        x, y = (index % columns) * cell_width, 48 + (index // columns) * cell_height
        with Image.open(record['outputs']['preview']) as source:
            image = source.convert('RGB')
        image.thumbnail((cell_width - 20, cell_height - 36), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (cell_width - image.width) // 2, y + 30))
        key = '/'.join(path.relative_to(args.output).parts)
        fraction = record['result']['shadow_fraction_of_valid']
        draw.text((x + 10, y + 8), f'{key} | predicted shadow {fraction:.2%}', fill='black')
    destination = args.output / ('preview_contact_sheet_all.jpg' if args.all else 'preview_contact_sheet.jpg')
    canvas.save(destination, quality=94)
    print(destination)


if __name__ == '__main__':
    main()
