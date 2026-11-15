"""Part 1 command-line demonstration. No hand/time reading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .preprocessing import draw_detection, load_image, prepare_clock


def main() -> None:
    parser = argparse.ArgumentParser(description='Detect and preprocess an analogue clock (Part 1 only).')
    parser.add_argument('image', type=Path, help='Path to an original clock photograph')
    parser.add_argument('--out', type=Path, default=Path('preview'), help='Folder for debug images and metadata')
    args = parser.parse_args()

    image = load_image(args.image)
    region = prepare_clock(image)
    args.out.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(args.out / '01_detection.jpg'), draw_detection(image, region))
    cv2.imwrite(str(args.out / '02_normalized_clock.png'), region.image_bgr)
    cv2.imwrite(str(args.out / '03_enhanced.png'), region.enhanced)
    cv2.imwrite(str(args.out / '04_edges.png'), region.edges)
    cv2.imwrite(str(args.out / '05_face_mask.png'), region.face_mask)
    (args.out / 'metadata.json').write_text(json.dumps(region.metadata(), indent=2), encoding='utf-8')
    print(json.dumps(region.metadata(), indent=2))
    if not region.found:
        print('WARNING: no reliable clock circle found; full-frame fallback used.')
    print(f'Part 1 preview saved to: {args.out.resolve()}')


if __name__ == '__main__':
    main()
