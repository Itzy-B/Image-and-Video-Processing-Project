
"""Run Part 2 on images or folders and save debug images to preview_hands/."""
 
import sys
from pathlib import Path
 
from clock_reader import prepare_clock_from_path
from clock_reader.hand_detection import detect_hands
 
out = Path('preview_hands')
out.mkdir(exist_ok=True)
 
paths = []
for arg in sys.argv[1:]:
    p = Path(arg)
    paths += sorted(p.glob('*.jpg')) + sorted(p.glob('*.png')) if p.is_dir() else [p]
 
for path in paths:
    print(path.name)
    clock = prepare_clock_from_path(path)
    detect_hands(clock, debug_path=out / f'{path.stem}_hands.png')
 