"""Analogue clock reader: Part 1 (preprocessing) is implemented.

Part 2 (hand detection) and Part 3 (time decoding) are not yet implemented.
"""

from .preprocessing import (
    ClockPreprocessConfig,
    ClockRegion,
    draw_detection,
    load_image,
    prepare_clock,
    prepare_clock_from_path,
)

__all__ = [
    'ClockPreprocessConfig',
    'ClockRegion',
    'draw_detection',
    'load_image',
    'prepare_clock',
    'prepare_clock_from_path',
]
