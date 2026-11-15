"""Runnable stdlib-unittest tests. No training data required."""

import unittest
from pathlib import Path

import cv2
import numpy as np

from clock_reader import (
    ClockPreprocessConfig,
    prepare_clock,
    prepare_clock_from_path,
)


def synthetic_clock(center=(280, 225), radius=150, shape=(500, 650)):
    """Generate a dummy clock automatically (not a manually corrected photo)."""
    h, w = shape
    image = np.full((h, w, 3), (230, 229, 225), np.uint8)
    cv2.circle(image, center, radius + 7, (40, 40, 40), thickness=13)
    cv2.circle(image, center, radius - 5, (250, 250, 250), thickness=-1)
    for i in range(12):
        angle = np.pi * i / 6
        a = (int(center[0] + (radius - 12) * np.sin(angle)), int(center[1] - (radius - 12) * np.cos(angle)))
        b = (int(center[0] + (radius - 28) * np.sin(angle)), int(center[1] - (radius - 28) * np.cos(angle)))
        cv2.line(image, a, b, (30, 30, 30), 3)
    cv2.line(image, center, (center[0] + 95, center[1] - 30), (20, 20, 20), 4)
    cv2.line(image, center, (center[0] - 40, center[1] - 55), (20, 20, 20), 7)
    return image


class PreprocessingTests(unittest.TestCase):
    def test_detects_synthetic_clock_and_exposes_contract(self):
        image = synthetic_clock()
        result = prepare_clock(image)
        self.assertTrue(result.found, msg=result.metadata())
        self.assertLess(np.linalg.norm(np.array(result.source_center_xy) - (280, 225)), 15)
        self.assertLess(abs(result.source_radius_px - 150), 20)
        self.assertEqual(result.image_bgr.shape, (512, 512, 3))
        for field in ('gray', 'enhanced', 'edges', 'face_mask'):
            self.assertEqual(getattr(result, field).shape, (512, 512))
            self.assertEqual(getattr(result, field).dtype, np.uint8)
        self.assertEqual(result.center_xy, (255.5, 255.5))
        self.assertEqual(result.face_mask[256, 256], 255)
        self.assertEqual(result.face_mask[0, 0], 0)
        self.assertIsInstance(result.metadata(), dict)

    def test_handles_off_center_clock(self):
        result = prepare_clock(synthetic_clock(center=(460, 250), radius=110))
        self.assertTrue(result.found, msg=result.metadata())
        self.assertLess(abs(result.source_center_xy[0] - 460), 15)

    def test_handles_grayscale_and_config(self):
        gray = cv2.cvtColor(synthetic_clock(), cv2.COLOR_BGR2GRAY)
        result = prepare_clock(gray, ClockPreprocessConfig(output_size=256))
        self.assertEqual(result.gray.shape, (256, 256))

    def test_fallback_is_explicit(self):
        blank = np.full((300, 350, 3), 128, dtype=np.uint8)
        result = prepare_clock(blank)
        self.assertFalse(result.found)
        self.assertEqual(result.method, 'full_frame_fallback')
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.gray.shape, (512, 512))
        self.assertTrue(np.all(result.face_mask == 255))
        # Fallback must preserve the entire source frame, not crop a center square.
        rectangular = prepare_clock(np.full((120, 480, 3), 128, dtype=np.uint8))
        self.assertEqual(rectangular.source_bbox_xyxy, (0, 0, 480, 120))

    def test_course_template_photo_smoke_tests(self):
        # Illustrations supplied with the LaTeX template, NOT Kaggle training data.
        source = Path(__file__).resolve().parents[1] / 'report' / 'template' / 'Images'
        filenames = (
            'Phrontis-commons_wikimedia_org-braun-abw30-black.jpg',
            'kapa65-pixabay.jpg',
            'malvestida-unsplash.jpg',
        )
        for filename in filenames:
            with self.subTest(filename=filename):
                result = prepare_clock_from_path(source / filename)
                self.assertTrue(result.found)
                self.assertEqual(result.edges.shape, (512, 512))

    def test_rejects_invalid_data(self):
        with self.assertRaises(TypeError):
            prepare_clock(np.zeros((40, 40), dtype=np.float32))
        with self.assertRaises(ValueError):
            prepare_clock(np.empty((0, 0), dtype=np.uint8))
        with self.assertRaises(FileNotFoundError):
            prepare_clock_from_path('not_an_actual_image_file_001.jpg')


if __name__ == '__main__':
    unittest.main()
