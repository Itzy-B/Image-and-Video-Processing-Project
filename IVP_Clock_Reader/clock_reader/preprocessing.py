
"""Clock detection and preprocessing for the IVP project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True)
class ClockPreprocessConfig:
    """Parameters used for every input image."""

    output_size: int = 512
    crop_scale: float = 1.13
    max_detection_side: int = 1000
    min_radius_fraction: float = 0.075
    detection_threshold: float = 0.22

    def __post_init__(self) -> None:
        if self.output_size < 64:
            raise ValueError('output_size must be at least 64')
        if self.crop_scale <= 1.0:
            raise ValueError('crop_scale must exceed 1.0')
        if self.max_detection_side < 128:
            raise ValueError('max_detection_side must be at least 128')
        if not 0.01 <= self.min_radius_fraction <= 0.45:
            raise ValueError('min_radius_fraction must be between 0.01 and 0.45')
        if not 0.0 <= self.detection_threshold <= 1.0:
            raise ValueError('detection_threshold must be between 0 and 1')


@dataclass
class ClockRegion:
    """Processed images and clock geometry for hand detection."""

    image_bgr: np.ndarray
    gray: np.ndarray
    enhanced: np.ndarray
    edges: np.ndarray
    face_mask: np.ndarray
    center_xy: tuple[float, float]
    radius_px: float
    source_center_xy: tuple[float, float]
    source_radius_px: float
    source_bbox_xyxy: tuple[int, int, int, int]
    found: bool
    confidence: float
    method: Literal['hough_circle', 'contour_circle', 'full_frame_fallback']

    def metadata(self) -> dict:
        """Return detection metadata without image arrays."""
        return {
            'found': self.found,
            'confidence': round(float(self.confidence), 4),
            'method': self.method,
            'center_xy': list(self.center_xy),
            'radius_px': round(float(self.radius_px), 3),
            'source_center_xy': list(self.source_center_xy),
            'source_radius_px': round(float(self.source_radius_px), 3),
            'source_bbox_xyxy': list(self.source_bbox_xyxy),
            'output_size': self.gray.shape[0],
        }


def load_image(path: str | Path) -> np.ndarray:
    """Load an image in BGR format."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return image


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError('image must be a NumPy array (BGR, BGRA or grayscale)')
    if image.dtype != np.uint8:
        raise TypeError('image must be uint8; use cv2.imread for loading')
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError('image must be a nonempty 2D or 3D array')
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError('image must have 1, 3 or 4 channels')


def _rank_candidates(
        edges: np.ndarray,
        candidates: list[tuple[float, float, float, str]],
        lab_image: np.ndarray,
) -> list[tuple[float, float, float, str, float]]:
    if not candidates:
        return []
    h, w = edges.shape

    # Score circles using their edges, size, and interior appearance.
    near = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    angles = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    sins, coss = np.sin(angles), np.cos(angles)

    grid = np.linspace(-0.65, 0.65, 29)
    gx, gy = np.meshgrid(grid, grid)
    keep = gx * gx + gy * gy <= 0.65**2
    dx, dy = gx[keep], gy[keep]
    scored: list[tuple[float, float, float, str, float]] = []

    for x, y, radius, method in candidates:
        if radius < 8 or x < 0 or y < 0 or x >= w or y >= h:
            continue

        xs = np.rint(x + radius * coss).astype(np.int32)
        ys = np.rint(y + radius * sins).astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        seen = np.count_nonzero(valid)

        if seen < 0.7 * len(angles):
            continue

        coverage = np.count_nonzero(near[ys[valid], xs[valid]]) / len(angles)
        size_score = min(1.0, radius / (0.37 * min(h, w)))

        sample_x = np.rint(x + radius * dx).astype(np.int32)
        sample_y = np.rint(y + radius * dy).astype(np.int32)
        inside = (sample_x >= 0) & (sample_x < w) & (sample_y >= 0) & (sample_y < h)

        if np.count_nonzero(inside) < 0.75 * len(dx):
            continue

        center_edge_density = np.count_nonzero(
            edges[sample_y[inside], sample_x[inside]]
        ) / np.count_nonzero(inside)

        clutter_penalty = 1.2 * max(0.0, center_edge_density - 0.09)

        colors = lab_image[sample_y[inside], sample_x[inside]].astype(np.float32)
        typical_color = np.median(colors, axis=0)
        color_distance = np.linalg.norm(colors - typical_color, axis=1)
        coherence = float(np.mean(color_distance < 25.0))

        score = float(np.clip(
            0.78 * coverage + 0.10 * size_score + 0.22 * coherence - clutter_penalty,
            0.0, 1.0,
            ))

        scored.append((x, y, radius, method, score))

    return sorted(scored, key=lambda candidate: candidate[-1], reverse=True)


def _find_circle(
        gray: np.ndarray,
        bgr: np.ndarray,
        config: ClockPreprocessConfig
) -> tuple[float, float, float, str, float] | None:

    h, w = gray.shape
    scale = min(1.0, config.max_detection_side / max(h, w))

    if scale < 1.0:
        small = cv2.resize(
            gray,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA
        )
    else:
        small = gray

    sh, sw = small.shape

    if scale < 1.0:
        small_bgr = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small_bgr = bgr

    small_lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)
    short_side = min(sh, sw)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adjusted = clahe.apply(small)
    smooth = cv2.GaussianBlur(adjusted, (5, 5), 1.4)
    edges = cv2.Canny(smooth, 60, 160)

    min_radius = max(10, int(config.min_radius_fraction * short_side))
    max_radius = max(min_radius + 2, int(0.55 * short_side))
    candidates: list[tuple[float, float, float, str]] = []

    # Detect circular boundaries.
    detected = cv2.HoughCircles(
        smooth, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(25, int(short_side * 0.14)),
        param1=135, param2=31,
        minRadius=min_radius, maxRadius=max_radius,
    )

    if detected is not None:
        for x, y, r in detected[0][:100]:
            candidates.append((float(x), float(y), float(r), 'hough_circle'))

    # Try contour-based candidates as well.
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = np.pi * min_radius**2 * 0.35

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:120]:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < min_area or perimeter < 1:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < 0.58 or circularity > 1.2:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)

        if not (min_radius <= radius <= max_radius):
            continue

        bbox = cv2.boundingRect(contour)
        aspect_ratio = bbox[2] / max(bbox[3], 1)

        if not 0.75 <= aspect_ratio <= 1.35:
            continue

        candidates.append((float(x), float(y), float(radius), 'contour_circle'))

    ranked = _rank_candidates(edges, candidates, small_lab)

    if not ranked:
        return None

    # Refine the strongest candidates around their initial positions.
    refined_candidates: list[tuple[float, float, float, str]] = []
    offsets = (-0.15, -0.10, 0.0, 0.10, 0.15)

    for x, y, radius, method, _ in ranked[:12]:
        if method == 'contour_circle' and ranked[0][-1] > 0.90:
            continue

        for ox in offsets:
            for oy in offsets:
                for radius_scale in (0.90, 1.0, 1.10):
                    refined_candidates.append((
                        x + ox * radius, y + oy * radius,
                        radius * radius_scale, method,
                    ))

    refined = _rank_candidates(edges, refined_candidates, small_lab)
    match = max(
        (ranked[0], refined[0] if refined else ranked[0]),
        key=lambda value: value[-1]
    )

    x, y, radius, method, score = match
    return (x / scale, y / scale, radius / scale, method, score)


def prepare_clock(
        image: np.ndarray,
        config: ClockPreprocessConfig | None = None,
) -> ClockRegion:
    """Return processed images and geometry, with a fallback if detection fails."""

    config = config or ClockPreprocessConfig()
    bgr = _to_bgr(image)
    height, width = bgr.shape[:2]
    source_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    found_circle = _find_circle(source_gray, bgr, config)

    if found_circle is not None and found_circle[-1] >= config.detection_threshold:
        sx, sy, source_radius, method, confidence = found_circle
        found = True
    else:
        # Fall back to the complete image if detection fails.
        sx, sy = (width - 1) / 2.0, (height - 1) / 2.0
        source_radius = max(width, height) / (2.0 * config.crop_scale)
        method, confidence, found = 'full_frame_fallback', 0.0, False

    size = config.output_size
    normalized_center = (size - 1) / 2.0
    zoom = normalized_center / (config.crop_scale * source_radius)

    affine = np.float32([
        [zoom, 0, normalized_center - zoom * sx],
        [0, zoom, normalized_center - zoom * sy],
    ])

    # Center and resize the clock before creating the hand-detection inputs.
    square = cv2.warpAffine(
        bgr, affine, (size, size), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.GaussianBlur(enhanced, (5, 5), 1.1)
    edges = cv2.Canny(smooth, threshold1=55, threshold2=140)

    radius = normalized_center / config.crop_scale
    yy, xx = np.ogrid[:size, :size]

    # Remove edges outside the detected dial.
    if found:
        face_mask = np.asarray(
            (xx - normalized_center) ** 2 + (yy - normalized_center) ** 2 <= (radius * 0.91) ** 2,
            dtype=np.uint8,
            ) * 255
    else:
        face_mask = np.full((size, size), 255, dtype=np.uint8)

    edges = cv2.bitwise_and(edges, face_mask)

    half_width = source_radius * config.crop_scale
    bbox = (
        max(0, int(np.floor(sx - half_width))),
        max(0, int(np.floor(sy - half_width))),
        min(width, int(np.ceil(sx + half_width))),
        min(height, int(np.ceil(sy + half_width))),
    )

    return ClockRegion(
        image_bgr=square,
        gray=gray,
        enhanced=enhanced,
        edges=edges,
        face_mask=face_mask,
        center_xy=(normalized_center, normalized_center),
        radius_px=radius,
        source_center_xy=(sx, sy),
        source_radius_px=source_radius,
        source_bbox_xyxy=bbox,
        found=found,
        confidence=confidence,
        method=method,
    )


def prepare_clock_from_path(
        path: str | Path,
        config: ClockPreprocessConfig | None = None,
) -> ClockRegion:
    """Load and preprocess an image from disk."""
    return prepare_clock(load_image(path), config)


def draw_detection(image: np.ndarray, result: ClockRegion) -> np.ndarray:
    """Draw the detected circle on a copy of the original image."""

    vis = _to_bgr(image).copy()
    x, y = (int(round(v)) for v in result.source_center_xy)
    radius = int(round(result.source_radius_px))
    color = (0, 200, 0) if result.found else (0, 165, 255)

    cv2.circle(vis, (x, y), radius, color, max(2, round(min(vis.shape[:2]) / 300)))
    cv2.drawMarker(vis, (x, y), color, cv2.MARKER_CROSS, 22, 2)

    label = f'{result.method} (score={result.confidence:.2f})'
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return vis
