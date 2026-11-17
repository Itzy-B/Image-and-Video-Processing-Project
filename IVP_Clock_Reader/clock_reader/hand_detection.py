
"""Part 2: hand detection with the (probabilistic) Hough transform."""
 
from __future__ import annotations
 
from dataclasses import dataclass
import math
from pathlib import Path
 
import cv2
import numpy as np
 
from .preprocessing import ClockRegion
 
 
@dataclass
class HandCandidate:
    angle_deg: float     # clockwise from 12 o'clock, 0 to 360
    length: float        # as a fraction of the radius R
    thickness_px: float
    score: float         # total length of supporting segments
 
 
@dataclass
class Segment:
    """One Hough segment, described relative to the hub."""
    x1: float            # near end
    y1: float
    x2: float            # far end
    y2: float
    d_near: float        # distance hub -> near end
    d_far: float         # distance hub -> far end
    length: float
    perp: float          # signed distance from hub to the extended line
    angle_deg: float     # direction hub -> far end, clockwise from 12
 
 
# Step 1: find all straight edge segments.
def find_segments(clock: ClockRegion) -> list[list[float]]:
    R = clock.radius_px
    segs = cv2.HoughLinesP(clock.edges, rho=1, theta=np.pi / 180, threshold=25,
                           minLineLength=int(0.15 * R), maxLineGap=6)
    return [] if segs is None else segs.reshape(-1, 4).astype(float).tolist()
 
 
# Step 3: measure one segment relative to the hub.
def measure_segment(seg: list[float], hub: tuple[float, float]) -> Segment:
    hx, hy = hub
    x1, y1, x2, y2 = seg
    d1 = math.hypot(x1 - hx, y1 - hy)
    d2 = math.hypot(x2 - hx, y2 - hy)
    if d1 > d2:  # make (x1, y1) the near end
        x1, y1, x2, y2, d1, d2 = x2, y2, x1, y1, d2, d1
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    perp = (dx * (y1 - hy) - dy * (x1 - hx)) / length
    angle = math.degrees(math.atan2(x2 - hx, -(y2 - hy))) % 360
    return Segment(x1, y1, x2, y2, d1, d2, length, perp, angle)
 
 
# Step 3 filter: points at the hub and reaches far enough out.
def is_hand_like(s: Segment, R: float) -> bool:
    return abs(s.perp) < 0.06 * R and s.d_far > 0.25 * R
 
 
# Step 3b: a hand starts at the hub. Tick marks also point at the hub, but start near the rim.
def starts_at_hub(s: Segment, R: float) -> bool:
    return s.d_near < 0.4 * R
 
 
def angle_diff_deg(a: float, b: float) -> float:
    """Smallest difference between two angles, e.g. 359 vs 1 -> 2."""
    return abs((a - b + 180) % 360 - 180)
 
 
# Step 4b: far-out pieces (a hand broken by a gap) join a hand only if they line up with it.
def attach_pieces(groups: list[list[Segment]], pieces: list[Segment],
                  max_diff_deg: float = 5.0) -> list[Segment]:
    attached = []
    for s in pieces:
        for g in groups:
            g_angle = circular_mean_deg([t.angle_deg for t in g], [t.length for t in g])
            if angle_diff_deg(s.angle_deg, g_angle) <= max_diff_deg:
                g.append(s)
                attached.append(s)
                break
    return attached
 
 
# Step 4: segments whose angles are close together belong to the same hand.
def group_by_angle(segments: list[Segment], max_gap_deg: float = 8.0) -> list[list[Segment]]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s.angle_deg)
    groups = [[ordered[0]]]
    for s in ordered[1:]:
        if s.angle_deg - groups[-1][-1].angle_deg <= max_gap_deg:
            groups[-1].append(s)      # close to the previous one: same hand
        else:
            groups.append([s])        # big jump: start a new hand
    # Wrap-around: e.g. 358 and 2 degrees are only 4 degrees apart.
    if len(groups) > 1 and (ordered[0].angle_deg + 360) - ordered[-1].angle_deg <= max_gap_deg:
        groups[0] = groups.pop() + groups[0]
    return groups
 
 
def circular_mean_deg(angles: list[float], weights: list[float]) -> float:
    """Weighted average of angles, done with unit vectors so 359 and 1 average to 0, not 180."""
    x = sum(w * math.sin(math.radians(a)) for a, w in zip(angles, weights))
    y = sum(w * math.cos(math.radians(a)) for a, w in zip(angles, weights))
    return round(math.degrees(math.atan2(x, y)), 6) % 360  # round: avoid 359.9999999 for 0
 
 
# Step 5: check a group and turn it into one hand candidate.
def group_to_candidate(group: list[Segment], R: float) -> HandCandidate | None:
    if min(s.d_near for s in group) > 0.4 * R:
        return None  # nothing starts near the hub: a numeral or tick, not a hand
    weights = [s.length for s in group]
    angle = circular_mean_deg([s.angle_deg for s in group], weights)
    length = max(s.d_far for s in group) / R
    perps = [s.perp for s in group]
    thickness = max(perps) - min(perps)  # the two outline sides have opposite signs
    return HandCandidate(angle, length, thickness, score=sum(weights))
 
 
def draw_debug(clock: ClockRegion, hub, anchors: list[Segment], attached: list[Segment],
               rejected: list[Segment], candidates: list[HandCandidate]) -> np.ndarray:
    vis = cv2.cvtColor(clock.edges, cv2.COLOR_GRAY2BGR)
    # Red = rejected, green = starts at the hub, cyan = extra piece attached to a hand.
    for segments, colour in ((rejected, (0, 0, 255)), (anchors, (0, 255, 0)), (attached, (255, 255, 0))):
        for s in segments:
            cv2.line(vis, (round(s.x1), round(s.y1)), (round(s.x2), round(s.y2)), colour, 2)
    # Each final hand candidate as a yellow ray from the hub.
    for c in candidates:
        r = c.length * clock.radius_px
        tip = (round(hub[0] + r * math.sin(math.radians(c.angle_deg))),
               round(hub[1] - r * math.cos(math.radians(c.angle_deg))))
        cv2.line(vis, (round(hub[0]), round(hub[1])), tip, (0, 255, 255), 1)
        cv2.circle(vis, tip, 5, (0, 255, 255), 2)
    cv2.drawMarker(vis, (round(hub[0]), round(hub[1])), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
    return vis
 
 
def detect_hands(clock: ClockRegion, debug_path: str | Path | None = None) -> list[HandCandidate]:
    R = clock.radius_px
    hub = clock.center_xy  # Step 2 (finding the real hub) comes later.
 
    segments = [measure_segment(seg, hub) for seg in find_segments(clock)]
    hand_like = [s for s in segments if is_hand_like(s, R)]
    anchors = [s for s in hand_like if starts_at_hub(s, R)]
    pieces = [s for s in hand_like if not starts_at_hub(s, R)]
 
    # Group only the segments that start at the hub, so tick marks can't form or chain groups.
    groups = group_by_angle(anchors)
    attached = attach_pieces(groups, pieces)
    used = {id(s) for s in anchors + attached}
    rejected = [s for s in segments if id(s) not in used]
 
    candidates = [c for c in (group_to_candidate(g, R) for g in groups) if c is not None]
    candidates.sort(key=lambda c: c.score, reverse=True)
 
    if debug_path is not None:
        cv2.imwrite(str(debug_path), draw_debug(clock, hub, anchors, attached, rejected, candidates))
 
    print(f'  {len(segments)} segments, {len(anchors)} start at hub, {len(attached)} attached '
          f'-> {len(candidates)} hand candidates')
    for c in candidates:
        print(f'  hand: angle {c.angle_deg:6.1f}  length {c.length:.2f}R  '
              f'thickness {c.thickness_px:4.1f}px  score {c.score:5.0f}')
    return candidates
 