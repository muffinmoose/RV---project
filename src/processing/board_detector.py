# src/processing/board_detector.py
# Detects the 9HPT board bounding box in the first frame.
# Used to determine which hand is active (inside board area).
#
# Strategy:
#   1. Try automatic detection via edge + contour detection
#   2. Fallback to hardcoded BOARD_BBOX_PX from config if auto fails

import cv2
import numpy as np
from config import BOARD_BBOX_PX


def detect_board_bbox(frame: np.ndarray) -> tuple:
    """
    Detect the 9HPT board bounding box in the given frame.
    Returns (x1, y1, x2, y2) in pixels.

    Args:
        frame: undistorted BGR frame (first frame of video)
    Returns:
        (x1, y1, x2, y2) bounding box of the board
    """
    bbox = _auto_detect(frame)
    if bbox is not None:
        print(f"[BoardDetector] Auto-detected board bbox: {bbox}")
        return bbox

    print(f"[BoardDetector] Auto-detection failed — using fallback from config")
    return BOARD_BBOX_PX


def _auto_detect(frame: np.ndarray):
    """
    Try to detect the board automatically using edge + contour detection.
    The board is a large rectangular metallic surface.
    Returns (x1, y1, x2, y2) or None if detection fails.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to connect gaps
    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter contours by area — board should be large (>10% of frame)
    min_area = w * h * 0.10
    max_area = w * h * 0.90
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Approximate to polygon
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Board should be roughly rectangular (4-6 corners)
        if len(approx) < 4 or len(approx) > 8:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Board aspect ratio should be roughly 2:1 to 4:1 (wide rectangle)
        aspect = bw / max(bh, 1)
        if aspect < 1.5 or aspect > 5.0:
            continue

        candidates.append((area, x, y, x+bw, y+bh))

    if not candidates:
        return None

    # Pick largest candidate
    candidates.sort(reverse=True)
    _, x1, y1, x2, y2 = candidates[0]

    # Sanity check — bbox must be reasonable size
    if (x2-x1) < w*0.2 or (y2-y1) < h*0.1:
        return None

    return (x1, y1, x2, y2)