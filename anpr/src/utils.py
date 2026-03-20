"""
ANPR Pipeline Utilities
Shared functions for the Automatic Number Plate Recognition pipeline.
"""
import cv2
import numpy as np


class PlateConfig:
    """Configuration constants for plate detection and processing."""
    MIN_CONTOUR_AREA = 600
    ASPECT_RATIO_MIN = 2.0
    ASPECT_RATIO_MAX = 8.0
    OUTPUT_WIDTH = 450
    OUTPUT_HEIGHT = 140
    CANNY_LOW = 100
    CANNY_HIGH = 200
    BLUR_KERNEL = (5, 5)


def preprocess_frame(image):
    """Convert image to grayscale and apply Gaussian blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, PlateConfig.BLUR_KERNEL, 0)
    return blurred


def detect_edges(image):
    """Apply Canny edge detection to preprocessed image."""
    blurred = preprocess_frame(image)
    edges = cv2.Canny(blurred, PlateConfig.CANNY_LOW, PlateConfig.CANNY_HIGH)
    return edges


def locate_contours(edge_map):
    """Find external contours from edge map."""
    contours, _ = cv2.findContours(
        edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def filter_by_geometry(contour_list):
    """
    Filter contours based on area and aspect ratio constraints.
    Returns list of minAreaRect tuples suitable for plates.
    """
    valid_candidates = []
    for contour in contour_list:
        area = cv2.contourArea(contour)
        if area < PlateConfig.MIN_CONTOUR_AREA:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), _ = rect

        if width <= 0 or height <= 0:
            continue

        aspect = max(width, height) / max(1.0, min(width, height))
        if PlateConfig.ASPECT_RATIO_MIN <= aspect <= PlateConfig.ASPECT_RATIO_MAX:
            valid_candidates.append(rect)

    return valid_candidates


def sort_points_clockwise(points_array):
    """
    Order 4 corner points in clockwise direction:
    top-left, top-right, bottom-right, bottom-left.
    """
    pts = np.array(points_array, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1)

    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmin(diffs)]
    bl = pts[np.argmax(diffs)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_perspective_transform(source_rect):
    """
    Calculate perspective transform matrix from source rectangle.
    Returns the transformation matrix and destination dimensions.
    """
    box = cv2.boxPoints(source_rect)
    src_pts = sort_points_clockwise(box)

    dst_pts = np.array([
        [0, 0],
        [PlateConfig.OUTPUT_WIDTH - 1, 0],
        [PlateConfig.OUTPUT_WIDTH - 1, PlateConfig.OUTPUT_HEIGHT - 1],
        [0, PlateConfig.OUTPUT_HEIGHT - 1]
    ], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    output_size = (PlateConfig.OUTPUT_WIDTH, PlateConfig.OUTPUT_HEIGHT)

    return transform_matrix, output_size


def extract_plate_region(frame, plate_rect):
    """
    Apply perspective warp to extract normalized plate image.
    Returns warped plate image of standard dimensions.
    """
    matrix, dimensions = compute_perspective_transform(plate_rect)
    warped = cv2.warpPerspective(frame, matrix, dimensions)
    return warped


def select_largest_candidate(candidates):
    """Choose the candidate with largest bounding box area."""
    if not candidates:
        return None
    return max(candidates, key=lambda r: r[1][0] * r[1][1])
