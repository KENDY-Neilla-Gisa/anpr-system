"""
Stage 2: Alignment Module
Rectifies and normalizes detected plate regions using perspective transformation
to produce a standardized front-facing plate image.
"""
import cv2
import numpy as np
from utils import (
    detect_edges, locate_contours, filter_by_geometry,
    extract_plate_region, select_largest_candidate
)


def find_plate_candidates(frame):
    """Locate potential plate regions using edge detection."""
    edges = detect_edges(frame)
    contours = locate_contours(edges)
    candidates = filter_by_geometry(contours)
    return candidates


def align_and_normalize(frame, plate_rect):
    """
    Apply perspective warp to normalize plate orientation.
    Returns standardized plate image.
    """
    normalized_plate = extract_plate_region(frame, plate_rect)
    return normalized_plate


def main():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Failed to open camera device")

    print("Alignment Stage - Press 'q' to exit")

    while True:
        success, frame = capture.read()
        if not success:
            break

        display = frame.copy()
        candidates = find_plate_candidates(frame)
        aligned_plate = None

        if candidates:
            best_rect = select_largest_candidate(candidates)
            if best_rect:
                box = cv2.boxPoints(best_rect).astype(int)
                cv2.polylines(display, [box], True, (255, 0, 0), 2)

                for (px, py) in box:
                    cv2.circle(display, (px, py), 5, (0, 0, 255), -1)

                aligned_plate = align_and_normalize(frame, best_rect)
                status = "Plate aligned"
                color = (0, 255, 0)
            else:
                status = "Processing..."
                color = (0, 200, 255)
        else:
            status = "Detecting plate..."
            color = (0, 200, 255)

        cv2.putText(
            display, status, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )
        cv2.putText(
            display, "Press 'q' to quit", (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow("Stage 2: Plate Alignment", display)
        if aligned_plate is not None:
            cv2.imshow("Normalized Plate", aligned_plate)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
