"""
Stage 1: Detection Module
Locates potential number plate candidates in video frames using
contour-based edge detection and geometric filtering.
"""
import cv2
import numpy as np
from utils import detect_edges, locate_contours, filter_by_geometry


def find_plate_candidates(frame):
    """
    Detect potential license plate regions in an image.
    Pipeline: Edge Detection -> Contour Finding -> Geometric Filtering
    """
    edges = detect_edges(frame)
    contours = locate_contours(edges)
    candidates = filter_by_geometry(contours)
    return candidates


def annotate_candidates(display_frame, candidate_list):
    """Draw bounding boxes around detected plate candidates."""
    for rect in candidate_list:
        box = cv2.boxPoints(rect).astype(int)
        cv2.polylines(display_frame, [box], True, (0, 255, 0), 2)


def main():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Failed to open camera device")

    print("Detection Stage - Press 'q' to exit")

    while True:
        success, frame = capture.read()
        if not success:
            break

        display = frame.copy()
        candidates = find_plate_candidates(frame)

        if candidates:
            annotate_candidates(display, candidates)
            status = "Plate detected"
            color = (0, 255, 0)
        else:
            status = "Scanning for plates..."
            color = (0, 200, 255)

        cv2.putText(
            display, status, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )
        cv2.putText(
            display, "Press 'q' to quit", (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow("Stage 1: Plate Detection", display)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
