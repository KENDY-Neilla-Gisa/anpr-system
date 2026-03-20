"""
Stage 4: Validation Module
Validates extracted OCR text against number plate format patterns
and filters out invalid or noisy recognition results.
"""
import cv2
import numpy as np
import pytesseract
import re
import argparse
import time
from utils import (
    detect_edges, locate_contours, filter_by_geometry,
    extract_plate_region, select_largest_candidate
)

TESSERACT_CONFIG = (
    '--psm 8 --oem 3 '
    '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
)

PLATE_PATTERN = re.compile(r'[A-Z]{3}[0-9]{3}[A-Z]')
ALPHANUM_PATTERN = re.compile(r'[A-Z0-9]')
OCR_THROTTLE_SECONDS = 0.35


def find_plate_candidates(frame):
    """Locate potential plate regions using edge detection."""
    edges = detect_edges(frame)
    contours = locate_contours(edges)
    candidates = filter_by_geometry(contours)
    return candidates


def preprocess_for_ocr(plate_img):
    """Prepare plate image for OCR processing."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


def extract_text_from_plate(plate_img):
    """Perform OCR on plate image."""
    processed = preprocess_for_ocr(plate_img)
    raw_text = pytesseract.image_to_string(processed, config=TESSERACT_CONFIG)
    cleaned = raw_text.strip().replace(" ", "")
    return cleaned, processed


def validate_plate_format(text):
    """
    Check if text matches expected plate format: AAA000A
    Returns valid plate string or None.
    """
    text_upper = text.upper().replace(" ", "")
    match = PLATE_PATTERN.search(text_upper)
    return match.group(0) if match else None


def compute_candidate_score(valid_plate, alphanumeric_count):
    """Score candidate based on validity and alphanumeric content."""
    return (1 if valid_plate else 0, alphanumeric_count)


def select_best_candidate(candidates, input_frame):
    """
    Evaluate multiple candidates and return the best match.
    Returns tuple of (rect, plate_img, raw_text, processed_img, valid_plate).
    """
    best_result = None
    best_score = None

    sorted_candidates = sorted(
        candidates, key=lambda r: r[1][0] * r[1][1], reverse=True
    )[:6]

    for rect in sorted_candidates:
        candidate_plate = extract_plate_region(input_frame, rect)
        candidate_text, processed_img = extract_text_from_plate(candidate_plate)
        valid = validate_plate_format(candidate_text)

        cleaned = candidate_text.upper().replace(" ", "").replace("-", "")
        alnum_count = len(ALPHANUM_PATTERN.findall(cleaned))
        score = compute_candidate_score(valid, alnum_count)

        if best_result is None or score > best_score:
            best_result = (rect, candidate_plate, candidate_text, processed_img, valid)
            best_score = score

    return best_result


def main():
    parser = argparse.ArgumentParser(description="Plate Validation Stage")
    parser.add_argument(
        "--image", dest="image_path",
        help="Run validation on a single image file"
    )
    parser.add_argument(
        "--roi", action="store_true",
        help="Select region of interest for plate search"
    )
    args = parser.parse_args()

    video_source = None
    static_frame = None

    if args.image_path:
        static_frame = cv2.imread(args.image_path)
        if static_frame is None:
            raise FileNotFoundError(f"Cannot load image: {args.image_path}")
    else:
        video_source = cv2.VideoCapture(0)
        if not video_source.isOpened():
            raise RuntimeError("Failed to open camera device")

    roi_rectangle = None
    if args.roi and static_frame is not None:
        region = cv2.selectROI("Select ROI", static_frame, fromCenter=False)
        cv2.destroyWindow("Select ROI")
        rx, ry, rw, rh = [int(v) for v in region]
        if rw > 0 and rh > 0:
            roi_rectangle = (rx, ry, rw, rh)

    if not hasattr(main, "last_ocr_time"):
        main.last_ocr_time = 0.0
        main.cached_result = None

    print("Validation Stage - Press 'q' to exit")

    while True:
        if static_frame is not None:
            frame = static_frame.copy()
            success = True
        else:
            success, frame = video_source.read()

        if not success:
            break

        if args.roi and roi_rectangle is None and static_frame is None:
            region = cv2.selectROI("Select ROI", frame, fromCenter=False)
            cv2.destroyWindow("Select ROI")
            rx, ry, rw, rh = [int(v) for v in region]
            if rw > 0 and rh > 0:
                roi_rectangle = (rx, ry, rw, rh)

        display = frame.copy()
        offset_x, offset_y = 0, 0
        search_frame = frame

        if roi_rectangle is not None:
            rx, ry, rw, rh = roi_rectangle
            offset_x, offset_y = rx, ry
            search_frame = frame[ry:ry + rh, rx:rx + rw]
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)

        candidates = find_plate_candidates(search_frame)
        status = "Scanning for plates..."
        color = (0, 200, 255)
        plate_img = None
        thresholded = None

        current_time = time.time()
        should_run_ocr = (current_time - main.last_ocr_time) >= OCR_THROTTLE_SECONDS

        if candidates:
            if should_run_ocr:
                best = select_best_candidate(candidates, search_frame)
                main.cached_result = best
                main.last_ocr_time = current_time
            else:
                best = main.cached_result

            if best:
                rect, plate_img, raw_text, thresholded, valid_plate = best
                box = cv2.boxPoints(rect).astype(int)

                if roi_rectangle is not None:
                    box[:, 0] += offset_x
                    box[:, 1] += offset_y

                cv2.polylines(display, [box], True, (0, 255, 0), 2)
                status = "Validating OCR"
                color = (0, 255, 0)

                text_x = int(np.max(box[:, 0])) - 300
                text_y = int(np.max(box[:, 1])) + 25
                text_x = min(text_x, display.shape[1] - 200)
                text_y = min(text_y, display.shape[0] - 10)

                if valid_plate:
                    cv2.putText(
                        display, f"VALID: {valid_plate}", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )
                elif raw_text:
                    cv2.putText(
                        display, f"OCR: {raw_text}", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                    )

        cv2.putText(
            display, status, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )
        cv2.putText(
            display, "Press 'q' to quit", (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow("Stage 4: Validation", display)

        if plate_img is not None:
            cv2.imshow("Extracted Plate", plate_img)
        if thresholded is not None:
            cv2.imshow("Validation Threshold", thresholded)

        if static_frame is not None:
            cv2.waitKey(0)
            break

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    if video_source is not None:
        video_source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
