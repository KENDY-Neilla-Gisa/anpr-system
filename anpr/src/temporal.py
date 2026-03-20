"""
Stage 5 & 6: Temporal Validation and Save Module
Implements temporal consistency checking via majority voting across frames
and persists validated plates to CSV with cooldown to prevent duplicates.
"""
import cv2
import numpy as np
import pytesseract
import re
import csv
import os
import time
from collections import Counter
from utils import (
    detect_edges, locate_contours, filter_by_geometry,
    extract_plate_region, select_largest_candidate
)

TESSERACT_CONFIG = (
    '--psm 8 --oem 3 '
    '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
)

PLATE_PATTERN = re.compile(r'[A-Z]{3}[0-9]{3}[A-Z]')

TEMPORAL_BUFFER_SIZE = 5
SAVE_COOLDOWN_SECONDS = 10
LOG_FILE_PATH = "../data/plates.csv"


def initialize_storage():
    """Ensure log file exists with proper headers."""
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), LOG_FILE_PATH))
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["plate_number", "timestamp"])


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
    cleaned = raw_text.upper().replace(" ", "").replace("-", "")
    return cleaned


def validate_plate_format(text):
    """Check if text matches expected plate format: AAA000A"""
    match = PLATE_PATTERN.search(text)
    return match.group(0) if match else None


def majority_vote(plates_buffer):
    """
    Determine most frequent plate in temporal buffer.
    Returns plate with highest occurrence count.
    """
    if not plates_buffer:
        return None
    return Counter(plates_buffer).most_common(1)[0][0]


def save_plate_record(plate_number):
    """Append validated plate to CSV log with current timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), LOG_FILE_PATH))
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([plate_number, timestamp])
    print(f"[SAVED] {plate_number} at {timestamp}")


def main():
    initialize_storage()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Failed to open camera device")

    print("Temporal & Save Stage - Press 'q' to exit")

    temporal_buffer = []
    last_saved_plate = None
    last_save_timestamp = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        display = frame.copy()
        candidates = find_plate_candidates(frame)

        if candidates:
            best_rect = select_largest_candidate(candidates)
            if best_rect:
                box = cv2.boxPoints(best_rect).astype(int)
                cv2.polylines(display, [box], True, (0, 255, 0), 2)

                plate_img = extract_plate_region(frame, best_rect)
                raw_text = extract_text_from_plate(plate_img)
                valid_plate = validate_plate_format(raw_text)

                if valid_plate:
                    temporal_buffer.append(valid_plate)
                    if len(temporal_buffer) > TEMPORAL_BUFFER_SIZE:
                        temporal_buffer.pop(0)

                    confirmed_plate = majority_vote(temporal_buffer)

                    text_x = int(np.max(box[:, 0])) - 300
                    text_y = int(np.max(box[:, 1])) + 25
                    cv2.putText(
                        display, f"CONFIRMED: {confirmed_plate}",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )

                    current_time = time.time()
                    if (
                        confirmed_plate
                        and confirmed_plate != last_saved_plate
                        and (current_time - last_save_timestamp) > SAVE_COOLDOWN_SECONDS
                    ):
                        save_plate_record(confirmed_plate)
                        last_saved_plate = confirmed_plate
                        last_save_timestamp = current_time

        cv2.imshow("Stage 5 & 6: Temporal Validation & Save", display)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
