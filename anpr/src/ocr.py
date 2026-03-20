"""
Stage 3: OCR Module
Extracts alphanumeric text from normalized plate images using Tesseract OCR.
"""
import cv2
import numpy as np
import pytesseract
from utils import (
    detect_edges, locate_contours, filter_by_geometry,
    extract_plate_region, select_largest_candidate
)

TESSERACT_CONFIG = (
    '--psm 8 --oem 3 '
    '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
)


def find_plate_candidates(frame):
    """Locate potential plate regions using edge detection."""
    edges = detect_edges(frame)
    contours = locate_contours(edges)
    candidates = filter_by_geometry(contours)
    return candidates


def preprocess_for_ocr(plate_img):
    """
    Prepare plate image for OCR by converting to grayscale,
    applying blur, and Otsu thresholding.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


def extract_text_from_plate(plate_img):
    """
    Perform OCR on plate image.
    Returns cleaned text string and thresholded image.
    """
    processed = preprocess_for_ocr(plate_img)
    raw_text = pytesseract.image_to_string(processed, config=TESSERACT_CONFIG)
    cleaned_text = raw_text.strip().replace(" ", "")
    return cleaned_text, processed


def annotate_text_on_frame(display, text_content, bounding_box):
    """Draw extracted text near the detected plate region."""
    if text_content:
        x_pos = int(np.max(bounding_box[:, 0]))
        y_pos = int(np.max(bounding_box[:, 1])) + 25
        x_pos = min(x_pos, display.shape[1] - 200)
        y_pos = min(y_pos, display.shape[0] - 10)

        cv2.putText(
            display, text_content, (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )


def main():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Failed to open camera device")

    print("OCR Stage - Press 'q' to exit")

    while True:
        success, frame = capture.read()
        if not success:
            break

        display = frame.copy()
        candidates = find_plate_candidates(frame)
        plate_img = None
        thresholded = None

        if candidates:
            best_rect = select_largest_candidate(candidates)
            if best_rect:
                box = cv2.boxPoints(best_rect).astype(int)
                cv2.polylines(display, [box], True, (0, 255, 0), 2)

                plate_img = extract_plate_region(frame, best_rect)
                extracted_text, thresholded = extract_text_from_plate(plate_img)

                annotate_text_on_frame(display, extracted_text, box)
                status = "OCR active"
                color = (0, 255, 0)
            else:
                status = "Processing..."
                color = (0, 200, 255)
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

        cv2.imshow("Stage 3: Optical Character Recognition", display)

        if plate_img is not None:
            cv2.imshow("Extracted Plate", plate_img)
        if thresholded is not None:
            cv2.imshow("OCR Preprocessing", thresholded)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
