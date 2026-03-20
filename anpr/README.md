# Automatic Number Plate Recognition (ANPR) System

This repository implements a complete ANPR pipeline that processes video frames to detect, recognize, and validate vehicle license plates.

## Pipeline Architecture

The system follows a 6-stage sequential pipeline:

```
Detection → Alignment → OCR → Validation → Temporal → Save
```

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `detect.py` | Locates plate candidates using edge detection and contour filtering |
| 2 | `align.py` | Normalizes perspective using geometric transformation |
| 3 | `ocr.py` | Extracts text via Tesseract OCR |
| 4 | `validate.py` | Validates format against regex pattern `[A-Z]{3}[0-9]{3}[A-Z]` |
| 5 | `temporal.py` | Applies majority voting across frames for consistency |
| 6 | `temporal.py` | Persists validated plates to CSV with deduplication |

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── camera.py       # Camera verification utility
│   ├── detect.py       # Stage 1: Plate detection
│   ├── align.py       # Stage 2: Perspective normalization
│   ├── ocr.py         # Stage 3: Text extraction
│   ├── validate.py    # Stage 4: Format validation
│   ├── temporal.py    # Stages 5-6: Temporal consistency & persistence
│   └── utils.py       # Shared pipeline utilities
├── data/
│   └── plates.csv     # Validated plate storage
└── screenshots/       # Test evidence (populate with real captures)
```

## Prerequisites

Install Tesseract OCR on your system:

- **Windows**: Download installer from [tesseract-ocr GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run individual pipeline stages for testing:

```bash
# Stage 1: Verify camera
cd src && python camera.py

# Stage 2: Test detection
python detect.py

# Stage 3: Test alignment
python align.py

# Stage 4: Test OCR
python ocr.py

# Stage 5: Test validation
python validate.py

# Stage 6: Full pipeline (temporal + save)
python temporal.py
```

## Pipeline Details

### Stage 1: Detection
- Converts frame to grayscale
- Applies Gaussian blur and Canny edge detection
- Finds contours and filters by area (min 600px) and aspect ratio (2.0-8.0)

### Stage 2: Alignment
- Selects largest candidate region
- Computes perspective transform from corner points
- Warps to standardized 450x140 resolution

### Stage 3: OCR
- Preprocesses with Otsu thresholding
- Runs Tesseract with PSM 8 (single word) mode
- Whitelists A-Z and 0-9 characters

### Stage 4: Validation
- Validates against format `[A-Z]{3}[0-9]{3}[A-Z]`
- Scores candidates by validity and alphanumeric density
- Throttles OCR to 0.35s intervals for performance

### Stages 5-6: Temporal & Save
- Maintains rolling buffer of last 5 detections
- Applies majority voting for consistency
- Saves to CSV with 10-second cooldown per unique plate

## Testing with Images

The validation module supports single-image mode:

```bash
python validate.py --image path/to/plate.jpg
```

With optional ROI selection:

```bash
python validate.py --image path/to/plate.jpg --roi
```

## Data Storage

Detected plates are logged to `data/logs/plates_log.csv` with format:

```csv
plate_number,timestamp
AAA000A,2025-01-15 14:30:00
```

## Screenshots

Place real-world test captures in the `screenshots/` directory. These should show:
- Live detection annotations
- OCR output overlays
- Validated plate confirmations

---

**Note**: This implementation uses a modular utility architecture to share common preprocessing functions across all pipeline stages.
