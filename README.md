# ANPR: Automatic Number Plate Recognition Pipeline

Real-time license plate detection and recognition system built with OpenCV and Tesseract OCR.

---

## System Workflow

```
Frame Input → Contour Detection → Perspective Warp → OCR → Regex Check → Temporal Filter → CSV Output
```

## Required Pipeline (Per Assignment)

| Step | File | Core Function |
|------|------|---------------|
| **1. Detection** | `detect.py` | Edge detection → Contour extraction → Geometry filtering |
| **2. Alignment** | `align.py` | Corner detection → Perspective transform → 450×140 normalization |
| **3. OCR** | `ocr.py` | Grayscale → Gaussian blur → Otsu threshold → Tesseract PSM-8 |
| **4. Validation** | `validate.py` | Pattern match `[A-Z]{3}[0-9]{3}[A-Z]` → Candidate scoring |
| **5. Temporal** | `temporal.py` | 5-frame buffer → Majority vote → Consistency check |
| **6. Save** | `temporal.py` | Deduplication (10s cooldown) → CSV append |

---

## Quick Start

```bash
# 1. Install system dependency (Tesseract)
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt install tesseract-ocr
# macOS: brew install tesseract

# 2. Setup Python environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 3. Execute pipeline stages
cd src
python camera.py      # Verify hardware
python detect.py      # Test detection
python align.py       # Test alignment
python ocr.py         # Test recognition
python validate.py    # Test validation
python temporal.py    # Full system with logging
```

---

## Technical Implementation

### Detection Strategy
```python
Preprocessing:  BGR → Gray → Gaussian(5,5) → Canny(100,200)
Filtering:      Area > 600px, Aspect ratio 2.0-8.0
Output:         List of minAreaRect candidates
```

### Alignment Math
```python
Input:          Rotated bounding box (4 corner points)
Operation:      getPerspectiveTransform() → warpPerspective()
Output:         450×140 normalized plate image
```

### OCR Configuration
```
Engine:         Tesseract LSTM
Page Seg:       PSM 8 (single word)
Whitelist:      A-Z, 0-9
Preprocessing:  Otsu thresholding
```

### Validation Logic
```python
Pattern:        [A-Z]{3}[0-9]{3}[A-Z]
Example:        AAA000A
Scoring:        (is_valid, alnum_count)
Throttle:       350ms between OCR runs
```

### Temporal Smoothing
```python
Buffer:         5 frames
Consensus:      Counter.most_common(1)
Cooldown:       10 seconds per unique plate
Storage:        data/plates.csv
```

---

## Directory Layout

```
.
├── README.md              # Documentation
├── requirements.txt       # Dependencies
├── data/
│   ├── plates.csv         # Detected plates log
│   └── logs/              # Additional logs
├── screenshots/           # Test evidence
│   ├── detection.png
│   ├── alignment.png
│   └── ocr.png
└── src/
    ├── camera.py          # Hardware check
    ├── detect.py          # Stage 1
    ├── align.py           # Stage 2
    ├── ocr.py             # Stage 3
    ├── validate.py        # Stage 4
    ├── temporal.py        # Stages 5-6
    └── utils.py           # Shared components
```

---

## Module Extras

### validate.py CLI Options
```bash
# Single image test
python validate.py --image ../screenshots/test.jpg

# With ROI selection
python validate.py --image ../screenshots/test.jpg --roi
```

---

## CSV Schema

**File**: `data/plates.csv`

| Field | Format | Example |
|-------|--------|---------|
| plate_number | [A-Z]{3}[0-9]{3}[A-Z] | ABC123D |
| timestamp | YYYY-MM-DD HH:MM:SS | 2025-03-20 14:30:00 |

---

## Notes

- Modular architecture: Common functions extracted to `utils.py`
- All stages independently testable
- Runs on CPU (no GPU required)
- Real-time processing at ~15-30 FPS (camera dependent)
