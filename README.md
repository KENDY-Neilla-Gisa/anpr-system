# Car Number Plate Extraction System

This project implements a Car Number Plate Extraction pipeline in three main steps: Detection, Alignment, and Optical Character Recognition (OCR). It was developed based on the guiding principles from "Car Number Plate Extraction in Three Steps".

## Project Structure
```
.
├── data/
│   ├── logs/             # Contains the CSV output logs of detected plates
│   └── plates/           # Contains plate images
├── src/
│   ├── camera.py         # Validates camera functionality
│   ├── detect.py         # Plate candidate detection using contours
│   ├── align.py          # Plate rectification and normalization via perspective warp
│   ├── ocr.py            # Optical character recognition using Tesseract
│   ├── validate.py       # Regex validation against formatting norms
│   └── temporal.py       # Full live pipeline combining all stages and logging to CSV
├── README.md             # This documentation
```

## Features
- **Plate Detection**: Uses contour geometry (size and aspect ratio constraints) for lightweight CPU-oriented detection.
- **Plate Alignment**: Employs perspective transformation to warp the detected bounding box into a strictly uniform 450x140 resolution image.
- **OCR via Tesseract**: Processes the isolated and aligned plate image to extract alphanumeric text.
- **Regex Validation**: Validates the extracted candidates matching the layout `[A-Z]{3}[0-9]{3}[A-Z]`.
- **Temporal Consistency**: Takes sequential readings across video frames and verifies the majority vote to avoid single-frame hallucination.
- **CSV Logging**: Validated strings are logged with a timestamp and cooldown restrictions to avoid duplication.

## Setup Requirements

It's heavily recommended to use a Python virtual environment to avoid versioning conflicts. Ensure that `tesseract-ocr` is installed on your operating system.

```bash
# Set up a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the Python dependencies
python -m pip install --upgrade pip
pip install opencv-python numpy pytesseract pandas
```

### OS Dependencies
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

## Usage Guide
Navigate to the `src` directory or run the modules directly. They logically progress sequentially:

1. **Test Camera**
   ```bash
   python src/camera.py
   ```
2. **Detect Plates**
   ```bash
   python src/detect.py
   ```
3. **Align Plates**
   ```bash
   python src/align.py
   ```
4. **Extract Text (OCR)**
   ```bash
   python src/ocr.py
   ```
5. **Validate OCR Strings**
   ```bash
   python src/validate.py
   ```
6. **Full Sequence** (The final working system logging to CSV)
   ```bash
   python src/temporal.py
   ```

Press `q` to exit the graphical windows.
"# anpr-system" 
