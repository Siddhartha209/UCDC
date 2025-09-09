## Installation

Make sure you have Python installed. Then, install the required dependencies:

```bash
# Navigate to the project directory
cd UCDC

# Install Python dependencies
pip install pytesseract easyocr ultralytics numpy
```

You also need to install Tesseract OCR on your system:

- **Windows:** Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
- **macOS:** Install via Homebrew:
    ```bash
    brew install tesseract
    ```
- **Linux:** Install via apt:
    ```bash
    sudo apt-get install tesseract-ocr
    ```
## NOTE: 
Two models missing due to large file size

## Project Structure
Models - YOLOv11 trained object detection/ instance segmentation models
TestCases - Images of various tests 
Output - Corresponding outputs to TestCases images
main.py - Main file to run project
