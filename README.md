# Tools

A collection of useful Python scripts for various image and video processing tasks.

## Available Scripts

### 1. `img2pdf.py`
Converts images (JPG, PNG, BMP, GIF) into PDF format.
- Automatically scans the `documents/source/` directory for images.
- Converts and saves the output PDFs to `documents/result/`.

### 2. `compress.py`
Compresses images to ensure they are under a specified file size (target in KB).
- Iteratively reduces JPEG quality until the target file size is reached.
- Converts images to RGB if they contain alpha channels or use paletted colors.

### 3. `rotate.py`
Rotates images by 90 degrees.
- Reads images from `result/frames/`.
- Saves rotated images to `result/rotated/`.

### 4. `extract_frames.py`
Extracts individual frames from a video file.
- Reads a video file (using OpenCV).
- Saves each frame as a JPEG image in a specified output directory.

## Setup

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install Pillow opencv-python
   ```

## Directory Structure
Some scripts rely on specific folder structures:
- `documents/source/` : Place source images here for PDF conversion.
- `documents/result/` : Generated PDFs will be saved here.
- `result/frames/` : Place images here for rotation.
- `result/rotated/` : Rotated images will be saved here.
