# Battery-Chemical-Classification-for-Line-Scan-X-ray-Machine

 real-time computer vision system designed for industrial line scan camera applications that detects and classifies battery chemicals using Halcon and deep learning.

 
## Prerequisites

- Python 3.7+

- MVTec Halcon library with valid license

- Line scan camera or sample images

## Installation & Usage

- main.py (this code file)

- Batteryclassifier.hdl (trained model)

## Code Structure 

1. Load image → Threshold detection → Shape selection
2. Count detected objects → Process each battery individually
3. Crop battery region → Preprocess for DL model
4. Run classification → Output chemical type and depth

# Key Configuration Parameters

```python
MaxImagesRegions = 2      # Frames processed simultaneously
ImageWidth = 1536         # Line scan camera width
ImageHeight = 547         # Frame height per acquisition

# Detection parameters (adjust based on your battery sizes):
min_area = 1000, max_area = 30000
min_width = 20, max_width = 760
```


