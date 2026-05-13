# test_ocr.py
import sys
print(f"Python: {sys.executable}")

try:
    import pytesseract
    print(f"✓ pytesseract: {pytesseract.__version__}")
except ImportError as e:
    print(f"✗ pytesseract: {e}")

try:
    from PIL import Image
    print("✓ Pillow: installed")
except ImportError as e:
    print(f"✗ Pillow: {e}")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV: {e}")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")