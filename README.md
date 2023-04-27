# handwritings-to-txt


## Requirements

* Python 3.x
* OpenCV
* pytesseract

## Installation

You can install the required libraries using pip:

```
pip install opencv-python
pip install pytesseract
```

## Usage

To use this code, simply import the function `scan_handwritten_text` and pass the path of the image as a parameter. The function will return the text extracted from the image as a string.

```python
import cv2
import numpy as np
import pytesseract

def scan_handwritten_text(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the image
    threshold = 100
    binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract text from each contour using OCR
    text = ''
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = binary[y:y+h, x:x+w]
        kernel = np.ones((3,3), np.uint8)
        roi = cv2.erode(roi, kernel, iterations=1)
        text += pytesseract.image_to_string(roi)

    return text

text = scan_handwritten_text('image.jpg')
print(text)
```

## Notes

* The accuracy of the OCR will depend on the quality of the input image and the legibility of the handwriting.
* Adjusting the threshold value may improve the OCR accuracy for some images.
