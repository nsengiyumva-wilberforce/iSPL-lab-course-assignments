"""
show_image.py
---------------
Loads an image from the computer and displays it using OpenCV.
Press any key to close the window.
"""

import cv2

# Replace with your image file path
IMAGE_PATH = "F:\cat_images\cat1.jpg"

def show_image(path):
    img = cv2.imread(path)

    # make the image bigger
    img = cv2.resize(img, (500, 400))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    cv2.imshow("Showing Cat Image", img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_image(IMAGE_PATH)
