"""
transform_class.py
---------------
Defines a FrameTransformer class with an 'apply' method
to flip, translate, rotate, and zoom images.
"""

import cv2
import numpy as np

class FrameTransformer:
    def __init__(self):
        pass

    def apply(self, frame, flip=None, translate=(0, 0), rotate=0.0, zoom=1.0):
        rows, cols = frame.shape[:2]
        out = frame.copy()

        # Flip
        if flip:
            flip_code = {'h': 1, 'v': 0, 'both': -1}[flip]
            out = cv2.flip(out, flip_code)

        # Translate
        tx, ty = translate
        if tx or ty:
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            out = cv2.warpAffine(out, M, (cols, rows))

        # Rotate
        if rotate:
            M = cv2.getRotationMatrix2D((cols/2, rows/2), rotate, 1.0)
            out = cv2.warpAffine(out, M, (cols, rows))

        # Zoom
        if zoom != 1.0:
            if zoom > 1.0:
                new_w, new_h = int(cols/zoom), int(rows/zoom)
                x1, y1 = (cols-new_w)//2, (rows-new_h)//2
                out = cv2.resize(out[y1:y1+new_h, x1:x1+new_w], (cols, rows))
            elif zoom > 0:
                new_w, new_h = int(cols*zoom), int(rows*zoom)
                small = cv2.resize(out, (new_w, new_h))
                canvas = np.zeros_like(out)
                x1, y1 = (cols-new_w)//2, (rows-new_h)//2
                canvas[y1:y1+new_h, x1:x1+new_w] = small
                out = canvas
        return out

if __name__ == "__main__":
    transformer = FrameTransformer()
    img = cv2.imread("F:\cat_images\cat1.jpg")
    if img is None:
        raise FileNotFoundError("Image not found")

    result = transformer.apply(img, flip='h', rotate=90, zoom=0.8)
    cv2.imshow("Transformed Image (class)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
