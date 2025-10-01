import numpy as np
from PIL import Image

def bilinear_resize(image, new_width, new_height):
    # Convert to numpy array
    src = np.array(image)
    h, w, c = src.shape
    dst = np.zeros((new_height, new_width, c), dtype=np.uint8)

    x_ratio = (w - 1) / (new_width - 1) if new_width > 1 else 0
    y_ratio = (h - 1) / (new_height - 1) if new_height > 1 else 0

    for i in range(new_height):
        for j in range(new_width):
            x_l, y_l = int(x_ratio * j), int(y_ratio * i)
            x_h, y_h = min(x_l + 1, w - 1), min(y_l + 1, h - 1)

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_l

            a = src[y_l, x_l]
            b = src[y_l, x_h]
            c_ = src[y_h, x_l]
            d = src[y_h, x_h]

            pixel = (a * (1 - x_weight) * (1 - y_weight) +
                     b * x_weight * (1 - y_weight) +
                     c_ * (y_weight) * (1 - x_weight) +
                     d * (x_weight * y_weight))

            dst[i, j] = pixel.astype(np.uint8)

    return Image.fromarray(dst)


if __name__ == "__main__":
    # Load a 100x100 image
    img = Image.open(r"F:\cat_images\big-cat1-resized.jpg").convert("RGB")

    # Resize to 199x199 using bilinear interpolation
    resized_img = bilinear_resize(img, 199, 199)
    resized_img.save("output_199x199.jpg")
    resized_img.show()