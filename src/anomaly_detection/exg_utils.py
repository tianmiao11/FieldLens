import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def show_image(image_path: str):
    """Display image with matplotlib."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

def generate_exg(input_path: str, output_path: str = None):
    """Generate ExG (Excess Green Index) image and save."""
    if output_path is None:
        name, _ = os.path.splitext(input_path)
        output_path = f"{name}_exg.jpg"

    img_bgr = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    exg = 2 * G.astype(np.float32) - R.astype(np.float32) - B.astype(np.float32)
    exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-8)
    exg_img = (exg_norm * 255).astype(np.uint8)

    exg_color = cv2.applyColorMap(exg_img, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(output_path, exg_color)
    print(f"✅ ExG image saved to: {output_path}")