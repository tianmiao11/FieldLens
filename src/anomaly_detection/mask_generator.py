import cv2
import numpy as np

def generate_mask(
    input_path: str,
    output_path: str,
    threshold_value: int = 60,
    min_area: int = 2000,
    kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
):
    """Generate sparse-area mask from ExG image."""
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ Failed to load image: {input_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    mask = np.zeros_like(gray)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255

    mask[:int(gray.shape[0] / 2), :] = 0

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(output_path, mask_bgr)
    print(f"✅ Saved mask: {output_path}")