import cv2
import os

def draw_boxes_from_mask(original_path: str, mask_path: str, output_path: str):
    """Draw red bounding boxes based on the binary mask."""
    if not os.path.exists(original_path):
        print(f"❌ Original image not found: {original_path}")
        return
    if not os.path.exists(mask_path):
        print(f"❌ Mask image not found: {mask_path}")
        return

    original_img = cv2.imread(original_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxed_img = original_img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(boxed_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imwrite(output_path, boxed_img)
    print(f"✅ Saved boxed image: {output_path}")