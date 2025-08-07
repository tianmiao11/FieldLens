from anomaly_detection.exg_utils import generate_exg, show_image
from anomaly_detection.mask_generator import generate_mask
from anomaly_detection.box_drawer import draw_boxes_from_mask

if __name__ == "__main__":
    input_image = "input/original.jpg"
    exg_image = "output/exg.jpg"
    mask_image = "output/mask.jpg"
    box_image = "output/box.jpg"

    generate_exg(input_image, exg_image)
    generate_mask(exg_image, mask_image)
    draw_boxes_from_mask(input_image, mask_image, box_image)

    show_image(box_image)