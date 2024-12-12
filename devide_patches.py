import os
import pydicom
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import cv2

SAM_DEVICE =  "cuda:0" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## BB 찾기

def load_sam_model(model_type="vit_h"):
    sam_checkpoint = "/home/juneha/data_compression/compression/models/sam_vit_h_4b8939.pth"
    device = torch.device(SAM_DEVICE)

    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, weights_only=True)
    except TypeError:
        print("Loading entire model as `weights_only=True` is not supported for this checkpoint.")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    sam.to(device=device)
    return sam

def perform_segmentation(sam_model, image):
    height, width = image.shape[:2]
    image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    mask_generator = SamAutomaticMaskGenerator(
        sam_model,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.9,
    )
    masks = mask_generator.generate(image)
    return masks

def filter_masks_by_tissue(masks, min_tissue_area=1):
    filtered_masks = []
    for mask in masks:
        tissue_area = np.sum(mask["segmentation"] > 0)
        if tissue_area >= min_tissue_area:
            filtered_masks.append(mask)
        else:
            print(f"Filtered out a mask with area {tissue_area}")
    return filtered_masks

def calculate_overall_bounding_box(masks):
    best_bounding_box = None
    max_dimension_sum = -1
    '''overall_x_min = float('inf')
    overall_y_min = float('inf')
    overall_x_max = float('-inf')
    overall_y_max = float('-inf')'''

    for mask in masks:
        tissue_pixels = np.where(mask["segmentation"] > 0)

        if len(tissue_pixels[0]) == 0 or len(tissue_pixels[1]) == 0:
            continue

        y_min, y_max = int(np.min(tissue_pixels[0])/2), int(np.max(tissue_pixels[0])/2)
        x_min, x_max = int(np.min(tissue_pixels[1])/2), int(np.max(tissue_pixels[1])/2)

        if (x_min, y_min) == (0, 0) and (x_max, y_max) == (255, 255):
            continue

        dimension_sum = (x_max - x_min) + (y_max - y_min)

        # Update the best bounding box if this one has a larger dimension sum
        if dimension_sum > max_dimension_sum:
            max_dimension_sum = dimension_sum
            best_bounding_box = (x_min, y_min, x_max, y_max)

    if best_bounding_box:
        overall_x_min, overall_y_min, overall_x_max, overall_y_max = best_bounding_box
        print(f"Overall Bounding Box Coordinates: x_min={overall_x_min}, y_min={overall_y_min}, x_max={overall_x_max}, y_max={overall_y_max}")
        return overall_x_min, overall_y_min, overall_x_max, overall_y_max
    else:
        print("No valid bounding box found.")
        return None

        '''overall_x_min = min(overall_x_min, x_min)
        overall_y_min = min(overall_y_min, y_min)
        overall_x_max = max(overall_x_max, x_max)
        overall_y_max = max(overall_y_max, y_max)

    if overall_x_min < overall_x_max and overall_y_min < overall_y_max:
        print(f"Overall Bounding Box Coordinates: x_min={overall_x_min}, y_min={overall_y_min}, x_max={overall_x_max}, y_max={overall_y_max}")
        return overall_x_min, overall_y_min, overall_x_max, overall_y_max
    else:
        print("No valid bounding box found.")
        return None'''

def divide_image_with_bb_in_center(image, bounding_box):
    """Divide the image into 9 patches, central patch determined by bounding box."""
    h, w, _ = image.shape
    x_min, y_min, x_max, y_max = bounding_box

    patches = []
    patches.append((image[0:y_min, 0:x_min], (0, 0, x_min, y_min)))
    patches.append((image[0:y_min, x_min:x_max], (x_min, 0, x_max, y_min)))
    patches.append((image[0:y_min, x_max:w], (x_max, 0, w, y_min)))
    patches.append((image[y_min:y_max, 0:x_min], (0, y_min, x_min, y_max)))
    patches.append((image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)))
    patches.append((image[y_min:y_max, x_max:w], (x_max, y_min, w, y_max)))
    patches.append((image[y_max:h, 0:x_min], (0, y_max, x_min, h)))
    patches.append((image[y_max:h, x_min:x_max], (x_min, y_max, x_max, h)))
    patches.append((image[y_max:h, x_max:w], (x_max, y_max, w, h)))

    return patches
