import os
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
IMG_DIR = './data/CUB_200_2011/CUB_as_COCO/val2017'  # Folder with your original JPGs
MAT_DIR = './data/CUB_200_2011/CUB_as_COCO/COB-COCO'  # Folder with your generated .mat files
OUT_DIR = './data/vis_current_proposals'  # Where to save the output images
NUM_IMAGES_TO_DRAW = 20  # How many images to test
MAX_BOXES_TO_DRAW = 150  # Limit boxes so it's not a solid red block
# ---------------------

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    image_paths = glob(os.path.join(IMG_DIR, '*.jpg'))
    image_paths = image_paths[:NUM_IMAGES_TO_DRAW]

    if not image_paths:
        print(f"No images found in {IMG_DIR}!")
        return

    print(f"Visualizing proposals for {len(image_paths)} images...")

    for img_path in tqdm(image_paths):
        img_name = os.path.basename(img_path)
        mat_name = img_name.replace('.jpg', '.mat')
        mat_path = os.path.join(MAT_DIR, mat_name)

        if not os.path.exists(mat_path):
            print(f"  -> Missing .mat file for {img_name}, skipping.")
            continue

        # 1. Load the original image
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Load the existing .mat file
        try:
            mat_data = scipy.io.loadmat(mat_path, verify_compressed_data_integrity=False)
            masks = mat_data['maskmat']

            # Safely unpack the array based on how GrabCut saved it
            if masks.dtype == object or len(masks.shape) == 1:
                masks = np.array([np.array(p) for p in masks])
            elif len(masks.shape) == 2:
                masks = np.expand_dims(masks, axis=0)
        except Exception as e:
            print(f"  -> Error loading {mat_name}: {e}")
            continue

        # 3. Extract bounding boxes from the masks
        boxes = []
        for i in range(len(masks)):
            mask = masks[i]
            y_indices, x_indices = np.where(mask > 0)

            if len(y_indices) > 0 and len(x_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)

                # Only keep valid boxes
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

        # 4. Draw the boxes
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        ax.axis('off')

        boxes_to_draw = boxes[:MAX_BOXES_TO_DRAW]
        for (x, y, w, h) in boxes_to_draw:
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=1.5,
                edgecolor='#00ff00',  # Bright green boxes
                facecolor='none',
                alpha=0.7
            )
            ax.add_patch(rect)

        plt.title(f"{img_name} - {len(boxes)} Total Proposals Generated", fontsize=14)
        plt.tight_layout()

        # Save visualization
        save_path = os.path.join(OUT_DIR, img_name)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f"\nDone! Check the '{OUT_DIR}' folder to see your current proposals.")


if __name__ == '__main__':
    main()