import os
import sys
import torch
import numpy as np
from six.moves import cPickle as pickle
from scipy.io import loadmat
import scipy
from tqdm import tqdm

from pycocotools.coco import COCO
import multiprocessing
from pre_tools import *


# --- PURE PYTORCH IOU IMPLEMENTATION ---
def mask_iou(masks_a, masks_b):
    """Calculates Intersection over Union natively in PyTorch on the GPU."""
    intersection = torch.logical_and(masks_a, masks_b).sum(dim=(1, 2))
    union = torch.logical_or(masks_a, masks_b).sum(dim=(1, 2))

    iou = torch.zeros_like(intersection, dtype=torch.float32)
    valid = union > 0
    iou[valid] = intersection[valid].float() / union[valid].float()

    return iou.unsqueeze(1)


def generate_cub_as_coco(imgIds, cocoGt, data_dir):
    base_path = "./data/CUB_200_2011/CUB_as_COCO/COB-COCO"

    # Check if GPU is available, otherwise fallback to CPU safely
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for index in tqdm(range(len(imgIds))):
        img_id = imgIds[index]
        path = cocoGt.loadImgs(img_id)[0]['file_name']

        base_name = os.path.basename(path)
        file_name = base_name.replace('.jpg', '.mat')
        pkl_name = base_name.replace('.jpg', '.pkl')

        if os.path.exists(os.path.join(data_dir, pkl_name)):
            continue

        try:
            COB_proposals = scipy.io.loadmat(os.path.join(base_path, file_name),
                                             verify_compressed_data_integrity=False)['maskmat']
        except Exception as e:
            print(f"FAILED on {file_name}: {e}")
            continue

        mask_proposals = [np.array(p) for p in COB_proposals]
        mask_proposals = np.array(mask_proposals)
        num_proposal = len(mask_proposals)

        # Move array to PyTorch GPU instead of CuPy
        mask_proposals_pt = torch.tensor(mask_proposals, device=device, dtype=torch.bool)

        iou_map = []
        for j in range(num_proposal):
            proposal_iou = mask_iou(mask_proposals_pt, mask_proposals_pt[j].unsqueeze(0))
            iou_map.append(proposal_iou)

        # Concatenate, cast to float16, move back to CPU, and convert back to standard NumPy
        iou_map = torch.cat(iou_map, dim=1).half().cpu().numpy()

        pickle.dump(iou_map, open(os.path.join(data_dir, pkl_name), 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset

    label_file_list = []

    if dataset == "cub_as_coco":
        label_file_list.append("./data/CUB_200_2011/CUB_as_COCO/annotations/instances_train2017.json")
        label_file_list.append("./data/CUB_200_2011/CUB_as_COCO/annotations/instances_val2017.json")
        data_dir = "./data/cob_iou/coco2017"
    else:
        # Just halting the others to keep the code perfectly clean for CUB
        raise NotImplementedError("This script has been optimized specifically for cub_as_coco using PyTorch.")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print(label_file_list)

    # Workers lowered to 4 for PyTorch GPU stability to prevent VRAM spikes
    worker = 4

    for label_file in label_file_list:
        print(label_file)
        cocoGt = COCO(label_file)
        imgIds = sorted(cocoGt.getImgIds())

        per_len = int(len(imgIds) / worker)

        jobs = []
        for work_id in range(worker):
            if work_id + 1 != worker:
                p = multiprocessing.Process(target=generate_cub_as_coco,
                                            args=(imgIds[work_id * per_len:(work_id + 1) * per_len],
                                                  cocoGt, data_dir))
            else:
                p = multiprocessing.Process(target=generate_cub_as_coco,
                                            args=(imgIds[work_id * per_len:],
                                                  cocoGt, data_dir))
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()