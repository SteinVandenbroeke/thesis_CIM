import os
import numpy as np
from six.moves import cPickle as pickle
from scipy.io import loadmat
import scipy
from tqdm import tqdm
from pycocotools.coco import COCO
import multiprocessing
import cv2  # Safe replacement for deprecated imresize

trash = "./data/trash"
os.makedirs(trash, exist_ok=True)
os.makedirs("./data/cob", exist_ok=True)  # Ensure output directory exists


def generate_pkl_coco2017(imgIds, worker_id, cocoGt):
    COB_dir = './data/CUB_200_2011/CUB_as_COCO/COB-COCO'
    SBD_mask_val_proposals = dict(indexes=[], masks=[], boxes=[], scores=[])
    for index in tqdm(range(len(imgIds))):
        img_id = imgIds[index]

        path = cocoGt.loadImgs(img_id)[0]['file_name']

        # Check standard name first (matching your GrabCut script)
        file_name = path[:-4] + '.mat'
        if not os.path.exists(os.path.join(COB_dir, file_name)):
            file_name = 'COCO_train2014_' + path[:-4] + '.mat'
        if not os.path.exists(os.path.join(COB_dir, file_name)):
            file_name = 'COCO_val2014_' + path[:-4] + '.mat'

        try:
            COB_proposals = scipy.io.loadmat(
                os.path.join(COB_dir, file_name),
                verify_compressed_data_integrity=False)['maskmat']
        except Exception as e:
            print(f"Skipping {file_name} due to error: {e}")
            continue

        mask_proposals = COB_proposals.copy()

        boxes = np.empty((0, 4), dtype=np.uint16)
        masks = np.empty((0, mask_size, mask_size), dtype=bool)  # CHANGED from np.bool
        scores = np.zeros(len(COB_proposals))

        for pro_ind in range(len(mask_proposals)):
            ind_xy = np.nonzero(mask_proposals[pro_ind])

            # Safety check for empty masks
            if len(ind_xy[0]) == 0 or len(ind_xy[1]) == 0:
                continue

            xmin, ymin, xmax, ymax = ind_xy[1].min(), ind_xy[0].min(), ind_xy[1].max() + 1, ind_xy[0].max() + 1
            mask = mask_proposals[pro_ind][ymin:ymax, xmin:xmax]

            # CHANGED: Safely resize using OpenCV instead of deprecated imresize
            mask = cv2.resize(mask.astype(np.uint8), (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)

            boxes = np.append(boxes, np.array([[xmin, ymin, xmax, ymax]], dtype=np.uint16), axis=0)
            masks = np.append(masks, mask[np.newaxis, :].astype(bool), axis=0)

        SBD_mask_val_proposals['indexes'].append(img_id)
        SBD_mask_val_proposals['masks'].append(masks)
        SBD_mask_val_proposals['boxes'].append(boxes)
        SBD_mask_val_proposals['scores'].append(scores)

    pickle.dump(SBD_mask_val_proposals, open(os.path.join(trash, "coco_{}.pkl".format(worker_id)), 'wb'),
                pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    mask_size = 7
    worker = 8  # Lowered from 24 to prevent memory crashes

    # data - We only do range(2) for CUB (Train and Val)
    for i in range(2):
        if i == 0:
            # train:
            label_file = "./data/CUB_200_2011/CUB_as_COCO/annotations/instances_train2017.json"
            cob_proposal_file = "./data/cob/coco_2017_train.pkl"
        elif i == 1:
            # val:
            label_file = "./data/CUB_200_2011/CUB_as_COCO/annotations/instances_val2017.json"
            cob_proposal_file = "./data/cob/coco_2017_val.pkl"

        cocoGt = COCO(label_file)
        imgIds = sorted(cocoGt.getImgIds())

        per_len = int(len(imgIds) / worker)

        print(f"Processing: {label_file}")
        print(f"Outputting to: {cob_proposal_file}")

        jobs = []
        for worker_id in range(worker):
            if worker_id + 1 != worker:
                p = multiprocessing.Process(target=generate_pkl_coco2017,
                                            args=(imgIds[worker_id * per_len:(worker_id + 1) * per_len], worker_id,
                                                  cocoGt))
            else:
                p = multiprocessing.Process(target=generate_pkl_coco2017,
                                            args=(imgIds[worker_id * per_len:], worker_id, cocoGt))
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

        res = dict(indexes=[], masks=[], boxes=[], scores=[])
        worker_id = 0
        while (worker_id != worker):
            path = os.path.join(trash, "coco_{}.pkl".format(worker_id))
            try:
                result = pickle.load(open(path, 'rb'))
                res['indexes'] += result['indexes']
                res['masks'] += result['masks']
                res['boxes'] += result['boxes']
                res['scores'] += result['scores']
                os.remove(path)
                worker_id += 1
            except:
                pass

        print("imgs len: " + str(len(imgIds)))
        print("indexes len: " + str(len(res["indexes"])))
        print("masks len: " + str(len(res["masks"])))
        print("boxes len: " + str(len(res["boxes"])))
        print("scores len: " + str(len(res["scores"])))

        pickle.dump(res, open(cob_proposal_file, 'wb'), pickle.HIGHEST_PROTOCOL)