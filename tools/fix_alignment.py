import os
import numpy as np
from six.moves import cPickle as pickle
from pycocotools.coco import COCO


def align_pkl(json_file, pkl_file):
    print(f"Aligning {pkl_file}...")
    coco = COCO(json_file)
    # Get IDs in the exact order the framework reads them (no sorting!)
    img_ids = coco.getImgIds()

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Create a mapping so we can pull the right mask for the right image
    id_to_idx = {img_id: idx for idx, img_id in enumerate(data['indexes'])}

    new_data = {'indexes': [], 'masks': [], 'boxes': [], 'scores': []}
    missing_count = 0

    for img_id in img_ids:
        if img_id in id_to_idx:
            idx = id_to_idx[img_id]
            new_data['indexes'].append(data['indexes'][idx])
            new_data['masks'].append(data['masks'][idx])
            new_data['boxes'].append(data['boxes'][idx])
            new_data['scores'].append(data['scores'][idx])
        else:
            # If an image was skipped during prep, insert safe empty arrays so it doesn't crash
            missing_count += 1
            new_data['indexes'].append(img_id)
            new_data['masks'].append(np.empty((0, 7, 7), dtype=bool))
            new_data['boxes'].append(np.empty((0, 4), dtype=np.uint16))
            new_data['scores'].append(np.zeros(0))

    with open(pkl_file, 'wb') as f:
        pickle.dump(new_data, f, pickle.HIGHEST_PROTOCOL)
    print(f"Done! Fixed {missing_count} misaligned/missing entries.\n")


if __name__ == '__main__':
    # Align the train file
    align_pkl('./data/CUB_200_2011/CUB_as_COCO/annotations/instances_train2017.json', './data/cob/coco_2017_train.pkl')

    # Align the val file
    align_pkl('./data/CUB_200_2011/CUB_as_COCO/annotations/instances_val2017.json', './data/cob/coco_2017_val.pkl')