# import sys
# sys.path.insert(0, '../')
from coco_citypersons import COCO_citypersons
from eval_MR_multisetup import COCOeval_citypersons
import os
import json

classes = [
    'Average Miss Rate  (MR) @ Reasonable         [ IoU=0.50',
    'Average Miss Rate  (MR) @ Reasonable_small',
    'Average Miss Rate  (MR) @ Reasonable_occ=heavy(RO)',
    'Average Miss Rate  (MR) @ RO+H ',
    'Average Miss Rate  (MR) @ All                [ IoU=0.50',
    'Average Miss Rate  (MR) @ All-50',
    'Average Miss Rate  (MR) @ R_occ=None',
    'Average Miss Rate  (MR) @ R_occ=Partial',
    'Average Miss Rate  (MR) @ R_occ=heaavy ',
    'Average Miss Rate  (MR) @ Heavy',
    'Average Miss Rate  (MR) @ Partial',
    'Average Miss Rate  (MR) @ Bare',
]

annType = 'bbox'      #specify type here
# print('Running demo for *%s* results.'%(annType))
# iterations = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
iterations = [30000]
version = 'v50_10'

# iteration = 30000

root = "/media/tianliang/DATA/DataSets/Cityscapes/shanshanzhang-citypersons/res/val"
cfg_file = "citypersons_free_anchor_R-50-FPN_1gpu_1x"

# res_dir = os.path.join(root, cfg_file, val_dataset, str(iteration), 'bbox.json')
res_dir = os.path.join(root, cfg_file, "bbox.json")
output_dir = os.path.join(root, cfg_file, "bbox_new.json")
out = []
detections = json.load(open(res_dir, 'r'))
for det in detections:
    det['image_id'] = det['image_id'] + 1
    out.append(det)

with open(output_dir, "w") as f:
    json.dump(out, f)

print("done!")