# import sys
# sys.path.insert(0, '../')
from coco_citypersons import COCO_citypersons
from eval_MR_multisetup import COCOeval_citypersons
import os


annType = 'bbox'      #specify type here
# print('Running demo for *%s* results.'%(annType))

res_dir = '../res/coco_citypersons_val_citypersons_1gpu_e2e_faster_rcnn_R-50-FPN_v1_16-171999_dt.json'
# res_dir = '../val_groundtruth_dt.json'
annFile = '../val_gt.json'

# running evaluation
res_file = open("results.txt", "w")
for id_setup in range(0, 8):
    cocoGt = COCO_citypersons(annFile)
    cocoDt = cocoGt.loadRes(res_dir)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval_citypersons(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate(id_setup)
    cocoEval.accumulate()
    cocoEval.summarize(id_setup, res_file)

res_file.close()


