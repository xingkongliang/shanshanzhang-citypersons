# import sys
# sys.path.insert(0, '../')
from coco_citypersons import COCO_citypersons
from eval_MR_multisetup import COCOeval_citypersons
import os


annType = 'bbox'      #specify type here
# print('Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
# annFile = 'E:/cityscapes/CityPersons_annos/test_gt.json'
# res_dir = '../../res/citypersons_1gpu_e2e_faster_rcnn_R-50-FPN_v1_13'

res_dir = '../coco_citypersons_val_citypersons_1gpu_e2e_faster_rcnn_R-50-FPN_v1_13-239999_dt.json'
i = 237
# resFile = os.path.join(res_dir, "coco_citypersons_val_citypersons_1gpu_e2e_faster_rcnn_R-50-FPN_v1_13-{}999_dt.json".format(i))
# res_dir = '../val_groundtruth_dt.json'
annFile = '../val_gt.json'

# initialize COCO detections api

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


