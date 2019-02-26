# import sys
# sys.path.insert(0, '../')
from coco_citypersons import COCO_citypersons
from eval_MR_multisetup import COCOeval_citypersons
import os


annType = 'bbox'      #specify type here
# print('Running demo for *%s* results.'%(annType))
iterations = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
version = 'v3_01'

# iteration = 30000
for iteration in iterations:
    print(version, iteration)
    val_dataset = 'citypersons_o20h20_val'
    root = '/media/tianliang/Cloud/PyTorch_Projects/maskrcnn-benchmark/Output/inference'
    cfg_file = "e2e_faster_rcnn_R_50_C4_1x_1_gpu_citypersons_{}".format(version)

    res_dir = os.path.join(root, cfg_file, val_dataset, str(iteration), 'bbox.json')
    # res_dir = '../val_groundtruth_dt.json'
    annFile = '../val_gt.json'
    output_dir = os.path.join(root, cfg_file, val_dataset, str(iteration), 'bbox_2.txt')
    # running evaluation
    res_file = open(output_dir, "w")
    res = []
    for id_setup in range(0, 11):
        cocoGt = COCO_citypersons(annFile)
        cocoDt = cocoGt.loadRes(res_dir)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval_citypersons(cocoGt, cocoDt, annType)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        out = cocoEval.summarize(id_setup, res_file)
        res.append(out)
    for r in res:
            print(r.split(' ')[-1][:-1])
            res_file.write(r.split(' ')[-1][:-1])
            res_file.write('\n')
    res_file.close()
    print(30*'-')


