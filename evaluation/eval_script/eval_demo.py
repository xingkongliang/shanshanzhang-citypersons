# import sys
# import os
# ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, ROOT_PATH)
from coco_citypersons import COCO_citypersons
from eval_MR_multisetup import COCOeval_citypersons
import os

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
version = 'v16_01'

# iteration = 30000
for iteration in iterations:
    print(version, iteration)
    val_dataset = 'citypersons_o20h20_val'
    # root = '/media/tianliang/Cloud/PyTorch_Projects/maskrcnn-benchmark_visible_rate/Output/inference'
    # root = '/media/tianliang/Cloud/PyTorch_Projects/maskrcnn-benchmark/Output/inference'
    root = "/media/tianliang/Cloud/PyTorch_Projects/ICCV19_detections/0321_CityPersons_Results"
    cfg_file = "e2e_faster_rcnn_R_50_C4_1x_1_gpu_citypersons_{}".format(version)

    # res_dir = os.path.join(root, cfg_file, val_dataset, str(iteration), 'bbox.json')
    res_dir = os.path.join(root, version, "bbox.json")
    output_dir = os.path.join(root, version, "{}_valset_new_result.txt".format(version))
    # res_dir = '../val_groundtruth_dt.json'
    annFile = '../val_gt.json'
    # output_dir = os.path.join(root, cfg_file, val_dataset, str(iteration), 'bbox_new.txt')
    # running evaluation
    res_file = open(output_dir, "w")
    res = []
    for id_setup in range(0, 12):
        cocoGt = COCO_citypersons(annFile)
        cocoDt = cocoGt.loadRes(res_dir)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval_citypersons(cocoGt, cocoDt, annType)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate(id_setup)
        out = cocoEval.summarize(id_setup, res_file)
        res.extend(out)
    for cls in classes:
        for r in res:
            if cls in r:
                print(r.split(' ')[-1][:-1])
                res_file.write(r.split(' ')[-1][:-1])
                res_file.write('\n')
    res_file.close()
    print(60*'*')


