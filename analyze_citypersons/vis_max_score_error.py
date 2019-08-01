#! /usr/bin/env python
# -*- coding:utf8 -*-
# __author__ : "ZhangTianliang"
# Date: 19-3-8

# import sys
# sys.path.insert(0, '../')
from utils.coco_citypersons import COCO_citypersons
from utils.MR_Error_visualization import COCOeval_citypersons
import os

subset = {
    'Reasonable': 0,
    'Reasonable_small': 1,
    'Reasonable_occ=heavy': 2,
    'All': 3,
    'All-50': 4,
    'R_occ=None': 5,
    'R_occ=Partial': 6,
    'R_occ=heaavy': 7,
    'Heavy': 8,
    'Partial': 9,
    'Bare': 10,
}

annType = 'bbox'      #specify type here

version = 'v51_14'
iteration = 30000
id_setup = subset['Reasonable']
# ----------------------------------------------------------
print(version, iteration, id_setup)
val_dataset = 'citypersons_o20h20_val'
root = '/media/tianliang/Cloud/PyTorch_Projects/maskrcnn-benchmark_visible_rate/Output/inference'
# root = '/media/tianliang/Cloud/PyTorch_Projects/maskrcnn-benchmark/Output/inference'
cfg_file = "e2e_faster_rcnn_R_50_C4_1x_1_gpu_citypersons_{}".format(version)

# res_dir = os.path.join(root, cfg_file, val_dataset, str(iteration), 'bbox.json')
project_dir = os.path.dirname(os.path.dirname(__file__))
res_dir = os.path.join(project_dir, "res/val/{}/bbox_new.json".format(version))
annFile = '../evaluation/val_gt.json'
# annFile = os.path.join(project_dir, 'json_annotations/citypersons_o20h50_val.json')
output_dir = os.path.join('/tmp/bbox_2.txt')

res_file = open(output_dir, "w")
# for id_setup in range(0, 11):
cocoGt = COCO_citypersons(annFile)
cocoDt = cocoGt.loadRes(res_dir)
imgIds = sorted(cocoGt.getImgIds())
cocoEval = COCOeval_citypersons(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate(id_setup)
cocoEval.accumulate(id_setup, version)
out = cocoEval.summarize(id_setup, res_file)
res_file.close()
print(60*'*')


