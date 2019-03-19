#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ : "ZhangTianliang"
# Date: 18-10-16

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import sys
from utils.cv2_util import vis_bbox, vis_class
from utils.analysis import evaluateImg, error_type_analysis
from utils.colormap import colormap
from utils.box_utils import jaccard

sys.path.insert(0, '../evaluation/eval_script')

from coco_citypersons import COCO_citypersons
import tlutils


def main():
    # --------------------------------------------------------------------
    annType = 'bbox'      #specify type here
    print('Running demo for *%s* results.'%(annType))
    plot_ignore = True
    version = 'caltech_newo65h50_trainval'
    iteration = '30000'
    GPU = '1_gpu'
    root = '/media/tianliang/DATA/DataSets/Pedestrian_Datasets/data-USA'
    dataset = 'train'
    val_dataset = 'citypersons_o20h20_val'
    gt_file = '/media/tianliang/DATA/DataSets/Pedestrian_Datasets/data-USA/json_annotations/caltech_newo65h50_trainval.json'
    # --------------------------------------------------------------------
    project_dir = os.path.dirname(os.path.dirname(__file__))
    print("project dir: ", project_dir)
    cfg_file = "e2e_faster_rcnn_R_50_C4_1x_{}_citypersons_{}".format(GPU, version)
    # root = '/media/tianliang/Cloud/PyTorch_Projects/maskrcnn-benchmark_visible_rate/Output/inference'
    # gt_file = os.path.join(project_dir, 'evaluation/val_gt.json')
    output_dir = os.path.join(project_dir, 'img_output', version)
    print("Output dir: {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dpi = 200
    rpn = False
    color = colormap()

    gts = json.load(open(gt_file, 'r'))

    cocoGt = COCO_citypersons(gt_file)

    catIds = cocoGt.getCatIds(catNms=['pedestrian'])
    imgIds_pedestrian = cocoGt.getImgIds(catIds=catIds)
    imgIds_pedestrian = sorted(imgIds_pedestrian)
    # plt.ion()
    for idx_img in imgIds_pedestrian:
        catIds = cocoGt.getCatIds(catNms=['pedestrian'])
        imgIds = cocoGt.getImgIds(catIds=catIds)
        img = cocoGt.loadImgs(imgIds[idx_img])[0]
        im_name = os.path.basename(img['file_name'])
        im = cv2.imread(os.path.join(root, dataset, 'images', img['file_name']))
        print("{}-----{}------{}".format(idx_img, len(imgIds_pedestrian), img['file_name']))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        annIds = cocoGt.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = cocoGt.loadAnns(annIds)
        try:
            gt_xywh = []
            ignore = []
            for idx, ann in enumerate(anns):
                gt_xywh.append(ann['bbox'])
                ignore.append(ann['ignore'])
            gt_xywh = np.array(gt_xywh)
            ignore = np.array(ignore)

            gt = tlutils.utils.boxes.xywh_to_xyxy(gt_xywh)

            targets_clearn_bbox = gt_xywh[ignore != 1]
            targets_ignore_bbox = gt_xywh[ignore == 1]

            for bbox in targets_clearn_bbox:
                im = vis_bbox(im, bbox, color._GREEN, thick=2)
            for bbox in targets_ignore_bbox:
                im = vis_bbox(im, bbox, color._YELLOW, thick=2)

        except:
            print("Error!")
        finally:
            im = Image.fromarray(im)
            im.save(os.path.join(output_dir, im_name))


if __name__ == '__main__':
    main()




