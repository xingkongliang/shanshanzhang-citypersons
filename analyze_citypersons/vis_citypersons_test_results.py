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
    version = 'testset_v42_04'
    iteration = '30000'
    GPU = '1_gpu'
    root = '/media/tianliang/Projects/Caffe2_Projects/detectron-data/citypersons/leftImg8bit'
    dataset = 'test'
    val_dataset = 'citypersons_o20h20_val'
    # --------------------------------------------------------------------
    project_dir = os.path.dirname(os.path.dirname(__file__))
    print("project dir: ", project_dir)
    cfg_file = "e2e_faster_rcnn_R_50_C4_1x_{}_citypersons_{}".format(GPU, version)
    dt_file = "/media/tianliang/Cloud/PyTorch_Projects/ICCV19_detections/val_test_results/8_gpu_v42_04_improved_testset_bbox.json"
    # dt_file = os.path.join(project_dir, "res/val/{}/{}/{}/bbox.json".format(cfg_file, val_dataset, iteration))
    # root = '/media/tianliang/Cloud/PyTorch_Projects/maskrcnn-benchmark_visible_rate/Output/inference'
    gt_file = "/media/tianliang/DATA/DataSets/Cityscapes/shanshanzhang-citypersons/json_annotations/citypersons_test_forvis.json"
    # gt_file = os.path.join(project_dir, 'evaluation/val_gt.json')
    output_dir = os.path.join(project_dir, 'img_output', version)
    print("Output dir: {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dpi = 200
    rpn = False
    color = colormap()

    dts = json.load(open(dt_file, 'r'))
    gts = json.load(open(gt_file, 'r'))

    cocoGt = COCO_citypersons(gt_file)
    cocoDt = cocoGt.loadRes(dt_file)
    catIds = cocoGt.getCatIds(catNms=['pedestrian'])
    imgIds_pedestrian = cocoDt.getImgIds(catIds=catIds)
    imgIds_pedestrian = sorted(imgIds_pedestrian)
    # plt.ion()
    for idx_img in imgIds_pedestrian:
        catIds = cocoGt.getCatIds(catNms=['pedestrian'])
        imgIds = cocoDt.getImgIds(catIds=catIds)
        img = cocoDt.loadImgs(imgIds[idx_img])[0]
        im_name = os.path.basename(img['file_name'])
        im = cv2.imread(os.path.join(root, dataset, im_name.split('_')[0], im_name))
        print("{}-----{}------{}".format(idx_img, len(imgIds_pedestrian), img['file_name']))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        try:
            img_dt = cocoDt.loadImgs(imgIds[idx_img])[0]
            assert img_dt['file_name'] == img['file_name']
            annIds_dt = cocoDt.getAnnIds(imgIds=img_dt['id'], catIds=catIds, iscrowd=None)
            anns_dt = cocoDt.loadAnns(annIds_dt)

            dt_xywh = []
            scores = []
            proposals_xywh = []
            objectness = []
            if "proposals" in anns_dt[0].keys():
                rpn = True
            for idx_dt, ann_dt in enumerate(anns_dt):
                dt_xywh.append(ann_dt['bbox'])
                scores.append(ann_dt['score'])
                if rpn:
                    proposals_xywh.append(ann_dt['proposals'])
                    objectness.append(ann_dt['objectness'])

            dt_xywh = np.array(dt_xywh)
            scores = np.array(scores)
            if rpn:
                proposals_xywh = np.array(proposals_xywh)
                objectness = np.array(objectness)
            if dt_xywh.shape[0] == 0:
                continue
            dt = tlutils.utils.boxes.xywh_to_xyxy(dt_xywh)
            for i, bbox in enumerate(dt_xywh):
                if scores[i] >= 0.05:
                    im = vis_bbox(im, bbox, color._RED, thick=2)
            # if rpn:
            #     for bbox in proposals_xywh:
            #         im = vis_bbox(im, bbox, color._BLUE, thick=2)
            for i, bbox in enumerate(dt_xywh):
                if scores[i] >= 0.05:
                    randint = np.random.randint(0, 80)
                    # if rpn:
                    #     im = vis_class(im, (bbox[0], bbox[1] - 16 - randint),
                    #                    str("{:.2f}".format(objectness[i])),
                    #                    color=color._YELLOW, font_scale=0.5)
                    im = vis_class(im, (bbox[0], bbox[1] - 30 - randint),
                                   str("{:.2f}".format(scores[i])),
                                   color=color._LIME, font_scale=0.5)
        except:
            print("Error!")
        finally:
            im = Image.fromarray(im)
            im.save(os.path.join(output_dir, im_name))


if __name__ == '__main__':
    main()




