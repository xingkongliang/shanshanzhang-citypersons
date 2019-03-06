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
from eval_MR_multisetup import COCOeval_citypersons
import tlutils


def main():
    # --------------------------------------------------------------------
    annType = 'bbox'      #specify type here
    print('Running demo for *%s* results.'%(annType))
    plot_ignore = True
    version = 'v28_01'
    iteration = '40000'
    GPU = '1_gpu'
    root = '/media/tianliang/Projects/Caffe2_Projects/detectron-data/citypersons/leftImg8bit'
    dataset = 'val'
    val_dataset = 'citypersons_o20h20_val'
    # --------------------------------------------------------------------
    project_dir = os.path.dirname(os.path.dirname(__file__))
    print("project dir: ", project_dir)
    cfg_file = "e2e_faster_rcnn_R_50_C4_1x_{}_citypersons_{}".format(GPU, version)
    dt_file = os.path.join(project_dir, "res/val/{}/{}/{}/bbox.json".format(cfg_file, val_dataset, iteration))
    # root = '/media/tianliang/Cloud/PyTorch_Projects/maskrcnn-benchmark_visible_rate/Output/inference'
    gt_file = os.path.join(project_dir, 'evaluation/val_gt.json')
    output_dir = os.path.join(project_dir, 'img_output', version)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    id_setup = 0  # for reasonable
    dpi = 200
    color = colormap()

    dts = json.load(open(dt_file, 'r'))
    gts = json.load(open(gt_file, 'r'))
    res_file = open("results.txt", "w")

    cocoGt = COCO_citypersons(gt_file)
    cocoDt = cocoGt.loadRes(dt_file)
    catIds = cocoGt.getCatIds(catNms=['pedestrian'])
    imgIds_pedestrian = cocoGt.getImgIds(catIds=catIds)
    imgIds_pedestrian = sorted(imgIds_pedestrian)
    # plt.ion()
    for idx_img in imgIds_pedestrian:
        catIds = cocoGt.getCatIds(catNms=['pedestrian'])
        imgIds = cocoGt.getImgIds(catIds=catIds)
        img = cocoGt.loadImgs(imgIds[idx_img])[0]
        im_name = os.path.basename(img['im_name']) + '.png'
        im = cv2.imread(os.path.join(root, dataset, img['im_name'].split('_')[0], img['im_name']))
        print("{}-----{}------{}".format(idx_img, len(imgIds_pedestrian), img['im_name']))
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

            img_dt = cocoDt.loadImgs(imgIds[idx_img])[0]
            assert img_dt['im_name'] == img['im_name']
            annIds_dt = cocoDt.getAnnIds(imgIds=img_dt['id'], catIds=catIds, iscrowd=None)
            anns_dt = cocoDt.loadAnns(annIds_dt)

            dt_xywh = []
            scores = []
            proposals_xywh = []
            objectness = []
            for idx_dt, ann_dt in enumerate(anns_dt):
                dt_xywh.append(ann_dt['bbox'])
                scores.append(ann_dt['score'])
                proposals_xywh.append(ann_dt['proposals'])
                objectness.append(ann_dt['objectness'])

            dt_xywh = np.array(dt_xywh)
            scores = np.array(scores)
            proposals_xywh = np.array(proposals_xywh)
            objectness = np.array(objectness)
            if dt_xywh.shape[0] == 0 or gt_xywh.shape[0] == 0:
                continue
            dt = tlutils.utils.boxes.xywh_to_xyxy(dt_xywh)
            gt = tlutils.utils.boxes.xywh_to_xyxy(gt_xywh)
            ious = jaccard(torch.tensor(dt).float(), torch.tensor(gt).float())
            ious = ious.numpy()

            targets_clearn_bbox = gt_xywh[ignore != 1]
            targets_ignore_bbox = gt_xywh[ignore == 1]

            dtIg, dtm, gtIg, gtm = evaluateImg(dt, gt, ignore, ious)
            dt_error_type, error_type_count, num_to_error_type, error_type_color_map = error_type_analysis(
                dt, scores, dtIg, dtm, gt, gtIg, gtm, ignore, ious)

            for bbox in proposals_xywh:
                im = vis_bbox(im, bbox, color._BLUE, thick=2)
            for bbox in targets_clearn_bbox:
                im = vis_bbox(im, bbox, color._GREEN, thick=2)
            for bbox in targets_ignore_bbox:
                im = vis_bbox(im, bbox, color._YELLOW, thick=2)
            for i, bbox in enumerate(dt_xywh):
                im = vis_bbox(im, bbox, error_type_color_map[dt_error_type[i]], thick=2)
            for i, bbox in enumerate(dt_xywh):
                randint = np.random.randint(0, 80)
                im = vis_class(im, (bbox[0], bbox[1] - 16 - randint),
                               str("{:.2f}".format(objectness[i])),
                               color=color._YELLOW, font_scale=0.5)
                im = vis_class(im, (bbox[0], bbox[1] - 30 - randint),
                               str("{:.2f}".format(scores[i])),
                               color=color._LIME, font_scale=0.5)
                im = vis_class(im, (bbox[0], bbox[1] - 44 - randint),
                               str("{}".format(num_to_error_type[dt_error_type[i]])),
                               color=color._LIME, font_scale=0.5)
        except:
            pass
        finally:
            im = Image.fromarray(im)
            im.save(os.path.join(output_dir, im_name))


if __name__ == '__main__':
    main()




