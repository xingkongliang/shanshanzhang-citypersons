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
from utils import boxes
sys.path.insert(0, '../evaluation/eval_script')

from coco_citypersons import COCO_citypersons
# import tlutils


def draw_bbox_on_image(im, dt, scores, proposals, objectness, gt, gt_ignore, color, dt_error_type,
                       draw_proposals=False, draw_score=True, score_threshold=0.2, linewidth=2):
    """

    :param dt: xywh
    :param proposals: xywh
    :param gt: xywh
    :param gt_ignore: xywh
    :param draw_proposals:
    :param draw_score:
    :return:
    """
    if draw_proposals:
        for bbox in proposals:
            im = vis_bbox(im, bbox, color._BLUE, thick=linewidth)
    for bbox in gt:
        im = vis_bbox(im, bbox, color._GREEN, thick=linewidth)
    # for bbox in gt_ignore:
    #     im = vis_bbox(im, bbox, color._YELLOW, thick=linewidth)
    for i, bbox in enumerate(dt):
        if scores[i] > score_threshold:
            if dt_error_type[i] == -1:
                using_color = color._RED
            else:
                using_color = color._BLUE
            im = vis_bbox(im, bbox, using_color, thick=linewidth)
    for i, bbox in enumerate(dt):
        randint = np.random.randint(0, 80)
        if scores[i] > score_threshold:
            if draw_proposals:
                im = vis_class(im, (bbox[0], bbox[1] - 16 - randint),
                               str("{:.2f}".format(objectness[i])),
                               color=color._YELLOW, font_scale=0.5)
            im = vis_class(im, (bbox[0], bbox[1] - 2),
                           str("{:.2f}".format(scores[i])),
                           color=color._RED, font_scale=0.5)
    return im


def draw_bbox_on_image_with_errortype(im, dt, scores, proposals, objectness, gt, gt_ignore, color, error_type_color_map,
                                      dt_error_type, num_to_error_type, draw_proposals=True, draw_score=True,
                                      score_threshold=0.2, linewidth=3):
    """

    :param dt: xywh
    :param proposals: xywh
    :param gt: xywh
    :param gt_ignore: xywh
    :param draw_proposals:
    :param draw_score:
    :return:
    """
    if draw_proposals:
        for bbox in proposals:
            im = vis_bbox(im, bbox, color._BLUE, thick=linewidth)
    for bbox in gt:
        im = vis_bbox(im, bbox, color._GREEN, thick=linewidth)
    for bbox in gt_ignore:
        im = vis_bbox(im, bbox, color._YELLOW, thick=linewidth)
    for i, bbox in enumerate(dt):
        if scores[i] > score_threshold:
            im = vis_bbox(im, bbox, error_type_color_map[dt_error_type[i]], thick=linewidth)
    for i, bbox in enumerate(dt):
        if scores[i] > score_threshold:
            randint = np.random.randint(0, 80)
            if draw_proposals:
                im = vis_class(im, (bbox[0], bbox[1] - 16 - randint),
                               str("{:.2f}".format(objectness[i])),
                               color=color._YELLOW, font_scale=0.5)
            im = vis_class(im, (bbox[0], bbox[1] - 30 - randint),
                           str("{:.2f}".format(scores[i])),
                           color=color._LIME, font_scale=0.5)
            im = vis_class(im, (bbox[0], bbox[1] - 44 - randint),
                           str("{}".format(num_to_error_type[dt_error_type[i]])),
                           color=color._LIME, font_scale=0.5)
    return im


def main():
    # --------------------------------------------------------------------
    annType = 'bbox'      #specify type here
    print('Running demo for *%s* results.'%(annType))
    plot_ignore = True
    version = 'FreeAnchor'
    iteration = '30000'
    GPU = '1_gpu'
    root = '/media/tianliang/Projects/Caffe2_Projects/detectron-data/citypersons/leftImg8bit'
    dataset = 'val'
    # --------------------------------------------------------------------
    project_dir = os.path.dirname(os.path.dirname(__file__))
    print("project dir: ", project_dir)
    # cfg_file = "e2e_faster_rcnn_R_50_C4_1x_{}_citypersons_{}".format(GPU, version)
    cfg_file = 'v51_14'
    # dt_file = os.path.join(project_dir, "res/val/{}/{}/{}/bbox.json".format(cfg_file, val_dataset, iteration))
    dt_file = os.path.join(project_dir, "res/val/{}/bbox_new.json".format(cfg_file))
    # root = '/media/tianliang/Cloud/PyTorch_Projects/maskrcnn-benchmark_visible_rate/Output/inference'
    gt_file = os.path.join(project_dir, 'evaluation/val_gt.json')
    # gt_file = os.path.join(project_dir, 'json_annotations/citypersons_o20h50_val.json')
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
    imgIds_pedestrian = cocoGt.getImgIds(catIds=catIds)
    imgIds_pedestrian = sorted(imgIds_pedestrian)
    # plt.ion()
    # for idx_img in imgIds_pedestrian:
    for idx_img, img in cocoGt.imgs.items():
        catIds = cocoGt.getCatIds(catNms=['pedestrian'])
        imgIds = cocoGt.getImgIds(catIds=catIds)
        # img = cocoGt.loadImgs(imgIds[idx_img])[0]
        try:
            im_name_coco = img['im_name']
            im_name = os.path.basename(im_name_coco)
            im = cv2.imread(os.path.join(root, dataset, im_name_coco.split('_')[0], im_name_coco))
        except:
            im_name_coco = img['file_name']
            im_name = os.path.basename(im_name_coco)
            im = cv2.imread(os.path.join(root, dataset, im_name_coco))
        print("{}-----{}------{}".format(idx_img, len(cocoGt.imgs), im_name_coco))
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

            img_dt = cocoDt.loadImgs(idx_img)[0]
            # assert img_dt['im_name'] == im_name_coco
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
            # if dt_xywh.shape[0] == 0 or gt_xywh.shape[0] == 0:
            #     continue
            dt = boxes.xywh_to_xyxy(dt_xywh)
            gt = boxes.xywh_to_xyxy(gt_xywh)
            ious = jaccard(torch.tensor(dt).float(), torch.tensor(gt).float())
            ious = ious.numpy()

            targets_clearn_bbox = gt_xywh[ignore != 1]
            targets_ignore_bbox = gt_xywh[ignore == 1]

            dtIg, dtm, gtIg, gtm = evaluateImg(dt, gt, ignore, ious)
            dt_error_type, error_type_count, num_to_error_type, error_type_color_map = error_type_analysis(
                dt, scores, dtIg, dtm, gt, gtIg, gtm, ignore, ious)
            im = draw_bbox_on_image(im, dt_xywh, scores, proposals_xywh, objectness, targets_clearn_bbox,
                                    targets_ignore_bbox, color,
                                    dt_error_type, draw_proposals=rpn, draw_score=True, score_threshold=0.2, linewidth=3)
            # im = draw_bbox_on_image_with_errortype(im, dt_xywh, scores, proposals_xywh, objectness, targets_clearn_bbox,
            #                                        targets_ignore_bbox, color, error_type_color_map, dt_error_type,
            #                                        num_to_error_type, draw_proposals=rpn, draw_score=True, linewidth=3)
        except:
            print("Error!")
        finally:
            im = Image.fromarray(im)
            im.save(os.path.join(output_dir, im_name))


if __name__ == '__main__':
    main()




