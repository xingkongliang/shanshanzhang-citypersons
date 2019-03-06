#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ : "ZhangTianliang"
# Date: 18-10-16

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
sys.path.insert(0, '../evaluation/eval_script')

from coco_citypersons import COCO_citypersons
from eval_MR_multisetup import COCOeval_citypersons
import tlutils

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_LIGHTGREEN = (144, 238, 144)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_BLUE = (0, 0, 255)
_YELLOW = (255, 255, 0)
_RED = (255, 0, 0)
_ORANGE = (255, 165, 0)

# --------------------------------------------------------------------
annType = 'bbox'      #specify type here
print('Running demo for *%s* results.'%(annType))
plot_ignore = True
version = 'v28_01'
iteration = '4000'
root = '/media/tianliang/Projects/Caffe2_Projects/detectron-data/citypersons/leftImg8bit'
dataset = 'val'
val_dataset = 'citypersons_o20h20_val'
# --------------------------------------------------------------------
project_dir = os.path.dirname(os.path.dirname(__file__))
print("project dir: ", project_dir)
cfg_file = "e2e_faster_rcnn_R_50_C4_1x_1_gpu_citypersons_{}".format(version)
dt_file = os.path.join(project_dir, "res/val/{}/citypersons_o20h20_val/40000/bbox.json".format(cfg_file, version))
root = '/media/tianliang/Cloud/PyTorch_Projects/maskrcnn-benchmark_visible_rate/Output/inference'
gt_file = os.path.join(project_dir, 'evaluation/val_gt.json')
output_dir = os.path.join(project_dir, 'img_output', version)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
id_setup = 0  # for reasonable
dpi = 200

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
    im = cv2.imread(os.path.join(root, dataset, img['im_name'].split('_')[0], img['im_name']))
    print("{}-----{}------{}".format(idx_img, len(imgIds_pedestrian), img['im_name']))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    annIds = cocoGt.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = cocoGt.loadAnns(annIds)

    idx_useful = []
    bbs = []
    ignore = []
    for idx, ann in enumerate(anns):
        if ann['ignore'] == 0 or plot_ignore:
            idx_useful.append(idx)
            bbs.append(ann['bbox'])
            ignore.append(ann['ignore'])

    if len(bbs) != 0:
        bbs = np.array(bbs)
        bboxes = tlutils.utils.boxes.xywh_to_xyxy(bbs)

        classes_boxes = ['pedestrian' if i == 0 else 'ignore' for i in ignore]
        color_classes = [_GREEN if i == 0 else _YELLOW for i in ignore]

        out = tlutils.utils.vis.vis_one_image_opencv(im, bboxes, classes=classes_boxes,
                                                     show_box=1, show_class=1, color=color_classes, line=3)

    img_dt = cocoDt.loadImgs(imgIds[idx_img])[0]
    assert img_dt['im_name'] == img['im_name']
    annIds_dt = cocoDt.getAnnIds(imgIds=img_dt['id'], catIds=catIds, iscrowd=None)
    anns_dt = cocoDt.loadAnns(annIds_dt)

    bbs_dt = np.zeros((len(anns_dt), 4))
    for idx_dt, ann_dt in enumerate(anns_dt):
        bbs_dt[idx_dt, :] = ann_dt['bbox']

    bboxes_dt = tlutils.utils.boxes.xywh_to_xyxy(bbs_dt)

    classes_boxes = ['pedestrian' for i in range(len(anns_dt))]
    color_classes_dt = [_RED for i in range(len(anns_dt))]
    if len(bbs) == 0:
        out_dt = tlutils.utils.vis.vis_one_image_opencv(out, bboxes_dt, classes=classes_boxes,
                                                        show_box=1, show_class=0, color=color_classes_dt, line=3)
    else:
        out_dt = tlutils.utils.vis.vis_one_image_opencv(out, bboxes_dt, classes=classes_boxes,
                                                        show_box=1, show_class=0, color=color_classes_dt, line=3)
    output_name = os.path.basename(img['im_name']) + '.png'
    # plt.cla()
    # plt.axis('off')
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.imshow(out_dt)
    # plt.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=200)
    # plt.pause(3)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(out_dt)
    fig.savefig(os.path.join(output_dir, output_name), dpi=dpi)
    # plt.show()
    plt.close('all')




