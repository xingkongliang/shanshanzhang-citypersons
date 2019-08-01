#! /usr/bin/env python
# -*- coding:utf8 -*-
# __author__ : "ZhangTianliang"
# Date: 19-4-25

import numpy as np
import os
import sys
import cv2
# import cPickle as pickle
import pickle

sys.path.insert(0, '../evaluation/eval_script')
from coco_citypersons import COCO_citypersons


def iou(dts, gts):
    dts = np.asarray(dts)
    gts = np.asarray(gts)
    ious = np.zeros((len(dts), len(gts)))
    for j, gt in enumerate(gts):
        gx1 = gt[0]  # x_min
        gy1 = gt[1]  # y_min
        gx2 = gt[0] + gt[2]  # x_max
        gy2 = gt[1] + gt[3]  # y_max
        garea = gt[2] * gt[3]
        for i, dt in enumerate(dts):
            dx1 = dt[0]
            dy1 = dt[1]
            dx2 = dt[0] + dt[2]
            dy2 = dt[1] + dt[3]
            darea = dt[2] * dt[3]

            unionw = min(dx2, gx2) - max(dx1, gx1)
            if unionw <= 0:
                continue
            unionh = min(dy2, gy2) - max(dy1, gy1)
            if unionh <= 0:
                continue
            t = unionw * unionh
            unionarea = darea + garea - t

            ious[i, j] = float(t) / unionarea
    return ious


test_annotation_json = '/media/tianliang/DATA/DataSets/Pedestrian_Datasets/data-USA/json_annotations/caltech_test.json'
test_det_proposals = "/media/tianliang/Projects/Caffe2_Projects/MDetectron-v2/test/coco_caltech_test/generalized_rcnn/detections_v3_18_29999.pkl"

det_proposals = open(test_det_proposals, 'rb')
det_proposals = pickle.load(det_proposals, encoding='iso-8859-1')
det_proposals = det_proposals['all_proposals'][1]

cocoGt = COCO_citypersons(test_annotation_json)
catIds = cocoGt.getCatIds(catNms=['pedestrian'])
imgIds_pedestrian = cocoGt.getImgIds(catIds=catIds)
imgIds_pedestrian = sorted(imgIds_pedestrian)
# plt.ion()
for idx_img in imgIds_pedestrian:
    catIds = cocoGt.getCatIds(catNms=['pedestrian'])
    imgIds = cocoGt.getImgIds(catIds=catIds)
    img = cocoGt.loadImgs(imgIds[idx_img])[0]
    try:
        im_name_coco = img['im_name']
        im_name = os.path.basename(im_name_coco)
    except:
        im_name_coco = img['file_name']
        im_name = os.path.basename(im_name_coco)
    print("{}-----{}------{}".format(idx_img, len(imgIds_pedestrian), im_name_coco))
    annIds = cocoGt.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = cocoGt.loadAnns(annIds)
    gt_xywh = []
    ignore = []
    for idx, ann in enumerate(anns):
        gt_xywh.append(ann['bbox'])
        ignore.append(ann['ignore'])
    gt_xywh = np.array(gt_xywh)
    ignore = np.array(ignore)

print('Done!')