#! /usr/bin/env python
# -*- coding:utf8 -*-
# __author__ : "ZhangTianliang"
# Date: 19-3-3

from __future__ import division
import torch
import time
import numpy as np
from .colormap import colormap


def evaluateImg(dt, gt, ignore, ious):
    """
    perform evaluation for single category and image
    :return: dict (single image results)
    """
    G = len(gt)
    D = len(dt)
    gtm = -1 * np.ones(G)   # ground truth 匹配是否匹配 标志位
    dtm = -1 * np.ones(D)   # detection 匹配是否匹配 标志位
    gtIg = np.array([g for g in ignore])  # ground truth 忽略 标志位
    dtIg = np.zeros(D)  # detection 忽略 标志位
    for dind, d in enumerate(dt):
        # information about best match so far (m=-1 -> unmatched)
        iou = min([0.5, 1 - 1e-10])
        bstOa = iou
        bstg = -2  # 最好匹配的 gt 序号
        bstm = -2  # 找到了最好的匹配
        for gind, g in enumerate(gt):
            m = gtm[gind]  # 第 gind 个 ground truth 的匹配标志位
            # if this gt already matched, and not a crowd, continue
            if m > 0:  # 如果这个 ground truth 被匹配了 跳到下一个ground truth
                continue
            # if dt matched to reg gt, and on ignore gt, stop
            if bstm != -2 and gtIg[gind] == 1:  # 如果 dt 被匹配到了 gt 并且 比配到了忽略的 gt 则停止
                break
            # continue to next gt unless better match made
            if ious[dind, gind] < bstOa:  # 继续下一个 gt 直到更好的匹配 即遇到更大的IoU
                continue
            # if match successful and best so far, store appropriately
            # 如果匹配成功 并且匹配到了当前最好的结果 存储结果
            bstOa = ious[dind, gind]  # 把最好的IoU赋值给bstOa
            bstg = gind  # 把当前的gt的序号 赋值给 bstg
            if gtIg[gind] == 0:  # 如果当前gt 不是被忽略的
                bstm = 1  # 则将 bstm 置 1
            else:
                bstm = -1  # 如果当前gt 是忽略的 则将 bstm 置 -1

        # if match made store id of match for both dt and gt
        if bstg == -2:  # 没有找到gt与之匹配 则跳出当前dt
            continue
        dtIg[dind] = gtIg[bstg]  # 把当前检测结果标志 是否 忽略
        dtm[dind] = bstg  # 检测结果匹配的 gt 的 id
        if bstm == 1:  # 找到最好匹配
            gtm[bstg] = dind  # gt 匹配标志位 置为 检测结果的id
    # store results for given image and category
    return dtIg, dtm, gtIg, gtm


def error_type_analysis(dt, dt_scores, dtIg, dtm, gt, gtIg, gtm, ignore, ious, crowded_iou=0.1, method="mean"):
    """
    :return:
    """
    color = colormap()
    error_type_count = {"double detections": 0, "crowded": 0, "larger bbs": 0, "body parts": 0, "background": 0, "others": 0}
    num_to_error_type = {-1: "", 0: "double detections", 1: "crowded", 2: "larger bbs", 3: "body parts",
                         4: "background", 5: "others"}
    error_type_color_map = {-1: color._RED, 0: color._GRAY, 1: color._BLACK, 2: color._ORANGE,
                            3: color._PURPLE, 4: color._BLUEVIOLET, 5: color._BROWN}
    error_type = {"double detections": 0, "crowded": 1, "larger bbs": 2, "body parts": 3, "background": 4, "others": 5}

    error_index = np.where((dtIg == 0) & (dtm == -1))[0]
    dt_error_type = -1 * np.ones(len(dt))
    for e_i in error_index:
        non_ignore = 1 - ignore
        non_ignore_index = non_ignore.nonzero()[0]
        iou = ious[e_i, ...][non_ignore.nonzero()]
        if np.sum(iou >= 0.5) >= 1:
            this_error_type = error_type["double detections"]
        elif (np.sum(iou > 0.1) >= 1) and (np.sum(iou > 0.5) < 1):
            gt_index = iou.argmax()
            gt_index = non_ignore_index[gt_index]
            dt_area = (dt[e_i][3] - dt[e_i][1]) * (dt[e_i][2] - dt[e_i][0])
            gt_area = (gt[gt_index][3] - gt[gt_index][1]) * (gt[gt_index][2] - gt[gt_index][0])
            if gt_area >= dt_area:
                this_error_type = error_type["body parts"]
            else:
                this_error_type = error_type["larger bbs"]
        elif np.sum(iou >= 0.1) < 1:
            this_error_type = error_type["background"]
        else:
            this_error_type = error_type["others"]
        if np.sum(iou >= crowded_iou) >= 2:
            this_error_type = error_type["crowded"]
        dt_error_type[e_i] = this_error_type
    for i, key in enumerate(error_type_count.keys()):
        if dt_scores[np.where(dt_error_type == i)[0]].shape[0] == 0:
            error_type_count[key] = float(0)
        else:
            if method == 'mean':
                error_type_count[key] = np.mean(dt_scores[np.where(dt_error_type == i)[0]])*10
            elif method == 'sum':
                error_type_count[key] = np.sum(dt_scores[np.where(dt_error_type == i)[0]]) * 10
            else:
                raise NotImplementedError("Unsupported {} method.".format(method))
    return dt_error_type, error_type_count, num_to_error_type, error_type_color_map


def vis_max_score_error():

    return True
