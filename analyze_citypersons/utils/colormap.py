#! /usr/bin/env python
# -*- coding:utf8 -*-
# __author__ : "ZhangTianliang"
# Date: 19-3-3


class colormap(object):
    def __init__(self):
        # -------------- color maps ----------------------------
        self._GREEN = (18, 127, 15)         # ground truth
        self._YELLOW = (255, 255, 0)        # ignore ground truth and rpn scores
        self._RED = (255, 0, 0)             # detections
        self._BLUE = (0, 0, 255)            # rpn boxes
        self._LIME = (0, 255, 0)            # detection scores
        self._LIGHTGREEN = (144, 238, 144)  # visible rate
        # -------------------------------------------------------
        self._GRAY = (218, 227, 218)        # double detections error
        self._BLACK = (0, 0, 0)             # crowded error
        self._ORANGE = (255, 165, 0)        # larger bbs error
        self._PURPLE = (128, 0, 128)        # body parts error
        self._BLUEVIOLET = (138, 43, 226)   # background error
        self._BROWN = (165, 42, 42)         # others error
        self._WHITE = (255, 255, 255)
        # ------------------------------------------------------
