#! /usr/bin/env python
# -*- coding:utf8 -*-
# __author__ : "ZhangTianliang"
# Date: 2019/8/19

import os
import numpy as np
import cv2
import glob

image_dir = "/media/tianliang/DATA/DataSets/Cityscapes/shanshanzhang-citypersons/img_output/FreeAnchor"

image_list = glob.glob(os.path.join(image_dir, "*.png"))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 24
save_size = (2048, 1024)
out = cv2.VideoWriter('output.avi', fourcc, fps, save_size)

for img_file in image_list:
    img = cv2.imread(img_file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # write the flipped frame
    img = cv2.resize(img, save_size)
    out.write(img)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()