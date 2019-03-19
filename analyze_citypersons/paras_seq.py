#! /usr/bin/env python
# -*- coding:utf8 -*-
# __author__ : "ZhangTianliang"
# Date: 19-3-18

import numpy as np
import json
import os
import scipy.io as sio

root_dir = "/media/tianliang/DATA/DataSets/Pedestrian_Datasets/data-ETH/videos/set00"
file_name = "V000.seq"
file = os.path.join(root_dir, file_name)

m = sio.loadmat(file)

print("Done!")
