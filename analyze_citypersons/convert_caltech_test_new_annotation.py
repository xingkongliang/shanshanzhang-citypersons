#! /usr/bin/env python
# -*- coding:utf8 -*-
# __author__ : "ZhangTianliang"
# Date: 19-3-17

import os
import path
import glob

rood_dir = "/media/tianliang/DATA/DataSets/Pedestrian_Datasets/data-USA/test"

source_dir = "anno_test_1xnew"
target_dir = "anno_test_1xnew_target"
file_test = "test.txt"
f = open(os.path.join(rood_dir, file_test), 'w')
file_list = glob.glob(os.path.join(rood_dir, source_dir, "*.txt"))
for file in file_list:
    file_name = os.path.basename(file)
    target_file = file_name.split('.')[0] + '.' + file_name.split('.')[2]
    target_file = os.path.join(rood_dir, target_dir, target_file)
    f.write("{}\n".format(target_file))
    os.system("cp {} {}".format(file, target_file))
f.close()
print("Done!")
