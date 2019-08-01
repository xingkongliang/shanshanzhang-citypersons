#! /usr/bin/env python
# -*- coding:utf8 -*-
# __author__ : "ZhangTianliang"
# Date: 19-3-19
import os
import numpy as np
import matplotlib.pyplot as plt
import json

root_dir = "/media/tianliang/Cloud/PyTorch_Projects/ICCV19_detections/val_test_results"
method1 = "8_gpu_v16_01_valset_bbox_error_rate.json"
method2 = "8_gpu_v48_02_valset_bbox_error_rate.json"  # Per-pixel
method3 = "8_gpu_v42_04_valset_bbox_error_rate.json"  # Holistic

# result1 = np.load(os.path.join(root_dir, method1))
# result2 = np.load(os.path.join(root_dir, method2))

reuslt1 = json.load(open(os.path.join(root_dir, method1), 'r'))
result1_background = reuslt1['background_error_rate']

result2 = json.load(open(os.path.join(root_dir, method2), 'r'))
result2_background = result2['background_error_rate']

result3 = json.load(open(os.path.join(root_dir, method3), 'r'))
result3_background = result3['background_error_rate']

start_index1 = np.where(np.isnan(result1_background))[0][-1] + 1
start_index2 = np.where(np.isnan(result2_background))[0][-1] + 1
start_index3 = np.where(np.isnan(result2_background))[0][-1] + 1
len1 = len(result1_background)
len2 = len(result2_background)
len3 = len(result3_background)
final_len = min(len1, len2)
final_len = min(final_len, len3)
final_len = 1996
start_both = 1232
plt.figure(figsize=(4, 3))
# plt.figure()
ax = plt.gca()
ax.plot(np.arange(start_both, final_len-1),
         result1_background[1+start_both:final_len], 'b-', label="Baseline")
ax.plot(np.arange(start_both, final_len-1),
         result2_background[1+start_both:final_len], 'k-', label="Pixel-wise calibration")
ax.plot(np.arange(start_both, final_len-1),
         result3_background[1+start_both:final_len], 'r-', label="Region calibration")
ax.set_xlabel('false positive per image (FPPI)')
ax.set_ylabel('proportion of background error')
plt.legend(fontsize='small', loc='upper right')
plt.xticks([1217, 1367, 1608, 1996],
           [0.018, 0.056, 0.316, 1.])
# plt.grid()
# plt.grid(axis="y", ls='--')
plt.grid(axis="y", linestyle='-.')
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# v48_02 fppi array([1132,   1217,   1294,   1367,   1436,   1502,   1608,   1749,    1996])
# fppi        array([0.01  , 0.0178, 0.0316, 0.0562, 0.1   , 0.1778, 0.3162, 0.5623,  1.    ])
# v16_01 array([1026, 1173, 1284, 1348, 1440, 1515, 1612, 1753, 1989])
print("done!")