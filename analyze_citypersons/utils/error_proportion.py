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
method2 = "8_gpu_v48_02_valset_bbox_error_rate.json"

# result1 = np.load(os.path.join(root_dir, method1))
# result2 = np.load(os.path.join(root_dir, method2))

reuslt1 = json.load(open(os.path.join(root_dir, method1), 'r'))
result1_background = reuslt1['background_error_rate']


result2 = json.load(open(os.path.join(root_dir, method2), 'r'))
result2_background = result2['background_error_rate']
start_index1 = np.where(np.isnan(result1_background))[0][-1] + 1
start_index2 = np.where(np.isnan(result2_background))[0][-1] + 1
len1 = len(result1_background)
len2 = len(result2_background)
final_len = min(len1, len2)
# final_len = 1800
start_both = 35
plt.figure(figsize=(4, 3))
# plt.figure()
ax = plt.gca()
ax.plot(np.arange(start_both, final_len-start_index1-1),
         result1_background[start_index1+1+start_both:final_len], 'b-', label="Baseline")
ax.plot(np.arange(start_both, final_len-start_index2-1),
         result2_background[start_index2+1+start_both:final_len], 'r-', label="SA")
ax.set_xlabel('num of false positive')
ax.set_ylabel('proportion of background error')
plt.legend(fontsize='large')
# plt.grid()
# plt.grid(axis="y", ls='--')
plt.grid(axis="y", linestyle='-.')
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

print("done!")