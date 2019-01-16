#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ : "ZhangTianliang"
# Date: 18-11-27

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
from random import shuffle
import os
import glob
import scipy.sparse
import time
import scipy.io as sio
import tlutils.utils as utils


class dataset_to_coco():
    def __init__(self, image_set, year, devkit_path):
        self._name = 'citypersons_' + image_set
        self._year = year
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._image_width = 2048
        self._image_height = 1024
        self._anno_dir = 'shanshanzhang-citypersons/annotations'
        self._images_dir = 'leftImg8bit'
        self._citypersons_classes = ['ignore', 'pedestrians', 'riders', 'sitting-persons', 'other', 'people']
        self._lbls = ['pedestrians']         # return objs with these labels (or [] to return all)
        self._ilbls = ['ignore', 'riders', 'sitting-persons', 'other', 'people']  # return objs with these labels but set to ignore
        # self._squarify = [0.41]             # controls optional reshaping of bbs to fixed aspect ratio
        # self._hRng = [30, np.inf]           # acceptable obj heights
        # self._vRng = [1, 1]               # no occlusion, acceptable obj occlusion levels
        # self._vRng = [0.65, 1]            # reasonable, acceptable obj occlusion levels
        # self._vRng = [0.2, 1]             # heavy occlusion, acceptable obj occlusion levels
        # self._vRng = [0.42, 1]              # heavy occlusion, acceptable obj occlusion levels
        self._squarify = None          # controls optional reshaping of bbs to fixed aspect ratio
        self._hRng = None              # acceptable obj heights
        self._vRng = [0.2, 1]              # acceptable obj occlusion levels

        self._data_path = os.path.join(self._devkit_path)

        self._classes = ('__background__',  # always index 0
                         'pedestrian')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.jpg', '.png']
        self._anno_file = os.path.join(self._devkit_path, self._anno_dir, 'anno_{}.mat'.format(self._image_set))
        if self._image_set != 'test':
            self._images_files, self._image_index, self._annotations = self._load_image_set_index()

        # print("Height: {}".format(self._hRng[0]))
        # print("Occlusion level: {}-{}".format(self._vRng[0], self._vRng[1]))

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'leftImg8bit', self._image_set, index + ext)
            if os.path.exists(image_path):
                break
        # assert os.path.exists(image_path), \
        #     'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        assert os.path.exists(self._anno_file), \
                'Path does not exist: {}'.format(self._anno_file)
        data = sio.loadmat(self._anno_file)
        data = data['anno_{}_aligned'.format(self._image_set)]

        images_files = []
        image_index = []
        annotations = []
        for item in data[0]:
            anno = dict()
            anno['im_name'] = item['im_name'][0][0][0]
            anno['cityname'] = item['cityname'][0][0][0]
            # [class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis]
            anno['bbs'] = item['bbs'][0][0]
            anno['file'] = os.path.join(self._devkit_path, self._images_dir, anno['cityname'], anno['im_name'])
            images_files.append(anno['file'])
            image_index_tmp = os.path.join(anno['cityname'], anno['im_name'].split('.')[0])
            image_index.append(image_index_tmp)
            annotations.append(anno)

        return images_files, image_index, annotations

    def dataset_to_coco_test(self):
        coco_dict = dict()
        coco_dict[u'images'] = []

        ImageSets_Seg = os.path.join(self._devkit_path, "ImageSets", "Segmentation", "test.txt")
        JPEGImages_path = os.path.join(self._devkit_path, "JPEGImages")
        assert os.path.exists(ImageSets_Seg)
        # Test Json Format Example
        # "images": [{"file_name": "frankfurt/frankfurt_000000_000294_leftImg8bit.png",
        # "width": 2048, "id": 0, "height": 1024}]
        image_list = []
        f = open(ImageSets_Seg, 'r')
        for line in f.readlines():
            image_list.append(line.split()[0]+'.jpg')
        f.close()

        for index, image_name in enumerate(image_list):
            image_full_path = os.path.join(JPEGImages_path, image_name)
            im = cv2.imread(image_full_path)
            assert len(im.shape) == 3
            image_height = im.shape[0]
            image_width = im.shape[1]
            if self._year == '2007':
                image_id = int(image_name.split('.')[0])
            elif self._year == '2012':
                image_id = int(image_name.split('_')[0] + image_name.split('_')[1].split('.')[0])
            else:
                raise Exception("Error!")
            # images
            image_dict = {u'file_name': image_name.decode('utf-8'),
                          u'id': image_id,
                          u'height': image_height,
                          u'width': image_width}

            coco_dict[u'images'].append(image_dict)
        print('Num of Caltech Images: ', len(image_list))

        coco_categories = [{u'id': 1, u'name': u'semantic', u'supercategory': u'semantic'}]
        ###
        coco_info_dict = {u'contributor': u"PascalVOC{}".format(self._year),
                          u'date_created': u'2018-04-06 16:00:00',
                          u'description': u"This is a COCO Dataset version of PascalVOC{}.".format(self._year),
                          u'url': u'http://None',
                          u'version': u'1.0',
                          u'year': 2018}

        coco_type = u'semantic segmentation'
        coco_dict[u'info'] = coco_info_dict
        coco_dict[u'categories'] = coco_categories
        coco_dict[u'type'] = coco_type

        return coco_dict



if __name__ == '__main__':
    print('Convert Caltech Data to COCO Format...')
    year = '2012'
    if year == '2007':
        root = '/media/tianliang/DATA/DataSets/Pascal-2007/VOC2007'
    elif year == '2012':
        root = '/media/tianliang/DATA/DataSets/Pascal-2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    else:
        raise Exception("Error!")
    annotations_dir = os.path.join(root, '../json_annotations')

    image_set = 'test'
    citypersons = dataset_to_coco(image_set, year, root)
    coco_dict_test = citypersons.dataset_to_coco_test()
    if year == '2007':
        f = open(os.path.join(annotations_dir, "voc_{}_SemanticSegmentation_val.json".format(year)), 'w')
    elif year == '2012':
        f = open(os.path.join(annotations_dir, "voc_{}_SemanticSegmentation_test.json".format(year)), 'w')
    else:
        raise Exception("Error!")
    f.write(json.dumps(coco_dict_test))
    f.close()

    print('Done.')