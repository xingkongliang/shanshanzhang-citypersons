#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ : "ZhangTianliang"
# Date: 18-9-3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ : "ZhangTianliang"
# Date: 18-4-6

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

categories_dict = {1: 'pedestrian'}
categories3_dict = {'pedestrian': 1}


def check_anno(anno, im_file):
    """Draw detected bounding boxes."""
    roi = anno
    # only visualize the samples with 'gt_ignores' == 1
    # if 0 in roi['gt_ignores']:
    im = cv2.imread(im_file)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for j in range(len(roi['boxes'])):
        bbox = roi['boxes'][j]
        class_name = roi['gt_lbl'][j]
        if roi['gt_ignores'][j]:
            edgecolor = 'red'
        else:
            edgecolor = 'blue'
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=edgecolor, linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(class_name),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    ax.set_title(('{} detections {}').format('pedestrian', im_file.split('/')[-1]),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    time.sleep(0.5)
    plt.close()


class caltech_to_coco():
    def __init__(self, image_set, set, devkit_path):
        self._name = 'caltech_' + image_set
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._set = set
        self._lbls = 'person'             # return objs with these labels (or [] to return all)
        self._ilbls = ['people', 'ignore']            # return objs with these labels but set to ignore
        self._squarify = [0.41]           # controls optional reshaping of bbs to fixed aspect ratio
        self._hRng = [30, np.inf]         # acceptable obj heights
        self._vRng = [1, 1]               # acceptable obj occlusion levels

        self._data_path = os.path.join(self._devkit_path)

        self._classes = ('__background__',  # always index 0
                         'pedestrian')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.jpg', '.png']
        self._images_files, self._image_index, self._annotations_files = self._load_image_set_index()

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
            image_path = os.path.join(self._data_path, self._image_set, 'images', index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        assert os.path.exists('{}/{}/images'.format(self._devkit_path, self._image_set))
        assert os.path.exists('{}/{}/annotations'.format(self._devkit_path, self._image_set))

        images_files = sorted(glob.glob('{}/{}/images/*.jpg'.format(self._devkit_path, self._image_set)))

        def is_inset(file):
            return file.split('/')[-1].split('_')[0] in self._set

        images_files = filter(is_inset, images_files)

        image_index = map(lambda x: x.split('/')[-1].split('.')[0], images_files)
        annotations_files = map(lambda x: x.replace('images', 'annotations').replace('jpg', 'txt'), images_files)

        return images_files, image_index, annotations_files

    def _bbGt(self, objs):

        objs_list = []
        for i, obj in enumerate(objs):
            obj = obj.split(' ')
            obj_dict = dict()
            # TODO:keys!!!
            # obj_dict = dict(zip(keys, [id, pos, occl, lock, posv]))
            obj_dict['lbl'] = obj[0]
            pos = [int(obj[j]) for j in xrange(1, 5)]
            obj_dict['pos'] = pos
            obj_dict['occ'] = int(obj[5])
            posv = [int(obj[j]) for j in xrange(6, 10)]
            obj_dict['posv'] = posv
            obj_dict['ign'] = int(obj[10])
            obj_dict['ang'] = int(obj[11])
            objs_list.append(obj_dict)

        del objs
        objs = objs_list
        # only keep objects whose lbl is in lbls or ilbls
        # objs_new = [obj for obj in objs if obj['lbl'] in [lbls, ilbls]]
        objs = list(filter(lambda obj: obj['lbl'] in [self._lbls, self._ilbls], objs))
        # TODO: changed to lambda
        if self._ilbls is not None:
            for i in range(len(objs)):
                objs[i]['ign'] = False
                objs[i]['ign'] = objs[i]['ign'] or objs[i]['lbl'] in self._ilbls
        if self._hRng is not None:
            for i in range(len(objs)):
                v = objs[i]['pos'][3]
                objs[i]['ign'] = objs[i]['ign'] or v < self._hRng[0] or v > self._hRng[1]
        if self._vRng is not None:
            for i in range(len(objs)):
                bb = objs[i]['pos']
                bbv = objs[i]['posv']
                if (not objs[i]['occ']) or sum(bbv) == 0:
                    v = 1
                elif bbv == bb:
                    v = 0
                else:
                    v = (bbv[2] * bbv[3]) * 1.0 / (bb[2] * bb[3])
                objs[i]['ign'] = objs[i]['ign'] or v < self._vRng[0] or v > self._vRng[1]
        if self._squarify is not None:
            for i in range(len(objs)):
                if not objs[i]['ign']:
                    d = objs[i]['pos'][3] * self._squarify[0] - objs[i]['pos'][2]
                    objs[i]['pos'][0] = objs[i]['pos'][0] - float(d) / 2
                    objs[i]['pos'][2] = objs[i]['pos'][2] + d
        # bbs - [nx5] array containing ground truth bbs [x y w h ignore]
        # bbs = []
        return objs

    def _load_caltech_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of caltechPerson.
        """
        set, V, frame = index.split('_')
        annotation_file = '{}/{}/annotations/{}.txt'.format(self._devkit_path, self._image_set, index)

        with open(annotation_file) as f:
            objs = [x.strip() for x in f.readlines()]
        # delete the first line '% bbGt version=3'
        del objs[0]

        objs = self._bbGt(objs)
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        gt_ignores = np.zeros((num_objs), dtype=np.uint16)
        gt_lbl = []
        # "Seg" area here is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(obj['pos'][0])
            y1 = float(obj['pos'][1])
            x2 = float(obj['pos'][0] + obj['pos'][2] - 1)
            y2 = float(obj['pos'][1] + obj['pos'][3] - 1)
            if x1 < 0 or x1 > 640:
                x1 = 0
            if x2 < 0 or x2 > 640:
                x2 = 640
            if y1 < 0 or y1 > 480:
                y1 = 0
            if y2 < 0 or y2 > 480:
                y2 = 480
            if x2 < x1:
                x2 = x1 + 1
                obj['ign'] = 1
            if y2 < y1:
                y2 = y1 + 1
                obj['ign'] = 1
            assert x2 >= x1
            assert y2 >= y1
            cls = self._class_to_ind['pedestrian']
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            gt_ignores[ix] = obj['ign']
            gt_lbl.append(obj['lbl'])
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas,
                'gt_ignores': gt_ignores,
                'gt_lbl': gt_lbl}

    def caltech_to_coco_train(self):
        coco_dict = dict()
        coco_dict[u'images'] = []
        coco_dict[u'annotations'] = []
        anno_id = 1
        print('Num of Caltech Images: ', len(self.image_index))

        for image_id, idx in enumerate(self.image_index):
            print('---', image_id, '---', len(self._annotations_files))
            anno = self._load_caltech_annotation(idx)
            if anno['boxes'].shape[0] == 0 or len(anno['gt_ignores'])-sum(anno['gt_ignores']) == 0:
                continue

            image_name = '{}.jpg'.format(idx)
            image_file = self.image_path_from_index(idx)
            img = cv2.imread(image_file, 0)
            height, width = img.shape
            # check_anno(anno, image_file)

            # gt_inds = np.where(anno['gt_classes'] != 0)[0]
            # gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
            # gt_boxes[:, 0:4] = anno['boxes'][gt_inds, :]

            # images
            image_dict = {u'file_name': image_name.decode('utf-8'),
                          u'height': height,
                          u'id': image_id,
                          u'width': width}
            coco_dict[u'images'].append(image_dict)

            # annotations
            for j in range(len(anno['boxes'])):
                if anno['gt_ignores'][j] == 1:
                    continue
                x1, y1, x2, y2 = anno['boxes'][j]
                bbox = [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]  # [x,y,width,height]
                annotation_dict = {u'segmentation': [[]],
                                   u'area': (x2 - x1 + 1) * (y2 - y1 + 1),
                                   u'iscrowd': 0,  #
                                   u'image_id': image_id,
                                   u'bbox': bbox,
                                   u'category_id': self._class_to_ind['pedestrian'],
                                   u'id': anno_id}
                coco_dict[u'annotations'].append(annotation_dict)
                anno_id += 1

        ###
        coco_info_dict = {u'contributor': u'Caltech',
                          u'date_created': u'2018-04-06 16:00:00',
                          u'description': u'This is COCO Dataset version of Caltech.',
                          u'url': u'http://None',
                          u'version': u'1.0',
                          u'year': 2018}

        coco_type = u'instances'
        coco_categories = [{u'id': 1, u'name': u'pedestrian', u'supercategory': u'pedestrian'}]

        coco_dict[u'info'] = coco_info_dict
        coco_dict[u'categories'] = coco_categories
        coco_dict[u'type'] = coco_type

        print('{} pedestrians.'.format(anno_id-1))
        return coco_dict

    def caltech_to_coco_test(self, vis=False):
        coco_dict = dict()
        coco_dict[u'images'] = []
        coco_dict[u'annotations'] = []
        anno_id = 1
        print('Num of Caltech Images: ', len(self.image_index))

        for image_id, idx in enumerate(self.image_index):
            print('---', image_id, '---', len(self._annotations_files))
            anno = self._load_caltech_annotation(idx)
            # if anno['boxes'].shape[0] == 0 or len(anno['gt_ignores'])-sum(anno['gt_ignores']) == 0:
            #    continue

            image_name = '{}.jpg'.format(idx)
            image_file = self.image_path_from_index(idx)
            img = cv2.imread(image_file, 0)
            height, width = img.shape
            if vis and anno['boxes'].shape[0] != 0:
                check_anno(anno, image_file)

            # gt_inds = np.where(anno['gt_classes'] != 0)[0]
            # gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
            # gt_boxes[:, 0:4] = anno['boxes'][gt_inds, :]

            # images
            image_dict = {u'file_name': image_name.decode('utf-8'),
                          u'height': height,
                          u'id': image_id,
                          u'width': width}
            coco_dict[u'images'].append(image_dict)

            # annotations
            for j in range(len(anno['boxes'])):
                if anno['gt_ignores'][j] == 1:
                    continue
                x1, y1, x2, y2 = anno['boxes'][j]
                bbox = [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]  # [x,y,width,height]
                annotation_dict = {u'segmentation': [[]],
                                   u'area': (x2 - x1 + 1) * (y2 - y1 + 1),
                                   u'iscrowd': 0,  #
                                   u'image_id': image_id,
                                   u'bbox': bbox,
                                   u'category_id': self._class_to_ind['pedestrian'],
                                   u'id': anno_id}
                coco_dict[u'annotations'].append(annotation_dict)
                anno_id += 1

        ###
        coco_info_dict = {u'contributor': u'Caltech',
                          u'date_created': u'2018-04-06 16:00:00',
                          u'description': u'This is COCO Dataset version of Caltech.',
                          u'url': u'http://None',
                          u'version': u'1.0',
                          u'year': 2018}

        coco_type = u'instances'
        coco_categories = [{u'id': 1, u'name': u'pedestrian', u'supercategory': u'pedestrian'}]

        coco_dict[u'info'] = coco_info_dict
        coco_dict[u'categories'] = coco_categories
        coco_dict[u'type'] = coco_type

        print('{} pedestrians.'.format(anno_id-1))
        return coco_dict


if __name__ == '__main__':
    print('Convert Caltech Data to COCO Format...')
    caltech_root = '/media/tianliang/Projects/Caffe2_Projects/detectron-data/pedestrian_datasets/data-USA'
    image_set = 'train'
    trainval_set = ['set{:0>2}'.format(i) for i in range(0, 6)]
    train_set = ['set{:0>2}'.format(i) for i in range(0, 5)]
    val_set = ['set{:0>2}'.format(i) for i in range(5, 6)]
    test_set = ['set{:0>2}'.format(i) for i in range(6, 11)]

    # caltech = caltech_to_coco(image_set, train_set, caltech_root)
    # coco_dict_train = caltech.caltech_to_coco_train()
    #
    # f = open('/home/tianliang/SSAP2/Cloud/Caffe2_projects/detectron/lib/datasets/data/'
    #          'pedestrian_datasets/data-USA/json_annotations/caltech_train.json', 'w')
    # f.write(json.dumps(coco_dict_train))
    # f.close()
    #
    # caltech = caltech_to_coco(image_set, val_set, caltech_root)
    # coco_dict_val = caltech.caltech_to_coco_train()
    #
    # f = open('/home/tianliang/SSAP2/Cloud/Caffe2_projects/detectron/lib/datasets/data/'
    #          'pedestrian_datasets/data-USA/json_annotations/caltech_val.json', 'w')
    # f.write(json.dumps(coco_dict_val))
    # f.close()
    #
    caltech = caltech_to_coco(image_set, trainval_set, caltech_root)
    coco_dict_trainval = caltech.caltech_to_coco_train()
    f = open("{}/json_annotations/caltech_trainval.json".format(caltech_root), 'w')
    f.write(json.dumps(coco_dict_trainval))
    f.close()

    caltech = caltech_to_coco('test', test_set, caltech_root)
    coco_dict_test = caltech.caltech_to_coco_test(vis=False)
    f = open("{}/json_annotations/caltech_test.json".format(caltech_root), 'w')
    f.write(json.dumps(coco_dict_test))
    f.close()


    print('Done.')