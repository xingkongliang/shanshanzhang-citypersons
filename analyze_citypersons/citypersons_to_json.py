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
import scipy.io as sio
import tlutils.utils as utils
from six.moves import xrange

"""
CityPersons annotations
(1) data structure: 
    one image per cell
    in each cell, there are three fields: city_name; im_name; bbs (bounding box annotations)

(2) bounding box annotation format:
　　 one object instance per row:
　　 [class_label, x1,y1,w,h, instance_id, x1_vis, y1_vis, w_vis, h_vis]

(3) class label definition:
    class_label =0: ignore regions (fake humans, e.g. people on posters, reflections etc.)
    class_label =1: pedestrians
    class_label =2: riders
    class_label =3: sitting persons
    class_label =4: other persons with unusual postures
    class_label =5: group of people

(4) boxes:
　　visible boxes [x1_vis, y1_vis, w_vis, h_vis] are automatically generated from segmentation masks; 
      (x1,y1) is the upper left corner.
      if class_label==1 or 2
        [x1,y1,w,h] is a well-aligned bounding box to the full body ;
      else
        [x1,y1,w,h] = [x1_vis, y1_vis, w_vis, h_vis];
　　
"""

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_LIGHTGREEN = (144, 238, 144)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_BLUE = (0, 0, 255)
_YELLOW = (255, 255, 0)
_RED = (255, 0, 0)
_ORANGE = (255, 165, 0)

categories_dict = {1: 'pedestrian'}
categories3_dict = {'pedestrian': 1}

origin_categories_dict = {0: 'ignore_regions',
                          1: 'pedestrian',
                          2: 'riders',
                          3: 'sitting_persons',
                          4: 'other_persons',
                          5: 'group'}

origin_categories_color_dict = {0: _BLUE,
                                1: _GREEN,
                                2: _ORANGE,
                                3: _YELLOW,
                                4: _LIGHTGREEN,
                                5: _BLACK}


def check_anno(anno, im_file):
    """Draw detected bounding boxes."""
    roi = anno
    # only visualize the samples with 'gt_ignores' == 1
    # if 0 in roi['gt_ignores']:
    im = cv2.imread(im_file)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.ion()
    plt.close()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for j in range(len(roi['boxes'])):
        posv = roi['posv'][j]
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
                          edgecolor=edgecolor, linewidth=1.5)
            )
        ax.add_patch(
            plt.Rectangle((posv[0], posv[1]),
                          bbox[2] - posv[0],
                          posv[3] - posv[1], fill=False,
                          edgecolor='green', linewidth=1)
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


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    """Clip coordinates to an image with the given height and width."""
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2


class dataset_to_coco():
    def __init__(self, image_set, devkit_path):
        self._name = 'citypersons_' + image_set
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
        # self._hRng = None              # acceptable obj heights
        # self._vRng = [0.2, 1]              # acceptable obj occlusion levels
        # For CityPersons
        self._hRng = [40, np.inf]              # acceptable obj heights
        self._vRng = [0.30, 1.0]              # acceptable obj occlusion levels

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

    def _bbGt(self, objs):

        objs_list = []
        for i, obj in enumerate(objs['bbs']):
            obj_dict = dict()
            obj_dict['class_label'] = self._citypersons_classes[obj[0]]
            obj_dict['bb_pos'] = obj[1:5].astype(np.float32)
            obj_dict['instance_id'] = int(obj[5])
            obj_dict['bb_posv'] = obj[6:10].astype(np.float32)
            obj_dict['ign'] = False

            x1 = float(obj_dict['bb_pos'][0])
            y1 = float(obj_dict['bb_pos'][1])
            x2 = float(obj_dict['bb_pos'][0] + obj_dict['bb_pos'][2])
            y2 = float(obj_dict['bb_pos'][1] + obj_dict['bb_pos'][3])
            x1, y1, x2, y2 = utils.boxes.clip_xyxy_to_image(x1, y1, x2, y2, self._image_height, self._image_width)
            if x2 <= x1:
                print('x2 < x1', x1, y1, x2, y2)
                x2 = x1 + 1
                obj_dict['ign'] = 1
            if y2 <= y1:
                print('y2 < y1', x1, y1, x2, y2)
                y2 = y1 + 1
                obj_dict['ign'] = 1
            assert x2 >= x1
            assert y2 >= y1

            # for vis boxes
            xv1 = float(obj_dict['bb_posv'][0])
            yv1 = float(obj_dict['bb_posv'][1])
            xv2 = float(obj_dict['bb_posv'][0] + obj_dict['bb_posv'][2])
            yv2 = float(obj_dict['bb_posv'][1] + obj_dict['bb_posv'][3])
            xv1, yv1, xv2, yv2 = utils.boxes.clip_xyxy_to_image(xv1, yv1, xv2, yv2, self._image_height, self._image_width)
            if xv2 < xv1:
                print('xv2 < xv1', xv1, yv1, xv2, yv2)
                xv2 = xv1 + 1
            if yv2 < yv1:
                print('yv2 < yv1', xv1, yv1, xv2, yv2)
                yv2 = yv1 + 1
            assert xv2 >= xv1
            assert yv2 >= yv1

            obj_dict['boxes'] = [x1, y1, x2, y2]
            obj_dict['posv'] = [xv1, yv1, xv2, yv2]
            objs_list.append(obj_dict)

        del objs
        objs = objs_list
        # only keep objects whose lbl is in lbls or ilbls
        # objs_new = [obj for obj in objs if obj['lbl'] in [lbls, ilbls]]
        objs = list(filter(lambda obj: obj['class_label'] in self._lbls or obj['class_label'] in self._ilbls, objs))
        # TODO: changed to lambda
        # if class_label of this object in self._ilbls, set the ign=True
        if self._ilbls is not None:
            for i in range(len(objs)):
                objs[i]['ign'] = False
                objs[i]['ign'] = objs[i]['ign'] or objs[i]['class_label'] in self._ilbls
        # consider the aspect ratio
        if self._hRng is not None:
            for i in range(len(objs)):
                v = objs[i]['bb_pos'][3]
                objs[i]['ign'] = objs[i]['ign'] or v < self._hRng[0] or v > self._hRng[1]
        # consider visible part ratio
        if self._vRng is not None:
            for i in range(len(objs)):
                bb = objs[i]['bb_pos']
                bbv = objs[i]['bb_posv']
                v = float((bbv[2] * bbv[3])) / (bb[2] * bb[3])
                objs[i]['ign'] = objs[i]['ign'] or v < self._vRng[0] or v > self._vRng[1]
        if self._squarify is not None:
            for i in range(len(objs)):
                if not objs[i]['ign']:
                    d = objs[i]['bb_pos'][3] * self._squarify[0] - objs[i]['bb_pos'][2]
                    objs[i]['bb_pos'][0] = objs[i]['bb_pos'][0] - float(d) / 2
                    objs[i]['bb_pos'][2] = objs[i]['bb_pos'][2] + d
        # bbs - [nx5] array containing ground truth bbs [x y w h ignore]
        # bbs = []
        return objs

    def _load_citypersons_annotation(self, anno):
        """
        Load image and bounding boxes info from txt files of caltechPerson.
        """
        objs = self._bbGt(anno)
        num_objs = len(objs)
        bb_pos = np.zeros((num_objs, 4), dtype=np.uint16)
        bb_posv = np.zeros((num_objs, 4), dtype=np.uint16)
        posv = np.zeros((num_objs, 4), dtype=np.uint16)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        gt_ignores = np.zeros((num_objs), dtype=np.uint16)
        gt_lbl = []
        # "Seg" area here is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            cls = self._class_to_ind['pedestrian']
            bb_pos[ix, :] = obj['bb_pos']
            bb_posv[ix, :] = obj['bb_posv']
            boxes[ix, :] = obj['boxes']
            posv[ix, :] = obj['posv']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['bb_pos'][2] * obj['bb_pos'][3]
            gt_ignores[ix] = obj['ign']
            gt_lbl.append(obj['class_label'])

        return {'bb_pos': bb_pos,
                'bb_posv': bb_posv,
                'boxes': boxes,
                'posv': posv,
                'gt_classes': gt_classes,
                'seg_areas': seg_areas,
                'gt_ignores': gt_ignores,
                'gt_lbl': gt_lbl}

    def dataset_to_coco(self, is_train=False, vis=False):
        coco_dict = dict()
        coco_dict[u'images'] = []
        coco_dict[u'annotations'] = []
        anno_id = 1
        count = 0
        print('Num of Caltech Images: ', len(self.image_index))

        for image_id, idx in enumerate(self.image_index):
            print("---{}---{}---{}".format(image_id, len(self._annotations), idx))
            anno = self._load_citypersons_annotation(self._annotations[image_id])
            # if is_train and (anno['boxes'].shape[0] == 0 or len(anno['gt_ignores'])-sum(anno['gt_ignores']) == 0):
            #     continue

            image_name = '{}.png'.format(idx)
            image_file = self.image_path_from_index(idx)

            if (count % 10 == 0) and vis and anno['boxes'].shape[0] != 0:
                check_anno(anno, image_file)

            count += 1

            # images
            image_dict = {u'file_name': image_name,  # image_name.decode('utf-8'),
                          u'height': self._image_height,
                          u'id': image_id,
                          u'width': self._image_width}
            coco_dict[u'images'].append(image_dict)

            # annotations
            for j in range(len(anno['boxes'])):
                # if anno['gt_ignores'][j] == 1:
                #     continue

                x1, y1, w, h = anno['bb_pos'][j]
                bb_pos = [int(x1), int(y1), int(w), int(h)]  # [x,y,width,height]
                xv1, yv1, wv, hv = anno['bb_posv'][j]
                bb_posv = [int(xv1), int(yv1), int(wv), int(hv)]  # [x,y,width,height]
                ingore = int(anno['gt_ignores'][j])

                annotation_dict = {u'segmentation': [[]],
                                   u'area': int(anno['seg_areas'][j]),
                                   u'iscrowd': 0,  #
                                   # TODO: u'image_id': image_id+1, fix this bug
                                   u'image_id': image_id,
                                   u'bbox': bb_pos,   # [x,y,width,height]
                                   u'posv': bb_posv,  # [x,y,width,height]
                                   u'ignore': ingore,
                                   u'category_id': self._class_to_ind['pedestrian'],
                                   u'id': anno_id}
                coco_dict[u'annotations'].append(annotation_dict)
                anno_id += 1

        ###
        coco_info_dict = {u'contributor': u'CityPersons',
                          u'date_created': u'2018-04-06 16:00:00',
                          u'description': u'This is COCO Dataset version of CityPersons.',
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

    def dataset_to_coco_test(self):
        coco_dict = dict()
        coco_dict[u'images'] = []

        image_testset_path = os.path.join(self._data_path, 'leftImg8bit', self._image_set)
        assert os.path.exists(image_testset_path)
        # Test Json Format Example
        # "images": [{"file_name": "frankfurt/frankfurt_000000_000294_leftImg8bit.png",
        # "width": 2048, "id": 0, "height": 1024}]

        image_list = []
        dir_list = os.listdir(image_testset_path)
        for dir_i in dir_list:
            image_list_one = glob.glob("{}/{}/*.png".format(image_testset_path, dir_i))
            print(dir_i, len(image_list_one))
            image_list.extend(image_list_one)

        image_list = sorted(image_list)

        for image_id, image_full_path in enumerate(image_list):
            cityname = image_full_path.split('/')[-2]
            image_name = image_full_path.split('/')[-1]
            image_name_with_city = os.path.join(cityname, image_name)

            # images
            image_dict = {u'file_name': image_name_with_city.decode('utf-8'),
                          u'height': self._image_height,
                          u'id': image_id,
                          u'width': self._image_width}

            coco_dict[u'images'].append(image_dict)
        print('Num of Caltech Images: ', len(image_list))

        ###
        coco_info_dict = {u'contributor': u'CityPersons',
                          u'date_created': u'2018-04-06 16:00:00',
                          u'description': u'This is COCO Dataset version of CityPersons.',
                          u'url': u'http://None',
                          u'version': u'1.0',
                          u'year': 2018}

        coco_type = u'instances'
        coco_categories = [{u'id': 1, u'name': u'pedestrian', u'supercategory': u'pedestrian'}]

        coco_dict[u'info'] = coco_info_dict
        coco_dict[u'categories'] = coco_categories
        coco_dict[u'type'] = coco_type

        return coco_dict

    def bbs_decomposition(self, bbs):
        # [class_label, x1,y1,w,h, instance_id, x1_vis, y1_vis, w_vis, h_vis]
        n = len(bbs)
        class_label = np.empty(n)
        boxes_xywh = np.empty((n, 4), dtype=np.int32)
        instance_ids = np.empty(n, dtype=np.int32)
        boxes_xywh_vis = np.empty((n, 4), dtype=np.int32)
        for i, bb in enumerate(bbs):
            class_label[i] = bb[0]
            boxes_xywh[i, :] = bb[1:5]
            instance_ids[i] = bb[5]
            boxes_xywh_vis[i, :] = bb[6:10]
        return class_label, boxes_xywh, instance_ids, boxes_xywh_vis

    def show_dataset(self, vis=False):
        """
        Browse the entire data set.
        :return:
        """
        if vis:
            plt.ion()
            for image_id, idx in enumerate(self.image_index):
                # if image_id < 350:
                #     continue
                print("---{}---{}---{}".format(image_id, len(self._annotations), idx))
                # anno = self._load_citypersons_annotation(self._annotations[image_id])
                anno = self._annotations[image_id]
                class_label, boxes_xywh, instance_ids, boxes_xywh_vis = self.bbs_decomposition(anno['bbs'])
                boxes = utils.boxes.xywh_to_xyxy(boxes_xywh)
                boxes_vis = utils.boxes.xywh_to_xyxy(boxes_xywh_vis)
                image_name = '{}.png'.format(idx)
                image_file = self.image_path_from_index(idx)
                try:
                    img = cv2.imread(image_file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    classes_boxes = [origin_categories_dict[i] for i in class_label]
                    color_classes = [origin_categories_color_dict[i] for i in class_label]
                    # out = utils.vis.vis_one_image_opencv(img, boxes_vis, classes=classes_boxes,
                    #                                       show_box=1, show_class=0, color=color_classes)
                    out = utils.vis.vis_one_image_opencv(img, boxes, classes=classes_boxes,
                                                         show_box=1, show_class=1, color=color_classes)
                    plt.cla()
                    plt.imshow(out)
                    plt.axis('off')
                    plt.pause(3)
                except:
                    pass
        return True


if __name__ == '__main__':
    print('Convert Caltech Data to COCO Format...')
    citypersons_root = '/media/tianliang/DATA/DataSets/Cityscapes'
    annotations_dir = os.path.join(citypersons_root, 'shanshanzhang-citypersons/json_annotations')

    image_set = 'train'
    citypersons = dataset_to_coco(image_set, citypersons_root)
    citypersons.show_dataset(vis=False)
    coco_dict_trainval = citypersons.dataset_to_coco(is_train=True, vis=False)
    f = open(os.path.join(annotations_dir, 'citypersons_o30h40_train.json'), 'w')
    f.write(json.dumps(coco_dict_trainval))
    f.close()

    # image_set = 'val'
    # citypersons = dataset_to_coco(image_set, citypersons_root)
    # citypersons.show_dataset(vis=False)
    # coco_dict_val = citypersons.dataset_to_coco(is_train=False, vis=False)
    # f = open(os.path.join(annotations_dir, 'citypersons_val.json'), 'w')
    # f.write(json.dumps(coco_dict_val))
    # f.close()
    #
    # image_set = 'test'
    # citypersons = dataset_to_coco(image_set, citypersons_root)
    # coco_dict_test = citypersons.dataset_to_coco_test()
    # f = open(os.path.join(annotations_dir, 'citypersons_test.json'), 'w')
    # f.write(json.dumps(coco_dict_test))
    # f.close()

    print('Done.')