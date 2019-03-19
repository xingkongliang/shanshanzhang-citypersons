#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ : "ZhangTianliang"
# Date: 18-9-3
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
Caltech annotations


"""
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
    for j in range(len(roi['bb_pos'])):
        posv = roi['bb_posv'][j]  # x, y, w, h
        bbox = roi['bb_pos'][j]
        class_name = roi['gt_lbl'][j]
        if roi['gt_ignores'][j]:
            edgecolor = 'red'
        else:
            edgecolor = 'blue'
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2],
                          bbox[3], fill=False,
                          edgecolor=edgecolor, linewidth=3.5)
            )
        ax.add_patch(
            plt.Rectangle((posv[0], posv[1]),
                          bbox[2],
                          posv[3], fill=False,
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
    time.sleep(0.5)
    plt.close()


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    """Clip coordinates to an image with the given height and width."""
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2


class caltech_to_coco():
    def __init__(self, image_set, set, devkit_path):
        self._name = 'caltech_' + image_set
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._image_width = 640
        self._image_height = 480
        self._set = set
        self._lbls = 'person'             # return objs with these labels (or [] to return all)
        self._ilbls = ['people', 'ignore']            # return objs with these labels but set to ignore
        self._squarify = [0.41]           # controls optional reshaping of bbs to fixed aspect ratio
        self._hRng = [40, np.inf]         # acceptable obj heights
        self._vRng = [0.4, 1.0]          # acceptable obj occlusion levels

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

        images_files = list(filter(is_inset, images_files))

        image_index = list(map(lambda x: x.split('/')[-1].split('.')[0], images_files))
        annotations_files = list(map(lambda x: x.replace('images', 'annotations').replace('jpg', 'txt'), images_files))

        return images_files, image_index, annotations_files

    def _bbGt(self, objs):

        objs_list = []
        for i, obj in enumerate(objs):
            obj = obj.split(' ')
            obj_dict = dict()
            # TODO:keys!!!
            # obj_dict = dict(zip(keys, [id, pos, occl, lock, posv]))
            obj_dict['lbl'] = obj[0]
            pos = [int(float(obj[j])) for j in xrange(1, 5)]
            obj_dict['bb_pos'] = pos
            obj_dict['occ'] = int(obj[5])
            posv = [int(obj[j]) for j in xrange(6, 10)]
            obj_dict['bb_posv'] = posv
            obj_dict['ign'] = int(obj[10])
            obj_dict['ang'] = int(obj[11])

            x1 = float(obj_dict['bb_pos'][0])
            y1 = float(obj_dict['bb_pos'][1])
            x2 = float(obj_dict['bb_pos'][0] + obj_dict['bb_pos'][2])
            y2 = float(obj_dict['bb_pos'][1] + obj_dict['bb_pos'][3])
            x1, y1, x2, y2 = utils.boxes.clip_xyxy_to_image(x1, y1, x2, y2, self._image_height, self._image_width)
            assert x2 >= x1
            assert y2 >= y1

            # for vis boxes
            xv1 = float(obj_dict['bb_posv'][0])
            yv1 = float(obj_dict['bb_posv'][1])
            xv2 = float(obj_dict['bb_posv'][0] + obj_dict['bb_posv'][2])
            yv2 = float(obj_dict['bb_posv'][1] + obj_dict['bb_posv'][3])
            xv1, yv1, xv2, yv2 = utils.boxes.clip_xyxy_to_image(xv1, yv1, xv2, yv2, self._image_height, self._image_width)
            assert xv2 >= xv1
            assert yv2 >= yv1

            obj_dict['bb_pos'] = [x1, y1, x2-x1, y2-y1]
            obj_dict['bb_posv'] = [xv1, yv1, xv2-xv1, yv2-yv1]

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
                v = objs[i]['bb_pos'][3]
                objs[i]['ign'] = objs[i]['ign'] or v < self._hRng[0] or v > self._hRng[1]
        if self._vRng is not None:
            for i in range(len(objs)):
                bb = objs[i]['bb_pos']
                bbv = objs[i]['bb_posv']
                if (not objs[i]['occ']) or sum(bbv) == 0:
                    v = 1
                elif bbv == bb:
                    v = 0
                else:
                    v = (bbv[2] * bbv[3]) * 1.0 / (bb[2] * bb[3])
                assert v <= 1.0
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
        bb_pos = np.zeros((num_objs, 4), dtype=np.uint16)
        bb_posv = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
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
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = obj['bb_pos'][2] * obj['bb_pos'][3]
            gt_ignores[ix] = obj['ign']
            gt_lbl.append(obj['lbl'])
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'bb_pos': bb_pos,
                'bb_posv': bb_posv,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas,
                'gt_ignores': gt_ignores,
                'gt_lbl': gt_lbl}

    def caltech_to_coco_train(self, is_train=False, vis=False):
        coco_dict = dict()
        coco_dict[u'images'] = []
        coco_dict[u'annotations'] = []
        anno_id = 1
        count = 0
        count_using_images = 0
        print('Num of Caltech Images: ', len(self.image_index))

        for image_id, idx in enumerate(self.image_index):
            print('---', image_id, '---', len(self._annotations_files))
            anno = self._load_caltech_annotation(idx)
            if anno['bb_pos'].shape[0] == 0 or len(anno['gt_ignores'])-sum(anno['gt_ignores']) == 0:
                continue
            else:
                count_using_images += 1

            image_name = '{}.jpg'.format(idx)
            image_file = self.image_path_from_index(idx)

            if (count % 10 == 0) and vis and anno['bb_pos'].shape[0] != 0:
                check_anno(anno, image_file)

            count += 1
            # img = cv2.imread(image_file, 0)
            # height, width = img.shape
            # check_anno(anno, image_file)

            # images
            image_dict = {u'file_name': image_name,  # image_name.decode('utf-8'),
                          u'height': self._image_height,
                          u'id': image_id,
                          u'width': self._image_width}
            coco_dict[u'images'].append(image_dict)

            # annotations
            for j in range(len(anno['bb_pos'])):
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
                                   u'image_id': image_id,
                                   u'bbox': bb_pos,   # [x,y,width,height]
                                   u'posv': bb_posv,  # [x,y,width,height]
                                   u'ignore': ingore,
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

        print("Height                   : {}".format(self._hRng[0]))
        print("Visible body level       : {}-{}".format(self._vRng[0], self._vRng[1]))
        print("The number of pedestrians: {}".format(anno_id-1))
        print('Num of Caltech Images    : {}'.format(len(self.image_index)))
        print("The number of used images: {}".format(count_using_images))
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

            image_name = '{}.jpg'.format(idx)
            image_file = self.image_path_from_index(idx)
            img = cv2.imread(image_file, 0)
            height, width = img.shape
            if vis and anno['boxes'].shape[0] != 0:
                check_anno(anno, image_file)

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
    caltech_root = '/media/tianliang/DATA/DataSets/Pedestrian_Datasets/data-USA'
    image_set = 'train'
    trainval_set = ['set{:0>2}'.format(i) for i in range(0, 6)]
    train_set = ['set{:0>2}'.format(i) for i in range(0, 5)]
    val_set = ['set{:0>2}'.format(i) for i in range(5, 6)]
    test_set = ['set{:0>2}'.format(i) for i in range(6, 11)]

    caltech = caltech_to_coco(image_set, trainval_set, caltech_root)
    train_json_name = "caltech_newo{}h{}_trainval.json".format(str(int(caltech._vRng[0]*100)), str(caltech._hRng[0]))
    caltech.show_dataset(vis=False)
    coco_dict_trainval = caltech.caltech_to_coco_train(is_train=True, vis=False)
    f = open("{}/json_annotations/{}".format(caltech_root, train_json_name), 'w')
    f.write(json.dumps(coco_dict_trainval))
    f.close()

    # caltech = caltech_to_coco('test', test_set, caltech_root)
    # coco_dict_test = caltech.caltech_to_coco_test(vis=False)
    # f = open("{}/json_annotations/caltech_test.json".format(caltech_root), 'w')
    # f.write(json.dumps(coco_dict_test))
    # f.close()


    print('Done.')