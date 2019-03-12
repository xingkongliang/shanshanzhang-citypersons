import numpy as np
import datetime
import time
from collections import defaultdict
# from . import mask as maskUtils
import copy
import matplotlib.pyplot as plt
import scipy.io as sio

class COCOeval_citypersons:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self, id_setup):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))


        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 1 if (gt['height'] < self.params.HtRng[id_setup][0] or gt['height'] > self.params.HtRng[id_setup][1]) or \
                (gt['vis_ratio'] < self.params.VisRng[id_setup][0] or gt['vis_ratio'] >= self.params.VisRng[id_setup][1]) else gt['ignore']
            # changed by tianliang, gt['vis_ratio'] >= self.params.VisRng[id_setup][1]
            # old version below
            #    ( gt['vis_ratio'] < self.params.VisRng[id_setup][0] or gt['vis_ratio'] > self.params.VisRng[id_setup][1]) else gt['ignore']

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self, id_setup):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        # print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        # print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare(id_setup)
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        computeIoU = self.computeIoU

        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        HtRng = self.params.HtRng[id_setup]
        VisRng = self.params.VisRng[id_setup]
        self.evalImgs = [evaluateImg(imgId, catId, HtRng, VisRng, maxDet)
                 for catId in catIds
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        # print('DONE (t={:0.2f}s).'.format(toc-tic))


    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]


        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')


        # compute iou between each dt and gt region
        iscrowd = [int(o['ignore']) for o in gt]
        ious = self.iou(d,g,iscrowd)
        return ious

    def iou( self, dts, gts, pyiscrowd ):
        dts = np.asarray(dts)
        gts = np.asarray(gts)
        pyiscrowd = np.asarray(pyiscrowd)
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            gx1 = gt[0]
            gy1 = gt[1]
            gx2 = gt[0] + gt[2]
            gy2 = gt[1] + gt[3]
            garea = gt[2] * gt[3]
            for i, dt in enumerate(dts):
                dx1 = dt[0]
                dy1 = dt[1]
                dx2 = dt[0] + dt[2]
                dy2 = dt[1] + dt[3]
                darea = dt[2] * dt[3]

                unionw = min(dx2,gx2)-max(dx1,gx1)
                if unionw <= 0:
                    continue
                unionh = min(dy2,gy2)-max(dy1,gy1)
                if unionh <= 0:
                    continue
                t = unionw * unionh
                if pyiscrowd[j]:
                    unionarea = darea
                else:
                    unionarea = darea + garea - t

                ious[i, j] = float(t)/unionarea
        return ious



    def evaluateImg(self, imgId, catId, hRng, vRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore']:
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0
        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        # exclude dt out of height range
        indexes = []
        dt_tmp = []
        for idx, d in enumerate(dt):
            if d['height'] >= hRng[0] / self.params.expFilter and d['height'] < hRng[1] * self.params.expFilter:
                indexes.append(idx)
                dt_tmp.append(d)
        dt = dt_tmp
        # dt = [d for d in dt if d['height'] >= hRng[0] / self.params.expFilter and d['height'] < hRng[1] * self.params.expFilter]

        dtind = np.array([int(d['id'] - dt[0]['id']) for d in dt])

        # load computed ious
        # if len(dtind) > 0:
        if len(indexes) > 0:
            # ious = self.ious[imgId, catId][dtind, :] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
            ious = self.ious[imgId, catId][indexes, :] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
            ious = ious[:, gtind]
        else:
            ious = []

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))  # ground truth 匹配是否匹配 标志位
        dtm  = np.zeros((T,D))  # detection 匹配是否匹配 标志位
        gtIg = np.array([g['_ignore'] for g in gt])  # ground truth 忽略 标志位
        dtIg = np.zeros((T,D))  # detection 忽略 标志位
        dt_error_type = -1 * np.ones((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    bstOa = iou
                    bstg = -2  # 最好匹配的 gt 序号
                    bstm = -2  # 找到了最好的匹配
                    for gind, g in enumerate(gt):
                        m = gtm[tind,gind]  # 第 gind 个 ground truth 的匹配标志位
                        # if this gt already matched, and not a crowd, continue
                        if m>0:                       # 如果这个 ground truth 被匹配了 跳到下一个ground truth
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if bstm!=-2 and gtIg[gind] == 1:  # 如果 dt 被匹配到了 gt 并且 比配到了忽略的 gt 则停止
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < bstOa:   # 继续下一个 gt 直到更好的匹配 即遇到更大的IoU
                            continue
                        # if match successful and best so far, store appropriately
                        # 如果匹配成功 并且匹配到了当前最好的结果 存储结果
                        bstOa=ious[dind,gind]         # 把最好的IoU赋值给bstOa
                        bstg = gind                   # 把当前的gt的序号 赋值给 bstg
                        if gtIg[gind] == 0:           # 如果当前gt 不是被忽略的
                            bstm = 1                  # 则将 bstm 置 1
                        else:
                            bstm = -1                 # 如果当前gt 是忽略的 则将 bstm 置 -1

                    # if match made store id of match for both dt and gt
                    if bstg ==-2:                     # 没有找到gt与之匹配 则跳出当前dt
                        continue
                    dtIg[tind,dind] = gtIg[bstg]      # 把当前检测结果标志 是否 忽略
                    dtm[tind,dind]  = gt[bstg]['id']  # 检测结果匹配的 gt 的 id
                    if bstm == 1:                     # 找到最好匹配
                        gtm[tind,bstg]     = d['id']  # gt 匹配标志位 置为 检测结果的id

            error_type = {"double detections": 0, "crowded": 1, "larger bbs": 2, "body parts": 3, "background": 4,
                          "others": 5}
            crowded_iou = 0.1
            error_index = np.where((dtIg[tind, ...] == 0) & (dtm[tind, ...] == 0))[0]
            for e_i in error_index:
                non_ignore = 1 - gtIg
                non_ignore_index = non_ignore.nonzero()[0]
                iou = ious[e_i, ...][non_ignore.nonzero()]
                if np.sum(iou >= 0.5) >= 1:
                    this_error_type = error_type["double detections"]
                elif (np.sum(iou > 0.1) >= 1) and (np.sum(iou > 0.5) < 1):
                    gt_index = iou.argmax()
                    gt_index = non_ignore_index[gt_index]
                    dt_area = dt[e_i]['bbox'][2] * dt[e_i]['bbox'][3]
                    gt_area = gt[gt_index]['bbox'][2] * gt[gt_index]['bbox'][3]
                    if gt_area >= dt_area:
                        this_error_type = error_type["body parts"]
                    else:
                        this_error_type = error_type["larger bbs"]
                elif np.sum(iou >= 0.1) < 1:
                    this_error_type = error_type["background"]
                else:
                    this_error_type = error_type["others"]
                if np.sum(iou >= crowded_iou) >= 2:
                    this_error_type = error_type["crowded"]
                dt_error_type[tind, e_i] = this_error_type
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'hRng':         hRng,
                'vRng':         vRng,
                'maxDet':       maxDet,
                'image_ids':    [imgId for d in dt],
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
                'dt_error_type': dt_error_type,
            }

    def calc_detection_voc_ae(self, prec, rec, use_07_metric=False):
        """Calculate average precisions based on evaluation code of PASCAL VOC.
        This function calculates average precisions
        from given precisions and recalls.
        The code is based on the evaluation code used in PASCAL VOC Challenge.
        Args:
            prec (list of numpy.array): A list of arrays.
                :obj:`prec[l]` indicates precision for class :math:`l`.
                If :obj:`prec[l]` is :obj:`None`, this function returns
                :obj:`numpy.nan` for class :math:`l`.
            rec (list of numpy.array): A list of arrays.
                :obj:`rec[l]` indicates recall for class :math:`l`.
                If :obj:`rec[l]` is :obj:`None`, this function returns
                :obj:`numpy.nan` for class :math:`l`.
            use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
                for calculating average precision. The default value is
                :obj:`False`.
        Returns:
            ~numpy.ndarray:
            This function returns an array of average precisions.
            The :math:`l`-th value corresponds to the average precision
            for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
            :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
        """

        n_fg_class = len(prec)
        ae = np.empty(n_fg_class)
        for l in range(n_fg_class):
            if prec[l] is None or rec[l] is None:
                ae[l] = np.nan
                continue

            if use_07_metric:
                # 11 point metric
                ae[l] = 0
                for t in np.arange(0.0, 1.1, 0.1):
                    if np.sum(rec[l] >= t) == 0:
                        p = 1
                    else:
                        p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                    ae[l] += p / 11
            else:
                # correct AP calculation
                # first append sentinel values at the end
                mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
                mrec = np.concatenate(([0], rec[l], [1]))

                mpre = np.maximum.accumulate(mpre[::-1])[::-1]

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec
                ae[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ae

    def accumulate(self, id_setup, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        # print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.fppiThrs)
        RECALL      = len([0.1*i for i in range(1, 11)])
        K           = len(p.catIds) if p.useCats else 1
        M           = len(p.maxDets)
        ys   = -np.ones((T,R,K,M)) # -1 for the precision of absent categories
        ys_double_detections_error_ae = -np.ones((T,1,K,M))
        ys_crowded_error_ae = -np.ones((T,1,K,M))
        ys_larger_bbs_error_ae = -np.ones((T,1,K,M))
        ys_body_parts_error_ae = -np.ones((T,1,K,M))
        ys_background_error_ae = -np.ones((T,1,K,M))
        ys_others_error_ae = -np.ones((T,1,K,M))

        ys_double_detections_error = -np.ones((T,RECALL,K,M))
        ys_crowded_error = -np.ones((T,RECALL,K,M))
        ys_larger_bbs_error = -np.ones((T,RECALL,K,M))
        ys_body_parts_error = -np.ones((T,RECALL,K,M))
        ys_background_error = -np.ones((T,RECALL,K,M))
        ys_others_error = -np.ones((T,RECALL,K,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = [1] #_pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]

        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)

        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs[Nk + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue

                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.

                inds = np.argsort(-dtScores, kind='mergesort')

                dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dtIds = np.concatenate([np.array(e['dtIds']) for e in E], axis=0)[inds]
                image_ids = np.concatenate([np.array(e['image_ids']) for e in E], axis=0)[inds]
                dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dt_error_type = np.concatenate([e['dt_error_type'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                error_type = {"double detections": 0, "crowded": 1, "larger bbs": 2, "body parts": 3, "background": 4,
                              "others": 5}
                double_detections_error = dt_error_type == 0
                crowded_error = dt_error_type == 1
                larger_bbs_error = dt_error_type == 2
                body_parts_error = dt_error_type == 3
                background_error = dt_error_type == 4
                others_error = dt_error_type == 5

                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                double_detections_error = np.logical_and(double_detections_error, np.logical_not(dtIg))
                crowded_error = np.logical_and(crowded_error, np.logical_not(dtIg))
                larger_bbs_error = np.logical_and(larger_bbs_error, np.logical_not(dtIg))
                body_parts_error = np.logical_and(body_parts_error, np.logical_not(dtIg))
                background_error = np.logical_and(background_error, np.logical_not(dtIg))
                others_error = np.logical_and(others_error, np.logical_not(dtIg))
                inds = np.where(dtIg == 0)[1]
                tps = tps[:, inds]
                fps = fps[:, inds]
                double_detections_error = double_detections_error[:, inds]
                crowded_error = crowded_error[:, inds]
                larger_bbs_error = larger_bbs_error[:, inds]
                body_parts_error = body_parts_error[:, inds]
                background_error = background_error[:, inds]
                others_error = others_error[:, inds]
                dt_error_type = dt_error_type[:, inds]
                dtIds = dtIds[inds]
                image_ids = image_ids[inds]

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                double_detections_error_sum = np.cumsum(double_detections_error, axis=1).astype(dtype=np.float)
                crowded_error_sum = np.cumsum(crowded_error, axis=1).astype(dtype=np.float)
                larger_bbs_error_sum = np.cumsum(larger_bbs_error, axis=1).astype(dtype=np.float)
                body_parts_error_sum = np.cumsum(body_parts_error, axis=1).astype(dtype=np.float)
                background_error_sum = np.cumsum(background_error, axis=1).astype(dtype=np.float)
                others_error_sum = np.cumsum(others_error, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                for t, (tp, fp, double_detections_error, crowded_error, larger_bbs_error, body_parts_error,
                        background_error, others_error) in enumerate(zip(tp_sum, fp_sum, double_detections_error_sum,
                                                                         crowded_error_sum, larger_bbs_error_sum,
                                                                         body_parts_error_sum, background_error_sum,
                                                                         others_error_sum)):
                    tp = np.array(tp)
                    fppi = np.array(fp)/I0
                    nd = len(tp)
                    recall = tp / npig
                    double_detections_error_rate = double_detections_error / (tp+fp)
                    crowded_error_rate = crowded_error / (tp+fp)
                    larger_bbs_error_rate = larger_bbs_error / (tp+fp)
                    body_parts_error_rate = body_parts_error / (tp+fp)
                    background_error_rate = background_error / (tp+fp)
                    others_error_rate = others_error / (tp+fp)

                    double_detections_error_ae = self.calc_detection_voc_ae([double_detections_error_rate], [recall], use_07_metric=True)
                    crowded_error_ae = self.calc_detection_voc_ae([crowded_error_rate], [recall], use_07_metric=True)
                    larger_bbs_error_ae = self.calc_detection_voc_ae([larger_bbs_error_rate], [recall], use_07_metric=True)
                    body_parts_error_ae = self.calc_detection_voc_ae([body_parts_error_rate], [recall], use_07_metric=True)
                    background_error_ae = self.calc_detection_voc_ae([background_error_rate], [recall], use_07_metric=True)

                    q = np.zeros((R,))
                    q_double_detections_error = np.zeros((RECALL,))
                    q_crowded_error = np.zeros((RECALL,))
                    q_larger_bbs_error = np.zeros((RECALL,))
                    q_body_parts_error = np.zeros((RECALL,))
                    q_background_error = np.zeros((RECALL,))
                    q_others_error = np.zeros((RECALL,))

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    double_detections_error = double_detections_error.tolist()
                    crowded_error = crowded_error.tolist()
                    larger_bbs_error = larger_bbs_error.tolist()
                    body_parts_error = body_parts_error.tolist()
                    background_error = background_error.tolist()
                    others_error = others_error.tolist()

                    recall = recall.tolist()
                    q = q.tolist()
                    q_double_detections_error = q_double_detections_error.tolist()
                    q_crowded_error = q_crowded_error.tolist()
                    q_larger_bbs_error = q_larger_bbs_error.tolist()
                    q_body_parts_error = q_body_parts_error.tolist()
                    q_background_error = q_background_error.tolist()
                    q_others_error = q_others_error.tolist()

                    for i in range(nd - 1, 0, -1):
                        if recall[i] < recall[i - 1]:
                            recall[i - 1] = recall[i]
                    inds = np.searchsorted(fppi, p.fppiThrs, side='right') - 1
                    inds_recall = np.searchsorted(recall, [0.1*i for i in range(1, 11)], side='right') - 1
                    # plt.plot(np.arange(len(double_detections_error_rate)), double_detections_error_rate, 'b-')
                    # plt.plot(np.arange(len(crowded_error_rate)), crowded_error_rate, 'r-')
                    # plt.plot(np.arange(len(larger_bbs_error_rate)), larger_bbs_error_rate, 'g-')
                    # plt.plot(np.arange(len(body_parts_error_rate)), body_parts_error_rate, 'y-')
                    # plt.plot(np.arange(len(background_error_rate)), background_error_rate, 'r--')
                    # plt.plot(inds_recall, np.zeros((len(inds_recall))), 'b*')
                    # plt.title("{}: error_rate".format(p.SetupLbl[id_setup]))
                    # plt.show()
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = recall[pi]
                        for ri, pi in enumerate(inds_recall):
                            q_double_detections_error[ri] = double_detections_error[pi]
                            q_crowded_error[ri] = crowded_error[pi]
                            q_larger_bbs_error[ri] = larger_bbs_error[pi]
                            q_body_parts_error[ri] = body_parts_error[pi]
                            q_background_error[ri] = background_error[pi]
                            q_others_error[ri] = others_error[pi]
                    except:
                        pass
                    ys[t,:,k,m] = np.array(q)
                    ys_double_detections_error_ae[t,:,k,m] = np.array(double_detections_error_ae)
                    ys_crowded_error_ae[t,:,k,m] = np.array(crowded_error_ae)
                    ys_larger_bbs_error_ae[t,:,k,m] = np.array(larger_bbs_error_ae)
                    ys_body_parts_error_ae[t,:,k,m] = np.array(body_parts_error_ae)
                    ys_background_error_ae[t,:,k,m] = np.array(background_error_ae)

                    ys_double_detections_error[t,:,k,m] = np.array(q_double_detections_error)
                    ys_crowded_error[t,:,k,m] = np.array(q_crowded_error)
                    ys_larger_bbs_error[t,:,k,m] = np.array(q_larger_bbs_error)
                    ys_body_parts_error[t,:,k,m] = np.array(q_body_parts_error)
                    ys_background_error[t,:,k,m] = np.array(q_background_error)
        self.eval = {
            'params': p,
            'counts': [T, R, K, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'TP':   ys,
            'double_detections_error_ae': ys_double_detections_error_ae,
            'crowded_error_ae': ys_crowded_error_ae,
            'larger_bbs_error_ae': ys_larger_bbs_error_ae,
            'body_parts_error_ae': ys_body_parts_error_ae,
            'background_error_ae': ys_background_error_ae,
            'double_detections_error': ys_double_detections_error,
            'crowded_error': ys_crowded_error,
            'larger_bbs_error': ys_larger_bbs_error,
            'body_parts_error': ys_body_parts_error,
            'background_error': ys_background_error,
        }
        toc = time.time()
        # print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self,id_setup, res_file):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(iouThr=None, maxDets=100 ):
            p = self.params
            iStr = " {:<18} {} @ {:<18} [ IoU={:<9} | height={:>6s} | visibility={:>6s} ] = {:0.2f}%"
            iStr_error = " {:<18} {} @ {:<18} [ IoU={:<9} | height={:>6s} | visibility={:>6s} ] = {:0.2f}% {}"
            titleStr = 'Average Miss Rate'
            typeStr = '(MR)'
            setupStr = p.SetupLbl[id_setup]
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            heightStr = '[{:0.0f}:{:0.0f}]'.format(p.HtRng[id_setup][0], p.HtRng[id_setup][1])
            occlStr = '[{:0.2f}:{:0.2f}]'.format(p.VisRng[id_setup][0], p.VisRng[id_setup][1])

            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            # dimension of precision: [TxRxKxAxM]
            s = self.eval['TP']
            double_detections_error_ae = self.eval['double_detections_error_ae']
            crowded_error_ae = self.eval['crowded_error_ae']
            larger_bbs_error_ae = self.eval['larger_bbs_error_ae']
            body_parts_error_ae = self.eval['body_parts_error_ae']
            background_error_ae = self.eval['background_error_ae']

            double_detections_error = self.eval['double_detections_error']
            crowded_error = self.eval['crowded_error']
            larger_bbs_error = self.eval['larger_bbs_error']
            body_parts_error = self.eval['body_parts_error']
            background_error = self.eval['background_error']

            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
                double_detections_error_rate = double_detections_error_ae[t]
                crowded_error_rate = crowded_error_ae[t]
                larger_bbs_error_rate = larger_bbs_error_ae[t]
                body_parts_error_rate = body_parts_error_ae[t]
                background_error_rate = background_error_ae[t]

                double_detections_error = double_detections_error[t]
                crowded_error = crowded_error[t]
                larger_bbs_error = larger_bbs_error[t]
                body_parts_error = body_parts_error[t]
                background_error = background_error[t]

            mrs = 1-s[:,:,:,mind]
            double_detections_error_rate = double_detections_error_rate[:,:,:,mind]
            crowded_error_rate = crowded_error_rate[:, :, :, mind]
            larger_bbs_error_rate = larger_bbs_error_rate[:, :, :, mind]
            body_parts_error_rate = body_parts_error_rate[:, :, :, mind]
            background_error_rate = background_error_rate[:, :, :, mind]

            double_detections_error = double_detections_error[:,:,:,mind]
            crowded_error = crowded_error[:, :, :, mind]
            larger_bbs_error = larger_bbs_error[:, :, :, mind]
            body_parts_error = body_parts_error[:, :, :, mind]
            background_error = background_error[:, :, :, mind]

            double_detections_error_rate = double_detections_error_rate[double_detections_error_rate > -1]
            crowded_error_rate = crowded_error_rate[crowded_error_rate > -1]
            larger_bbs_error_rate = larger_bbs_error_rate[larger_bbs_error_rate > -1]
            body_parts_error_rate = body_parts_error_rate[body_parts_error_rate > -1]
            background_error_rate = background_error_rate[background_error_rate > -1]

            double_detections_error = double_detections_error[double_detections_error > -1].astype(np.int)
            crowded_error = crowded_error[crowded_error > -1].astype(np.int)
            larger_bbs_error = larger_bbs_error[larger_bbs_error > -1].astype(np.int)
            body_parts_error = body_parts_error[body_parts_error > -1].astype(np.int)
            background_error = background_error[background_error > -1].astype(np.int)

            if len(mrs[mrs<2])==0:
                mean_s = -1
            else:
                mean_s = np.log(mrs[mrs<2])
                mean_s = np.mean(mean_s)
                mean_s = np.exp(mean_s)
            # index = 4  # fppi=0.1
            # index = 8  # fppi=1
            index = 8  # recall=0.9
            # index = 9  # recall=1.0
            out = [iStr.format(titleStr, typeStr,setupStr, iouStr, heightStr, occlStr, mean_s*100),
                   iStr_error.format("ERROR double_detections", "mean", setupStr, iouStr, heightStr, occlStr,
                                     np.mean(double_detections_error_rate) * 100, double_detections_error),
                   iStr_error.format("ERROR crowded", "mean", setupStr, iouStr, heightStr, occlStr,
                                     np.mean(crowded_error_rate) * 100, crowded_error),
                   iStr_error.format("ERROR larger_bbs", "mean", setupStr, iouStr, heightStr, occlStr,
                                     np.mean(larger_bbs_error_rate) * 100, larger_bbs_error),
                   iStr_error.format("ERROR body_parts", "mean", setupStr, iouStr, heightStr, occlStr,
                                     np.mean(body_parts_error_rate) * 100, body_parts_error),
                   iStr_error.format("ERROR background", "mean", setupStr, iouStr, heightStr, occlStr,
                                     np.mean(background_error_rate) * 100, background_error),
                   80 * '-']
            for out_i in out:
                print(out_i)
                res_file.write(out_i)
                res_file.write('\n')
            return out

        if not self.eval:
            raise Exception('Please run accumulate() first')
        return _summarize(iouThr=.5,maxDets=1000)

    def __str__(self):
        self.summarize()


class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value

        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.fppiThrs = np.array([0.0100,    0.0178,    0.0316,    0.0562,    0.1000,    0.1778,    0.3162,    0.5623,    1.0000])
        self.maxDets = [1000]
        self.expFilter = 1.25
        self.useCats = 1

        self.iouThrs = np.array([0.5])  # np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)

        self.HtRng = [[50, 1e5 ** 2], [50,75], [50, 1e5 ** 2], [20, 1e5 ** 2],
                      [50, 1e5 ** 2], [50, 1e5 ** 2], [50, 1e5 ** 2], [50, 1e5 ** 2],
                      [50, 1e5 ** 2], [50, 1e5 ** 2], [50, 1e5 ** 2]]
        self.VisRng = [[0.65, 1e5 ** 2], [0.65, 1e5 ** 2], [0.2,0.65], [0.2, 1e5 ** 2],
                       [0.2, 1e5 ** 2], [1, 1e5 ** 2], [0.65, 1.0], [0.2, 0.65],
                       [0, 0.65], [0.65, 0.90], [0.90, 1.0]]
        self.SetupLbl = ['Reasonable', 'Reasonable_small', 'Reasonable_occ=heavy', 'All',
                         'All-50', 'R_occ=None', 'R_occ=Partial', 'R_occ=heaavy',
                         'Heavy', 'Partial', 'Bare']


    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None