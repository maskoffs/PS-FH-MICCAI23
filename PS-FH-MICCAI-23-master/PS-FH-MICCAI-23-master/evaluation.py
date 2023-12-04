import json
import math
import os
import cv2
import SimpleITK
import numpy as np

from ellipse import drawline_AOD
from pathlib import Path
from tqdm import tqdm

# 是否开启本地测试
IS_LOCAL_TEST = False


class Evaluation:
    def __init__(self, preds_dir, truths_dir, output_path):
        self.predictions_path = Path(preds_dir)
        self.ground_truth_path = Path(truths_dir)
        self.output_path = Path(output_path)
        self.results = []

    def load_image(self, image_path) -> SimpleITK.Image:
        image = SimpleITK.ReadImage(str(image_path))
        return image

    def evaluation(self, pred: SimpleITK.Image, label: SimpleITK.Image):
        # 计算aop
        pred_aop = self.cal_aop(pred)
        label_aop = self.cal_aop(label)
        aop = abs(pred_aop - label_aop)

        result = dict()
        result['aop'] = float(aop)
        # 计算耻骨指标
        pred_data_ps = SimpleITK.GetArrayFromImage(pred)
        pred_data_ps[pred_data_ps == 2] = 0
        pred_ps = SimpleITK.GetImageFromArray(pred_data_ps)

        label_data_ps = SimpleITK.GetArrayFromImage(label)
        label_data_ps[label_data_ps == 2] = 0
        label_ps = SimpleITK.GetImageFromArray(label_data_ps)
        if (pred_data_ps == 0).all():
            result['asd_ps'] = 100.0
            result['dice_ps'] = 0.0
            result['hd_ps'] = 100.0
        else:
            result['asd_ps'] = float(self.cal_asd(pred_ps, label_ps))
            result['dice_ps'] = float(self.cal_dsc(pred_ps, label_ps))
            result['hd_ps'] = float(self.cal_hd(pred_ps, label_ps))

        # 计算胎头指标
        pred_data_head = SimpleITK.GetArrayFromImage(pred)
        pred_data_head[pred_data_head == 1] = 0
        pred_data_head[pred_data_head == 2] = 1
        pred_head = SimpleITK.GetImageFromArray(pred_data_head)

        label_data_head = SimpleITK.GetArrayFromImage(label)
        label_data_head[label_data_head == 1] = 0
        label_data_head[label_data_head == 2] = 1
        label_head = SimpleITK.GetImageFromArray(label_data_head)

        if (pred_data_head == 0).all():
            result['asd_fh'] = 100.0
            result['dice_fh'] = 0.0
            result['hd_fh'] = 100.0
        else:
            result['asd_fh'] = float(self.cal_asd(pred_head, label_head))
            result['dice_fh'] = float(self.cal_dsc(pred_head, label_head))
            result['hd_fh'] = float(self.cal_hd(pred_head, label_head))

        # 计算总体指标
        pred_data_all = SimpleITK.GetArrayFromImage(pred)
        pred_data_all[pred_data_all == 2] = 1
        pred_all = SimpleITK.GetImageFromArray(pred_data_all)

        label_data_all = SimpleITK.GetArrayFromImage(label)
        label_data_all[label_data_all == 2] = 1
        label_all = SimpleITK.GetImageFromArray(label_data_all)
        if (pred_data_all == 0).all():
            result['asd_all'] = 100.0
            result['dice_all'] = 0.0
            result['hd_all'] = 100.0
        else:
            result['asd_all'] = float(self.cal_asd(pred_all, label_all))
            result['dice_all'] = float(self.cal_dsc(pred_all, label_all))
            result['hd_all'] = float(self.cal_hd(pred_all, label_all))
        return result

    def process(self):
        metrics = dict()
        predictions_path = str(self.predictions_path)
        ground_truth_path = str(self.ground_truth_path)
        names = sorted(os.listdir(predictions_path))
        for pre_name in tqdm(names):
            truth_name = pre_name
            pre_image = self.load_image(predictions_path + "/" + pre_name)
            truth_image = self.load_image(ground_truth_path + "/" + truth_name)
            result = self.evaluation(pre_image, truth_image)
            self.results.append(result)
        score, self.aggregates = self.cal_score(self.results)

        metrics["aggregates"] = self.aggregates
        metrics["score"] = score
        print(self.aggregates, score)
        with open(self.output_path, "w") as f:
            f.write(json.dumps(metrics))

    def cal_asd(self, a, b):
        filter1 = SimpleITK.SignedMaurerDistanceMapImageFilter()  # 于计算二值图像中像素到最近非零像素距离的算法
        filter1.SetUseImageSpacing(True)  # 计算像素距离时要考虑像素之间的间距
        filter1.SetSquaredDistance(False)  # 计算距离时不要对距离进行平方处理
        a_dist = filter1.Execute(a)
        a_dist = SimpleITK.GetArrayFromImage(a_dist)
        a_dist = np.abs(a_dist)
        a_edge = np.zeros(a_dist.shape, a_dist.dtype)
        a_edge[a_dist == 0] = 1
        a_num = np.sum(a_edge)

        filter2 = SimpleITK.SignedMaurerDistanceMapImageFilter()
        filter2.SetUseImageSpacing(True)
        filter2.SetSquaredDistance(False)
        b_dist = filter2.Execute(b)

        b_dist = SimpleITK.GetArrayFromImage(b_dist)
        b_dist = np.abs(b_dist)
        b_edge = np.zeros(b_dist.shape, b_dist.dtype)
        b_edge[b_dist == 0] = 1
        b_num = np.sum(b_edge)

        a_dist[b_edge == 0] = 0.0
        b_dist[a_edge == 0] = 0.0

        asd = (np.sum(a_dist) + np.sum(b_dist)) / (a_num + b_num)

        return asd

    def cal_dsc(self, pd, gt):
        pd = SimpleITK.GetArrayFromImage(pd).astype(np.uint8)
        gt = SimpleITK.GetArrayFromImage(gt).astype(np.uint8)
        y = (np.sum(pd * gt) * 2 + 1) / (np.sum(pd * pd + gt * gt) + 1)
        return y

    def cal_hd(self, a, b):
        a = SimpleITK.Cast(SimpleITK.RescaleIntensity(a), SimpleITK.sitkUInt8)
        b = SimpleITK.Cast(SimpleITK.RescaleIntensity(b), SimpleITK.sitkUInt8)
        filter1 = SimpleITK.HausdorffDistanceImageFilter()
        filter1.Execute(a, b)
        hd = filter1.GetHausdorffDistance()
        return hd

    def onehot_to_mask(self, mask):
        ret = np.zeros([3, 256, 256])
        tmp = mask.copy()
        tmp[tmp == 1] = 255
        tmp[tmp == 2] = 0
        ret[1] = tmp
        tmp = mask.copy()
        tmp[tmp == 2] = 255
        tmp[tmp == 1] = 0
        ret[2] = tmp
        b = ret[0]
        r = ret[1]
        g = ret[2]
        ret = cv2.merge([b, r, g])
        mask = ret.transpose([0, 1, 2])
        return mask

    def cal_aop(self, pred):
        aop = 0.0
        ellipse = None
        ellipse2 = None
        pred_data = SimpleITK.GetArrayFromImage(pred)
        aop_pred = np.array(self.onehot_to_mask(pred_data)).astype(np.uint8)
        contours, _ = cv2.findContours(cv2.medianBlur(aop_pred[:, :, 1], 1), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        contours2, _ = cv2.findContours(cv2.medianBlur(aop_pred[:, :, 2], 1), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        maxindex1 = 0
        maxindex2 = 0
        max1 = 0
        max2 = 0
        flag1 = 0
        flag2 = 0
        for j in range(len(contours)):
            if contours[j].shape[0] > max1:
                maxindex1 = j
                max1 = contours[j].shape[0]
            if j == len(contours) - 1:
                approxCurve = cv2.approxPolyDP(contours[maxindex1], 1, closed=True)
                if approxCurve.shape[0] > 5:
                    ellipse = cv2.fitEllipse(approxCurve)
                flag1 = 1
        for k in range(len(contours2)):
            if contours2[k].shape[0] > max2:
                maxindex2 = k
                max2 = contours2[k].shape[0]
            if k == len(contours2) - 1:
                approxCurve2 = cv2.approxPolyDP(contours2[maxindex2], 1, closed=True)
                if approxCurve2.shape[0] > 5:
                    ellipse2 = cv2.fitEllipse(approxCurve2)
                flag2 = 1
        if flag1 == 1 and flag2 == 1 and ellipse2 != None and ellipse != None:
            aop = drawline_AOD(ellipse2, ellipse)
        return aop

    def cal_score(self, result):
        m = len(result)
        dice_all_score = 0.
        dice_fh_score = 0.
        dice_ps_score = 0.
        aop_score = 0.
        hd_ps_score = 0.
        hd_all_score = 0.
        hd_fh_score = 0.
        asd_all_score = 0.
        asd_fh_score = 0.
        asd_ps_score = 0.
        for i in range(m):
            # dice
            dice_all_score += float(result[i].get("dice_all"))
            dice_ps_score += float(result[i].get("dice_ps"))
            dice_fh_score += float(result[i].get("dice_fh"))
            # asa
            asd_all_score += float(result[i].get("asd_all"))
            asd_fh_score += float(result[i].get("asd_fh"))
            asd_ps_score += float(result[i].get("asd_ps"))
            # hd
            hd_all_score += float(result[i].get("hd_all"))
            hd_ps_score += float(result[i].get("hd_ps"))
            hd_fh_score += float(result[i].get("hd_fh"))
            # aop
            aop_score += float(result[i].get("aop"))
        dice_score = (dice_all_score + dice_ps_score + dice_fh_score) / (3 * m)
        hd_score = (hd_all_score + hd_ps_score + hd_fh_score) / (3 * m)
        asd_score = (asd_all_score + asd_ps_score + asd_fh_score) / (3 * m)
        aop_score /= m

        score =  0.25 * round(dice_score, 8) + 0.125 * (1 - round(hd_score / 100.0, 8)) + 0.125 * (1-round(
            asd_score / 100.0, 8)) + 0.5 * (1-aop_score / 180.0)

        aggregates = dict()
        aggregates['aop'] = aop_score
        aggregates['dice_ps'] = dice_ps_score / m
        aggregates['dice_fh'] = dice_fh_score / m
        aggregates['dice_all'] = dice_all_score / m
        aggregates['hd_ps'] = hd_ps_score / m
        aggregates['hd_fh'] = hd_fh_score / m
        aggregates['hd_all'] = hd_all_score / m
        aggregates['asd_ps'] = asd_ps_score / m
        aggregates['asd_fh'] = asd_fh_score / m
        aggregates['asd_all'] = asd_all_score / m

        return score, aggregates


if __name__ == "__main__":
    preds_dir = f"./pred"
    truths_dir = f"./gt"
    output_path = f"./output/result.txt"
    Evaluation(preds_dir, truths_dir, output_path).process()
