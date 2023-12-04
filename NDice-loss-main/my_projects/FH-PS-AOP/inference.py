import os
from mmseg.apis import init_model, inference_model
from medpy import metric
import cv2
import numpy as np
import torch


def multi_class_metrics(result, reference, target=None):
    classes = np.unique(result[result > 0]) if target is None else target
    output = []
    for i in classes:
        result_copy = np.zeros_like(result)
        result_copy[result == i] = 1
        reference_copy = np.zeros_like(reference)
        reference_copy[reference == i] = 1
        dice = metric.binary.dc(result_copy, reference_copy)
        asd = metric.binary.asd(result_copy, reference_copy)
        hd = metric.binary.hd(result_copy, reference_copy)
        output.append([dice, asd, hd])
    return output


config = 'sequent_aw-dice-3.0-aw-ce-1.0_40k'
images_pth = r'E:\data\PSFH_Dataset\images\val'
labels_pth = r'E:\data\PSFH_Dataset\labels\val'
out_pth = r'E:\data\PSFH_Dataset\tta_inference_{}'.format('NCE')
if not os.path.exists(out_pth):
    os.makedirs(out_pth)

model_config = r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\work_dirs\{}\{}.py'.format(config, config)
model_weight = r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\work_dirs\{}\iter_40000.pth'.format(config)
model_weight = r'C:\Users\qiuyaoyang\PycharmProjects\FH-PS-AOP-grandchallenge\best_model_{}.pth'.format('NCE')
model = init_model(model_config, model_weight)
# torch.save(model, 'test.pt')

results = dict(class_name=[], mDice=[], mASD=[], mHD=[], score=[])
images = [i for i in os.listdir(images_pth) if i.endswith('png')]
total_metrics = {'class1':[], 'class2':[]}
for i in images:
    print(i)
    # pred = inference_model(model, os.path.join(images_pth, i))
    img = cv2.imread(os.path.join(images_pth, i))
    pred1 = inference_model(model, img)
    # pred = pred.pred_sem_seg.numpy()
    pred1 = pred1.seg_logits.data
    pred1 = torch.softmax(pred1, dim=0)
    pred2 = inference_model(model, img[:, ::-1, :])
    pred2 = pred2.seg_logits.data
    pred2 = torch.softmax(pred2, dim=0)
    pred2 = torch.flip(pred2, dims=(2,))
    pred = pred1 + pred2
    pred = torch.argmax(pred, dim=0)

    pred = pred.cpu().numpy()
    # cv2.imwrite(os.path.join(out_pth, i), pred)
    # pred = cv2.imread(os.path.join(out_pth, i), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(os.path.join(labels_pth, i), cv2.IMREAD_GRAYSCALE)
    mask = np.zeros_like(img)
    mask[:, :, 0][np.logical_and(pred == 0, label > 0)] = 255
    mask[:, :, 1][np.logical_and(pred > 0, label > 0)] = 255
    mask[:, :, 2][np.logical_and(pred > 0, label == 0)] = 255
    out = img * 0.7 + mask * 0.3
    cv2.imwrite(os.path.join(out_pth, i), out)
    metrics = multi_class_metrics(pred, label, target=[1, 2])
    total_metrics['class1'].append(metrics[0])
    total_metrics['class2'].append(metrics[1])
metric_class1 = np.array(total_metrics['class1'])
metric_class1_m = np.mean(metric_class1, axis=0)
score_1 = 0.5 * metric_class1_m[0] + 0.5 * ((1 - metric_class1_m[1] / 100) + (1 - metric_class1_m[2] / 100)) / 2
metric_class2 = np.array(total_metrics['class2'])
metric_class2_m = np.mean(metric_class2, axis=0)
score_2 = 0.5 * metric_class2_m[0] + 0.5 * ((1 - metric_class2_m[1] / 100) + (1 - metric_class2_m[2] / 100)) / 2

with open(os.path.join(out_pth, 'metrics_{}.csv'.format(config)), 'w') as f:
    f.write('class_name,mDice,mASD,mHD,score\n')
    f.write('ps,{},{},{},{}\n'.format(*metric_class1_m, score_1))
    f.write('fh,{},{},{},{}\n'.format(*metric_class2_m, score_2))
pass
