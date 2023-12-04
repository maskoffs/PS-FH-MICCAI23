import cv2
import os
import SimpleITK as sitk
import numpy as np


root_path = r'E:\data\Pubic Symphysis-Fetal Head Segmentation and Angle of Progression'
out_img_pth = os.path.join(root_path, 'image_png')
out_label_pth = os.path.join(root_path, 'label_png')
visual_pth = os.path.join(root_path, 'visual_check')
if not os.path.exists(out_img_pth):
    os.makedirs(out_img_pth)
if not os.path.exists(out_label_pth):
    os.makedirs(out_label_pth)
if not os.path.exists(visual_pth):
    os.makedirs(visual_pth)
cases = [i for i in os.listdir(out_img_pth.replace('png', 'mha')) if i.endswith('.mha')]
for i in cases:
    img = sitk.ReadImage(os.path.join(out_img_pth.replace('png', 'mha'), i))
    img = sitk.GetArrayFromImage(img)
    img = np.transpose(img, (1, 2, 0))
    # cv2.imwrite(os.path.join(out_img_pth, i.replace('mha', 'png')), img)
    # label = sitk.ReadImage(os.path.join(out_label_pth.replace('png', 'mha'), i))
    # label = sitk.GetArrayFromImage(label)
    # assert label.max() <= 2
    # cv2.imwrite(os.path.join(out_label_pth, i.replace('mha', 'png')), label)
    img = img[:, :, 0]
    img[img > 0] = 255
    cv2.imwrite(os.path.join(visual_pth, i.replace('mha', 'png')), img)

    pass


