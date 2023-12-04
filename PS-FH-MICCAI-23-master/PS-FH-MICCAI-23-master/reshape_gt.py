import os
import SimpleITK as sitk
from importlib import import_module

import cv2

if __name__ == "__main__":
    pkg = import_module('data_us')
    gts = os.listdir("./gt")
    for gt in gts:
        gt_512 = sitk.ReadImage(os.path.join("./gt", gt))
        gt_256 = pkg.resize_image_itk(gt_512, (256, 256))
        gt_256 = sitk.GetArrayFromImage(gt_256)
        cv2.imwrite(f"./gt/{gt}",gt_256)