import os
import numpy as np
import cv2
import torch

image_pth = r'E:\data\PSFH_Dataset\images\train'
label_pth = r'E:\data\PSFH_Dataset\labels\train'
cases = [i for i in os.listdir(image_pth) if i.endswith('.png')]
image = np.zeros((3200, 256, 256), dtype='uint8')
label = np.zeros_like(image)
for i in range(len(cases)):
    image[i, :, :] = cv2.imread(os.path.join(image_pth, cases[i]), flags=cv2.IMREAD_GRAYSCALE)
    label[i, :, :] = cv2.imread(os.path.join(label_pth, cases[i]), flags=cv2.IMREAD_GRAYSCALE)

# input = torch.tensor(image[:, None, :, :], dtype=torch.long)
# weight = torch.tensor([[[[1, 1, 1],
#                        [1, 0, 1],
#                        [1, 1, 1]]]], dtype=torch.long)
# count = np.ones((3200, 256, 256), dtype='uint8') * 8
# count[:, [0, 255], :] = 5
# count[:, :, [0, 255]] = 5
# count[:, [0, 0, 255, 255], [0, 255, 0, 255]] = 3
# out = torch.nn.functional.conv2d(input, weight, stride=1, padding=1)
# out = out.numpy()[0]
# out = np.rint(out / count).astype('uint8')
# p = np.zeros((256, 256), 'float32')
# p[list[image.flatten()], [out.flatten()]] += 1
classes = np.unique(label)
info = np.zeros((len(classes), 3))
for i in classes:
    info[i, 0] = np.std(image[label == i])
    info[i, 1] = np.sum(label == i) / label.size
    info[i, 2] = 1 / (1 + info[i, 1])
# 0.041792888537803656, 2.678366907105043, 0.27984020435715307
# 0.24207537, 2.12425602, 0.63366862
# 0.40339167, 1.82450907, 0.77209926
# 0.6160191,  1.35035524, 1.03362566
# 0.67204101, 1.22604056, 1.10191842
# 1.53951, 1.98425, 1.88461
s = np.sum(info[:, 2])
print(info[:, 2] / s * 3)
pass
# mean = [np.mean(image[label == 0]), np.mean(image[label == 1]), np.mean(image[label == 2])]
# std = [np.std(image[label == 0]), np.std(image[label == 1]), np.std(image[label == 2])]
# data_mean = np.mean(image)
# data_std = np.std(image)
# print(data_mean, data_std)
