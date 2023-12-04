from pathlib import Path
import cv2
from torch.utils.data import Dataset
import SimpleITK
from segment_anything.modeling.mask_decoder import MLP, MaskDecoder
import numpy as np
from torch.utils.data import DataLoader
from importlib import import_module
from segment_anything import sam_model_registry
import torch


class Fetal_dataset(Dataset):
    def __init__(self, list_dir, transform=None):
        self.transform = transform  # using transform in torch!
        images = [SimpleITK.GetArrayFromImage(pkg2.resize_image_itk(SimpleITK.ReadImage(str(i)), (512, 512, 3))) for i
                  in
                  list_dir[0]]
        labels = [SimpleITK.GetArrayFromImage(pkg2.resize_image_itk(SimpleITK.ReadImage(str(i)), (512, 512))) for i in
                  list_dir[1]]
        self.images = np.array(images)
        self.labels = np.array(labels)
        print("-" * 20)
        print(self.images.shape)
        print(self.labels.shape)
        print("-" * 20)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, mask = pkg2.correct_dims(self.images[idx].transpose((1, 2, 0)), self.labels[idx])
        sample = {}
        if self.transform:
            image, mask, low_mask = self.transform(image, mask)

        sample['image'] = image
        sample['low_res_label'] = low_mask.unsqueeze(0)
        sample['label'] = mask.unsqueeze(0)

        return sample


if __name__ == "__main__":
    pkg2 = import_module('data_us')

    tf_val = pkg2.JointTransform2D(img_size=512, low_img_size=128, ori_size=256, crop=None, p_flip=0.0,
                                   color_jitter_params=None,
                                   long_mask=True)

    root_path = Path('./testdataset')
    image_files = np.array([(root_path / Path("image_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 701)])
    label_files = np.array([(root_path / Path("label_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 701)])

    num_classes = 2
    batch_size = 4
    multimask_output = True
    device = 0
    devices = [0, ]
    fold_n = 4

    with open("./test.txt", "r") as file:
        lines = file.readlines()
    test_index = [int(line.strip().split("/")[-1]) - 1 for line in lines]
    # print(test_index)
    print(len(test_index))

    db_test = Fetal_dataset(transform=tf_val,
                            list_dir=(image_files[np.array(test_index)], label_files[np.array(test_index)]))

    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    sam, img_embedding_size = sam_model_registry['vit_h'](image_size=512,
                                                          num_classes=8,
                                                          checkpoint='./checkpoints/sam_vit_h_4b8939.pth',
                                                          pixel_mean=[0, 0, 0],
                                                          pixel_std=[1, 1, 1])
    pkg = import_module('sam_lora_image_encoder')
    net = pkg.LoRA_Sam(sam, 4).to(device)

    net.sam.mask_decoder.iou_prediction_head = MLP(
        net.sam.mask_decoder.transformer_dim, 256, num_classes + 1, 3
    )
    net.sam.mask_decoder = MaskDecoder(transformer=net.sam.mask_decoder.transformer,
                                       transformer_dim=net.sam.mask_decoder.transformer_dim,
                                       num_multimask_outputs=num_classes
                                       )

    net.load_lora_parameters('./ckpts/epoch_19dice_0.027320221.pth')
    net = torch.nn.DataParallel(net, device_ids=devices)

    model = net
    model.to(device)
    print("The length of train set is: {}".format(len(db_test)))

    model.eval()
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(testloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, 1, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            low_res_label_batch = low_res_label_batch.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = model(image_batch, multimask_output, 512)  # masks [b,3,h,w]
                labels = outputs['masks'].argmax(dim=1)  # masks [b,h,w]
            for i in range(labels.shape[0]):
                label = labels[i].detach().cpu().numpy()
                label = pkg2.resize_image_itk(SimpleITK.GetImageFromArray(label), (256, 256))
                label = SimpleITK.GetArrayFromImage(label)

                gt = (sampled_batch['label'].squeeze(1))[i].detach().cpu().numpy()
                gt = pkg2.resize_image_itk(SimpleITK.GetImageFromArray(gt), (256, 256))
                gt = SimpleITK.GetArrayFromImage(gt)

                print(f"pred shape: {label.shape}, gt shape: {gt.shape}")
                cv2.imwrite(f"./pred/{i_batch * batch_size + i + 1}.png", label)
                cv2.imwrite(f"./gt/{i_batch * batch_size + i + 1}.png", gt)
