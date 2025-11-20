import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import copy
from configs.paths import dataset_root
from .base import BASE
import cv2
import math
from torchvision import transforms
from PIL import Image

CROWDPOSE14_TO_J19 = [9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0, 13, 12]
BK19_TO_CROWDPOSE14 = [5, 6, 7, 8, 9 ,10, 11, 12, 13, 14, 15, 16, 0, 2]
OCH19_TO_CROWDPOSE14 = [3, 0, 4, 1, 5, 2, 9, 6, 10, 7, 11, 8, 12, 13]

class RH(BASE):
    def __init__(self, split='test', downsample=1, **kwargs):
        super(RH, self).__init__(**kwargs)
        assert split in ['train', 'val', 'test']
        assert downsample == 1

        self.ds_name = 'rh'
        self.split = split
        self.dataset_path = os.path.join(dataset_root, 'RelativeHuman')
        annots_path = os.path.join(self.dataset_path, '{}_annots.npz'.format(split))
        self.annots = np.load(annots_path, allow_pickle=True)['annots'][()]
        self.img_names = list(self.annots.keys())
        
    def __len__(self):
        return len(self.img_names)

    def process_data(self, img, raw_data, rot = 0., flip = False, scale = 1., crop = False):
        meta_data = copy.deepcopy(raw_data)
        img, M = self.process_img(img, meta_data)
        return img, meta_data, M
    
    def get_raw_data(self, idx):
        img_id=idx%len(self.img_names)
        img_name=self.img_names[img_id]
        annots=copy.deepcopy(self.annots[img_name])
        pnum = len(annots)
        img_path=os.path.join(self.dataset_path,'images',img_name)

        depth_ids = []
        j2ds = []
        vis = []
        bboxes = []

        for annot in annots:
            # j2d = np.zeros((19,3))
            j2d = np.zeros((14,3))
            if 'kp2d' in annot and annot['kp2d'] is not None:
                kp2d = np.array(annot['kp2d']).reshape((-1,3))
                if len(kp2d) == 19:
                    is_BK = len(os.path.basename(img_name).replace('.jpg',''))==7
                    if is_BK:
                        # j2d[CROWDPOSE14_TO_J19] = kp2d[BK19_TO_CROWDPOSE14]
                        j2d = kp2d[BK19_TO_CROWDPOSE14]
                    else:
                        # j2d[CROWDPOSE14_TO_J19] = kp2d[OCH19_TO_CROWDPOSE14]
                        j2d = kp2d[OCH19_TO_CROWDPOSE14]
                else:
                    assert len(kp2d) == 14
                    # j2d[CROWDPOSE14_TO_J19] = kp2d
                    j2d = kp2d
            j2ds.append(j2d)
            j2d[[6, 7], 2] = 0  # make pelvis invisible
            vis.append(j2d[:,2] > 0)
            depth_ids.append(annot['depth_id'])
            bboxes.append(annot['bbox_wb'] if 'bbox_wb' in annot else annot['bbox'])

        j2ds = torch.tensor(np.array(j2ds)[..., :2]).float()
        vis = torch.tensor(np.array(vis))
        bboxes = torch.tensor(np.array(bboxes)).float()

        depth_ids = torch.from_numpy(np.array(depth_ids))

        raw_data={
                    'img_path': img_path,
                    'pnum': pnum,
                    'ds': 'rh',
                    'j2ds': j2ds,
                    'j2ds_mask': vis.unsqueeze(-1).repeat(1, 1, 2),
                    'depth_ids': depth_ids,
                    'bbox': bboxes,
                    '3d_valid': False,
                    'detect_all_people':True,
                }
        
        return raw_data

    def __getitem__(self, index):
        raw_data = copy.deepcopy(self.get_raw_data(index))
        
        # Load original image
        ori_img = cv2.imread(raw_data['img_path'])
        img_size = torch.tensor(ori_img.shape[:2])

        if img_size[1] >= img_size[0]:
            resize_rate = self.input_size / img_size[1]
            img = cv2.resize(ori_img, dsize=(self.input_size, int(resize_rate * img_size[0])))
            img_size = torch.tensor([int(resize_rate * img_size[0]), self.input_size])
        else:
            resize_rate = self.input_size / img_size[0]
            img = cv2.resize(ori_img, dsize=(int(resize_rate * img_size[1]), self.input_size))
            img_size = torch.tensor([self.input_size, int(resize_rate * img_size[1])])
        raw_data.update({'img_size': img_size, 'resize_rate': resize_rate})
        
        array2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        patch_size = 14
        if self.use_sat:
            patch_size = 56
        # pad image to support pooling
        pad_img = np.zeros((math.ceil(img.shape[0] / patch_size) * patch_size, 
                            math.ceil(img.shape[1] / patch_size) * patch_size, 3), dtype=img.dtype)
        pad_img[:img.shape[0], :img.shape[1]] = img
        assert max(pad_img.shape[:2]) == self.input_size
        pad_img = Image.fromarray(pad_img[:,:,::-1].copy())
        norm_img = array2tensor(pad_img)

        raw_data['j2ds'] *= resize_rate
        raw_data['bbox'] *= resize_rate

        self.get_boxes(raw_data)

        map_size = (raw_data['img_size'] + 27)//28
        map_h = math.ceil(map_size[0]/2)*2
        map_w = math.ceil(map_size[1]/2)*2
        raw_data['scale_map'] = torch.zeros((map_h, map_w, 2)).flatten(0,1)

        return norm_img, raw_data

if __name__ == '__main__':
    dataset = RH(split='test')
    print(len(dataset))
    for i in range(len(dataset)):
        raw_data = dataset.get_raw_data(i)
        print(raw_data['img_path'])