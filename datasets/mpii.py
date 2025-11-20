import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import copy
from configs.paths import dataset_root
from .base import BASE


class MPII(BASE):
    def __init__(self, split='train', downsample=1, **kwargs):
        super(MPII, self).__init__(**kwargs)
        assert split[:5] == 'train'

        self.ds_name = 'mpii'
        self.split = split
        self.dataset_path = os.path.join(dataset_root, 'mpii')
        if self.human_type == 'smpl':
            if split == 'train_opt':
                annots_path = os.path.join(self.dataset_path, 'MPII_CHMR_SMPL_OPT.npz')
            else:
                annots_path = os.path.join(self.dataset_path, 'MPII_CHMR_SMPL.npz')
        elif self.human_type == 'smplx':
            annots_path = os.path.join(self.dataset_path, 'MPII_NeuralAnnot_SMPLX.npz')
        self.annots = np.load(annots_path, allow_pickle=True)['annots'][()]
        if downsample != 1:
            self.annots = {
                k: v for idx, (k, v) in enumerate(self.annots.items()) 
                if idx % downsample == 0
            }
        self.img_names = list(self.annots.keys())
        
    def __len__(self):
        return len(self.img_names)
    
    def get_raw_data(self, idx):
        img_id = idx%len(self.img_names)
        img_name = self.img_names[img_id]
        annots = copy.deepcopy(self.annots[img_name])
        if self.human_type == 'smpl':
            img_path = os.path.join(self.dataset_path,'images',img_name)
        elif self.human_type == 'smplx':
            img_path = os.path.join(self.dataset_path,img_name)

        pnum = len(annots)
        cam_rot = torch.eye(3,3).repeat(pnum,1,1).float()
        cam_trans = torch.zeros(pnum,3).float()

        betas_list = []
        poses_list = []
        transl_list = []
        cam_intrinsics_list = []
        if self.human_type == 'smplx':
            face_valid_list = []
            lhand_valid_list = []
            rhand_valid_list = []

        for i in range(pnum):
            #smpl and cam
            if self.human_type == 'smpl':
                smpl_param = annots[i]['smpl_param']
            elif self.human_type == 'smplx':
                smpl_param = annots[i]['smplx_param']
            cam_param = annots[i]['cam_param']
            cam_intrinsics = torch.tensor([
                [cam_param['focal'][0], 0., cam_param['princpt'][0]],
                [0, cam_param['focal'][1], cam_param['princpt'][1]],
                [0, 0, 1]
            ])
            betas = smpl_param['shape']
            if len(betas) == 10 and self.use_kid:
                betas = np.concatenate([betas, np.zeros(1)], axis=0)
            betas = torch.tensor(betas)
            poses = torch.tensor(smpl_param['pose'])
            transl = torch.tensor(smpl_param['trans'])
            betas_list.append(betas)
            poses_list.append(poses)
            transl_list.append(transl)
            cam_intrinsics_list.append(cam_intrinsics)

            if self.human_type == 'smplx':
                face_valid_list.append(smpl_param['face_valid'])
                lhand_valid_list.append(smpl_param['lhand_valid'])
                rhand_valid_list.append(smpl_param['rhand_valid'])

        betas = torch.stack(betas_list).float()
        poses = torch.stack(poses_list).float()
        transl = torch.stack(transl_list).float()
        cam_intrinsics = torch.stack(cam_intrinsics_list).float()

        if self.human_type == 'smpl':
            face_valid = torch.zeros(pnum).float()
            lhand_valid = torch.zeros(pnum).float()
            rhand_valid = torch.zeros(pnum).float()
        elif self.human_type == 'smplx':
            face_valid = torch.tensor(face_valid_list).float()
            lhand_valid = torch.tensor(lhand_valid_list).float()
            rhand_valid = torch.tensor(rhand_valid_list).float()

        raw_data = {
            'img_path': img_path,
            'ds': 'mpii',
            'pnum': len(betas),
            'betas': betas,
            'poses': poses,
            'transl': transl,
            'cam_intrinsics': cam_intrinsics,
            'face_valid': face_valid,
            'lhand_valid': lhand_valid,
            'rhand_valid': rhand_valid,
            'cam_rot': cam_rot,
            'cam_trans': cam_trans,
            '3d_valid': True if self.split == 'train_opt' else False,
            # '3d_valid': True,
            'detect_all_people': False
        }
        
        return raw_data
