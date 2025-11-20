import random
from torch.utils.data.dataset import Dataset
import numpy as np
from .aic import AIC
from .agora import AGORA
from .bedlam import BEDLAM
from .crowdpose import CROWDPOSE
from .mpii import MPII
from .coco import COCO
from .pw3d import PW3D
from .h36m import H36M
from .rich import RICH
from .hi4d import Hi4D
from .insta import INSTA
from .fit3d import FIT3D
from .chi3d import CHI3D
from .mupots import MuPoTS
from .mpii3d import MPII3D
from .panoptic import PANOPTIC
from .harmony4d import Harmony4D
from .rh import RH

datasets_dict = {
    'bedlam': BEDLAM, 'agora': AGORA, 
    'crowdpose': CROWDPOSE, 'mpii': MPII, 'coco': COCO, 
    '3dpw': PW3D, 'h36m': H36M, 'mpii3d': MPII3D, 
    'fit3d': FIT3D, 'chi3d': CHI3D, 'rich': RICH, 
    'harmony4d': Harmony4D, 'hi4d': Hi4D, 
    'mupots': MuPoTS, 'panoptic': PANOPTIC, 'rh': RH,
    'aic': AIC, 'insta': INSTA,
}


class MultipleDatasets(Dataset):
    def __init__(self, datasets_used, datasets_split = None, downsample_rate = None, make_same_len = False, **kwargs):
        if downsample_rate is None:
            downsample_rate = [1] * len(datasets_used)
        if datasets_split is None:
            self.dbs = [datasets_dict[ds](downsample=ds_rate, **kwargs) for ds, ds_rate in zip(datasets_used, downsample_rate)]
        else:
            self.dbs = [datasets_dict[ds](split=split, downsample=ds_rate, **kwargs) for ds, split, ds_rate in zip(datasets_used, datasets_split, downsample_rate)]
        
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in self.dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in self.dbs])
        self.make_same_len = make_same_len
        self.human_model = self.dbs[0].human_model

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            db_idx = index // self.max_db_data_num
            data_idx = index % self.max_db_data_num 
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                data_idx = random.randint(0,len(self.dbs[db_idx])-1)
            else: # before last batch: use modular
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]