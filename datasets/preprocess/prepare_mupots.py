import os
import cv2
import numpy as np
from scipy import io
from tqdm import trange
from collections import defaultdict

H36M_MAPPING = [
    14, 8, 9, 10, 11, 12, 13, 
    15, 1, 16, 0, 
    5, 6, 7, 2, 3, 4
]


if __name__ == '__main__':
    data_root = "data/mupots/MultiPersonTestSet"
    annots_all = defaultdict(list)
    vis = False

    for subj in sorted(os.listdir(data_root)):
        annot_path = os.path.join(data_root, subj, "annot.mat")
        occlu_path = os.path.join(data_root, subj, "occlusion.mat")
        annots = io.loadmat(annot_path)['annotations']
        occlusions = io.loadmat(occlu_path)['occlusion_labels']

        total_frames = annots.shape[0]
        for t in trange(total_frames, desc=subj, ncols=80):
            img_path = os.path.join(data_root, subj, 'img_{:06d}.jpg'.format(t))
            assert os.path.isfile(img_path), img_path
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            num_persons = annots[t].shape[0]
            for p in range(num_persons):
                annot_2d = np.transpose(annots[t][p][0][0][0])[H36M_MAPPING]
                annot_3d = np.transpose(annots[t][p][0][0][1])[H36M_MAPPING]
                annot_3d_univ = np.transpose(annots[t][p][0][0][2])[H36M_MAPPING]
                is_valid = annots[t][p][0][0][3].item()
                annot_vis = (occlusions[t][p].reshape(-1) == 0)
                if not is_valid:
                    continue

                if vis:
                    for j in range(annot_2d.shape[0]):
                        x, y = int(annot_2d[j][0]), int(annot_2d[j][1])
                        img = cv2.circle(img, (x, y), 3, (255, 0, 0), 2)
                        img = cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    cv2.imwrite("data/mupots/examples/keypoints.jpg", img)
                    exit(0)
                else:
                    annots_all[os.path.join(subj, 'img_{:06d}.jpg'.format(t))].append({
                        'annot_2d': annot_2d, 
                        'annot_3d': annot_3d, 
                        'annot_3d_univ': annot_3d_univ, 
                        'vis': annot_vis
                    })

    annots_all = dict(annots_all)
    for k, v in annots_all.items():
        if len(v) == 0:
            del annots_all[k]
    np.savez('data/mupots/mupots_annots.npz', annots=annots_all)
    print("Annotations saved at data/mupots/mupots_annots.npz")
