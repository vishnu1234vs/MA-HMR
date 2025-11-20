import numpy as np
import os
import glob
from tqdm import tqdm
import argparse

def merge_and_restructure_npz(input_folder, output_file_path):
    npz_files = glob.glob(os.path.join(input_folder, "*opt*.npz"))

    combined_data = {}
    for file_path in tqdm(npz_files):
        data = np.load(file_path, allow_pickle=True)
        for img_path_key, original_annots in data.items():
            original_annots = original_annots.item()

            num_people = original_annots['pose_cam'].shape[0]
            
            cam_int = original_annots['cam_int'][0]
            focal = np.array([cam_int[0, 0], cam_int[1, 1]], dtype=np.float32)
            princpt = np.array([cam_int[0, 2], cam_int[1, 2]], dtype=np.float32)
            cam_param = {'focal': focal, 'princpt': princpt}
            
            new_annots_list = []
            for i in range(num_people):
                smpl_param = {
                    'shape': original_annots['shape'][i],
                    'pose': original_annots['pose_cam'][i],
                    'trans': original_annots['trans_cam'][i]
                }
                person_annot = {
                    'smpl_param': smpl_param,
                    'cam_param': cam_param
                }
                new_annots_list.append(person_annot)
            
            if 'insta' in input_folder:
                img_path_key = 'insta-train/' + img_path_key
            if img_path_key in combined_data:
                print(f"warning: repeated key '{img_path_key}', old value will be covered")
            combined_data[img_path_key] = new_annots_list


    final_combined_data = {'annots': combined_data}
    np.savez(output_file_path, **final_combined_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['insta', 'coco', 'aic', 'mpii'], default='coco', help='Dataset to process')
    args = parser.parse_args()
    data_path_dict = {
        "insta": ("data/insta", "data/insta/INSTA_CHMR_SMPL_OPT.npz"),
        "coco": ("data/coco2014", "data/coco2014/COCO_CHMR_SMPL_OPT.npz"),
        "aic": ("data/aic", "data/aic/AIC_CHMR_SMPL_OPT.npz"),
        "mpii": ("data/mpii", "data/mpii/MPII_CHMR_SMPL_OPT.npz"),
    }
    input_folder = data_path_dict[args.dataset][0]
    output_file_path = data_path_dict[args.dataset][1]
    merge_and_restructure_npz(input_folder, output_file_path)