import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
from collections import defaultdict

def convert_annot(annotation, fields, dir_depth):
    imgnames = annotation["imgname"]

    unique_imgnames, inverse_indices = np.unique(imgnames, return_inverse=True)

    group_indices = defaultdict(list)
    for i, img_idx in enumerate(inverse_indices):
        group_indices[img_idx].append(i)

    field_data = {f: np.array(annotation[f]) for f in fields if len(annotation[f]) > 0}

    annot_by_img = {}
    for img_idx, indices in group_indices.items():
        img_name = '/'.join(unique_imgnames[img_idx].split('/')[-dir_depth:])
        img_data = {}
        indices_arr = np.array(indices)

        for field in field_data.keys():
            img_data[field] = field_data[field][indices_arr]
        annot_by_img[str(img_name)] = img_data
    return annot_by_img

def merge_and_restructure_npz(input_folder, output_file_path):
    npz_files = glob.glob(os.path.join(input_folder, "*-release.npz"))

    combined_data = {}
    for file_path in tqdm(npz_files):
        data = np.load(file_path, allow_pickle=True)
        fields = list(data.keys())
        data = convert_annot(data, fields, dir_depth=data_path_dict[args.dataset][2])
        for img_path_key, original_annots in data.items():
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
        "insta": ("data/insta", "data/insta/INSTA_CHMR_SMPL.npz", 3),
        "coco": ("data/coco2014", "data/coco2014/COCO_CHMR_SMPL.npz", 1),
        "aic": ("data/aic", "data/aic/AIC_CHMR_SMPL.npz", 1),
        "mpii": ("data/mpii", "data/mpii/MPII_CHMR_SMPL.npz", 1),
    }
    input_folder = data_path_dict[args.dataset][0]
    output_file_path = data_path_dict[args.dataset][1]
    merge_and_restructure_npz(input_folder, output_file_path)