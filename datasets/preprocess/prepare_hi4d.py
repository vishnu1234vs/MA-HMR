import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import cv2
import glob
import torch
import smplx
import trimesh
import numpy as np


class SMPLHelper:
    def __init__(self, load_renderer=True):
        self.smpl_model = smplx.create('weights/smpl_data/smpl/SMPL_NEUTRAL.pkl', 'smpl', gender='neutral')
        self.image_shape = (940, 1280)
        if load_renderer:
            self.mesh_rasterizer = self.get_smpl_rasterizer()
        else:
            self.mesh_rasterizer = None
        
    def get_smpl_rasterizer(self):
        return pyrender.OffscreenRenderer(
            viewport_width=self.image_shape[0], 
            viewport_height=self.image_shape[1], 
            point_size=1.0
        )

    def render(self, vertices, frame, cam_params, vertices_in_world=True):
        blending_weight = 1.0
        if vertices_in_world:
            vertices = np.matmul(vertices, np.transpose(cam_params['extrinsics']['R'])) + cam_params['extrinsics']['T']
                
        vertices_to_render = vertices
        intrinsics = cam_params['intrinsics_wo_distortion']['f'].tolist() + cam_params['intrinsics_wo_distortion']['c'].tolist()
        background_image = frame

        vertex_colors = np.ones([vertices_to_render.shape[0], 4]) * [0.3, 0.3, 0.3, 1]
        tri_mesh = trimesh.Trimesh(vertices_to_render, self.smpl_model.faces,
                                   vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
        scene = pyrender.Scene(ambient_light=(0.0, 0.0, 0.0))
        scene.add(mesh, 'mesh')

        # 调整至OpenGL坐标系
        camera_pose = np.eye(4)
        rot = trimesh.transformations.euler_matrix(0, np.pi, np.pi, 'rxyz')
        camera_pose[:3, :3] = rot[:3, :3]

        camera = pyrender.IntrinsicsCamera(
          fx=intrinsics[0],
          fy=intrinsics[1],
          cx=intrinsics[2],
          cy=intrinsics[3]
        )
        scene.add(camera, pose=camera_pose)

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10)
        scene.add(light, pose=camera_pose)

        color, rend_depth = self.mesh_rasterizer.render(scene, flags=pyrender.RenderFlags.RGBA)
        blended_image = color

        if background_image is not None:
            # Rendering results needs to be rescaled to blend with background image.
            blended_image = cv2.resize(blended_image, (background_image.shape[1], background_image.shape[0]),
                interpolation=cv2.INTER_LANCZOS4)

            # Blend the rendering result with the background image.
            foreground = (rend_depth > 0) * blending_weight
            foreground = cv2.resize(
                foreground, (background_image.shape[1], background_image.shape[0]),
                interpolation=cv2.INTER_LANCZOS4)
            blended_image = (foreground[:, :, None] * blended_image
                             + (1. - foreground[:, :, None]) * background_image)
            
        return blended_image


if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, choices=['train', 'val', 'test'], default='val', help='Dataset split to process')
    args = parser.parse_args()

    vis = False
    split = args.set
    data_root = 'data/hi4d'
    split_file = os.path.join(data_root, 'train_val_test_split.npz')
    pair_names = np.load(split_file, allow_pickle=True)[split].tolist()
    annots = defaultdict(list)

    for seq_path in sorted(glob.glob(os.path.join(data_root, '*', '*'))):
        curr_pair = os.path.basename(os.path.dirname(seq_path))[len('pair'):]
        if curr_pair not in pair_names:
            continue
        cameras_path = os.path.join(seq_path, 'cameras', 'rgb_cameras.npz')
        cameras = np.load(cameras_path, allow_pickle=True)
        camera_params = {}
        for id, intrin, extrin, dist in zip(cameras['ids'], cameras['intrinsics'], cameras['extrinsics'], cameras['dist_coeffs']):
            camera_params[int(id)] = {
                'intrinsics_wo_distortion': {
                    'f': np.array([intrin[0, 0], intrin[1, 1]]).astype(np.float32), 
                    'c': np.array([intrin[0, 2], intrin[1, 2]]).astype(np.float32)
                },
                'extrinsics': {
                    'R': np.array(extrin[:3, :3]).astype(np.float32),
                    'T': np.array(extrin[:3, 3]).astype(np.float32)
                }
            }
        
        for smpl_path in tqdm(sorted(glob.glob(os.path.join(seq_path, 'smpl', '*.npz'))), desc=curr_pair, ncols=80):
            smpl_data = np.load(smpl_path, allow_pickle=True)
            meta_path = os.path.join(*smpl_path.split('/')[:-2], 'meta.npz')
            meta_data = np.load(meta_path, allow_pickle=True)
            pnum = smpl_data['betas'].shape[0]
            if vis:
                import pyrender
                smpl_helper = SMPLHelper()
                smpl_params = {
                    'betas': torch.from_numpy(smpl_data['betas']).float(), 
                    'global_orient': torch.from_numpy(smpl_data['global_orient']).float(),
                    'body_pose': torch.from_numpy(smpl_data['body_pose']).float(),
                    'transl': torch.from_numpy(smpl_data['transl']).float()
                }
                verts = smpl_helper.smpl_model(**smpl_params).vertices.detach().numpy()
                for cam_id in camera_params:
                    img_fn = os.path.join(seq_path, 'images', str(cam_id), os.path.basename(smpl_path).replace('.npz', '.jpg'))
                    frame = cv2.imread(img_fn)
                    rendered = smpl_helper.render(vertices=verts[0], frame=frame, cam_params=camera_params[cam_id], vertices_in_world=True)
                    cv2.imwrite('data/hi4d/rendered_{}.jpg'.format(cam_id), rendered)
                exit(0)
            else:
                for cam_id in camera_params:
                    img_fn = os.path.join(seq_path, 'images', str(cam_id), os.path.basename(smpl_path).replace('.npz', '.jpg'))
                    assert os.path.isfile(img_fn), 'Image not found: {}'.format(img_fn)
                    for p_id in range(pnum):
                        annots[img_fn[len(data_root)+1:]].append({
                            'focal': camera_params[cam_id]['intrinsics_wo_distortion']['f'],
                            'princpt': camera_params[cam_id]['intrinsics_wo_distortion']['c'],
                            'cam_rot': camera_params[cam_id]['extrinsics']['R'],
                            'cam_trans': camera_params[cam_id]['extrinsics']['T'],
                            'global_orient': smpl_data['global_orient'][p_id],
                            'body_pose': smpl_data['body_pose'][p_id],
                            'transl': smpl_data['transl'][p_id],
                            'betas': smpl_data['betas'][p_id],
                            'genders': meta_data['genders'][p_id].item()
                        })

    annots = dict(annots)
    annot_path = 'data/hi4d/hi4d_smpl_{}.npz'.format(split)
    np.savez(annot_path, annots=annots)
    print('Annotations saved at data/hi4d/hi4d_smpl_{}.npz'.format(split))
