import os
import pickle
from tqdm.auto import tqdm
import torch
import numpy as np
from utils.render import render_side_views, render_meshes
from utils.transforms import unNormalize
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, vis_boxes, vis_sat, vis_scale_img, get_colors_rgb
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
import time
import cv2
import trimesh
# from thop import profile

def inference(model, infer_dataloader, conf_thresh, results_save_path = None,
                        distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None

    accelerator.print(f'Results will be saved at: {results_save_path}')
    os.makedirs(results_save_path,exist_ok=True)
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model

    progress_bar = tqdm(total=len(infer_dataloader), disable=not accelerator.is_local_main_process, ncols=100)
    progress_bar.set_description('inference')
    
    for itr, (samples, targets) in enumerate(infer_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():    
           outputs = model(samples, targets)
        bs = len(targets)
        for idx in range(bs):
            img_size = targets[idx]['img_size'].detach().cpu().int().numpy()
            img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]

            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()

            pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach().cpu().numpy()
            ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
            ori_img[img_size[0]:,:,:] = 255
            ori_img[:,img_size[1]:,:] = 255
            ori_img[img_size[0]:,img_size[1]:,:] = 255
            ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)

            sat_img = vis_sat(ori_img.copy(),
                                input_size = model.input_size,
                                patch_size = 14,
                                sat_dict = outputs['sat'],
                                bid = idx)[:img_size[0],:img_size[1]]
            
            faces_list = [smpl_layer.faces] * max(1, len(pred_verts))
            K_vis = outputs['pred_intrinsics'][idx][0].detach().cpu().numpy()

            fov_h = np.arctan2(K_vis[1,2], K_vis[0,0]) * 2 * 180 / np.pi
            fov = np.arctan2(model.input_size/2, K_vis[0,0]) * 2 * 180 / np.pi

            print(f'Image: {img_name}, FOV_H: {fov_h:.2f}, FOV: {fov:.2f}')

            colors = get_colors_rgb(len(pred_verts))
            # pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
            #                             verts = pred_verts,
            #                             smpl_faces = smpl_layer.faces,
            #                             cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
            #                             colors=colors)[:img_size[0],:img_size[1]]

            pred_mesh_img = render_meshes(img=ori_img.copy(),
                            l_mesh=pred_verts, 
                            l_face=faces_list, 
                            cam_param={'focal': np.asarray([K_vis[0,0],K_vis[1,1]]), 'princpt': np.asarray([K_vis[0,-1],K_vis[1,-1]])}, 
                            color=colors)

            # faces list repeated

            _, pred_sideview, pred_bev = render_side_views(ori_img.copy(), colors, pred_verts, faces_list, pred_transl, K_vis)

            # if 'enc_outputs' not in outputs:
            #     pred_scale_img = np.zeros_like(ori_img)[:img_size[0],:img_size[1]]
            # else:
            #     enc_out = outputs['enc_outputs']
            #     h, w = enc_out['hw'][idx]
            #     flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

            #     ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
            #     xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
            #     scale_map = torch.zeros((h,w,2))
            #     scale_map[ys,xs] = flatten_map

            #     pred_scale_img = vis_scale_img(img = ori_img.copy(),
            #                                     scale_map = scale_map,
            #                                     conf_thresh = model.sat_cfg['conf_thresh'],
            #                                     patch_size=28)[:img_size[0],:img_size[1]]

            # pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
            # pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
            # pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]


            cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), np.vstack([np.hstack([ori_img, pred_mesh_img]),
                                                                                        np.hstack([pred_sideview, pred_bev])]))


        progress_bar.update(1)
    

