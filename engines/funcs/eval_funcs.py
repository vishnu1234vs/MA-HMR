import os
import cv2
import torch
import pickle
import zipfile
import datetime
import time
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from utils.transforms import unNormalize
from utils.constants import H36M_EVAL_JOINTS
from utils.box_ops import box_cxcywh_to_xyxy
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, vis_boxes, vis_sat, vis_scale_img, get_colors_rgb
from utils.render import render_side_views, render_meshes
from utils.evaluation import cal_3d_position_error, match_2d_greedy, get_matching_dict, compute_prf1, select_and_align, vectorize_distance
from engines.funcs.mupots_utils import match, procrustes, mpii_compute_3d_pck, _match_poses, _scale_to_gt
from engines.funcs.RH_evaluation import RH_Evaluation

def evaluate_rh(model, eval_dataloader, conf_thresh,
                        vis = True, vis_step = 40, results_save_path = None,
                        distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None
    
    os.makedirs(results_save_path, exist_ok=True)

    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    # simple step counter for visualization indexing
    step = 0

    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    
    annotations = {}
    # J_regressor_19 = torch.from_numpy(smpl_layer.J_regressor_19).float().to(cur_device)
    J_regressor_extra = torch.from_numpy(smpl_layer.J_regressor_extra).float().to(cur_device)
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=80)
    progress_bar.set_description('evaluate')

    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        bs = len(targets)
        with torch.no_grad():
            outputs = model(samples, targets)
        for idx in range(bs):
            img_name = targets[idx]['img_path'].split('/')[-1]
            output_keys = ['pred_j2ds', 'pred_j3ds', 'pred_poses', 'pred_betas', 
                           'pred_verts', 'pred_boxes', 'pred_confs', 'pred_transl']
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            output = {k: outputs[k][idx][select_queries_idx].clone() for k in output_keys}
            K = outputs['pred_intrinsics'][idx]

            pred_verts = output['pred_verts']
            pred_transl = output['pred_transl']
            pred_j3ds = output['pred_j3ds']

            # pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:,None,:], J_regressor_19]) + pred_transl[:,None,:]
            # pred_j3ds = pred_j3ds[:, [9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0, 13, 12]].clone()
            
            # pred_j3ds_extra = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:,None,:], J_regressor_extra]) + pred_transl[:,None,:]
            # pred_j3ds = torch.concat([pred_j3ds[:, :24], pred_j3ds_extra], dim=1)
            # pred_j3ds = pred_j3ds[:, [16,17,18,19,20,21,1,2,4,5,7,8,27,26]].clone()

            pred_j2ds_homo = torch.einsum('bjc,cd->bjd', pred_j3ds, K[0].transpose(0, 1))
            pred_j2ds = pred_j2ds_homo[..., :2] / (pred_j2ds_homo[..., 2:] + 1e-6)
            
            resize_rate = targets[idx]['resize_rate']
            pred_j2ds = pred_j2ds / resize_rate

            annot = {
                'kp2ds': pred_j2ds.cpu().numpy(),
                'trans': pred_transl.cpu().numpy()
            }

            annotations[img_name] = annot
            # visualization similar to evaluate_panoptic: render predicted meshes and side/bev views
            img_idx = step + accelerator.process_index * len(eval_dataloader) * bs
            if vis and (img_idx % vis_step == 0):
                try:
                    ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                    ori_img = pad_img(ori_img, model.input_size)

                    # prepare predicted verts list (numpy)
                    try:
                        pred_verts_np = [v for v in pred_verts.detach().cpu().numpy()]
                    except Exception:
                        pred_verts_np = [v for v in pred_verts]

                    colors = get_colors_rgb(len(pred_verts_np))
                    faces_list = [smpl_layer.faces] * max(1, len(pred_verts_np))

                    # use predicted intrinsics for visualization
                    try:
                        K_vis = K[0].detach().cpu().numpy()
                    except Exception:
                        K_vis = None

                    cam_param = None
                    if K_vis is not None:
                        cam_param = {'focal': np.asarray([K_vis[0,0], K_vis[1,1]]),
                                     'princpt': np.asarray([K_vis[0,2], K_vis[1,2]])}

                    pred_mesh_img = render_meshes(img=ori_img.copy(), l_mesh=pred_verts_np, l_face=faces_list, cam_param=cam_param, color=colors)

                    pred_trans_list = [t for t in pred_transl.detach().cpu().numpy()]

                    _, pred_sideview, pred_bev = render_side_views(ori_img.copy(), colors, pred_verts_np, faces_list, pred_trans_list, K_vis)

                    top_row = np.hstack([ori_img, pred_mesh_img])
                    bottom_row = np.hstack([pred_sideview, pred_bev])
                    if bottom_row.shape[1] != top_row.shape[1]:
                        pad_w = top_row.shape[1] - bottom_row.shape[1]
                        if pad_w > 0:
                            bottom_row = np.pad(bottom_row, ((0,0),(0,pad_w),(0,0)), mode='constant', constant_values=255)

                    full_img = np.vstack([top_row, bottom_row])
                    # downsample for saving
                    full_img = cv2.resize(full_img, (full_img.shape[1]//2, full_img.shape[0]//2))
                    cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)
                except Exception:
                    # don't break evaluation on visualization errors
                    accelerator.print(f'Warning: failed to visualize {img_name}')
            
            step += 1
            
        progress_bar.update(1)

    # Gather annotations from all processes
    if distributed:
        all_annotations = accelerator.gather_for_metrics([annotations])
        if accelerator.is_main_process:
            # Merge annotations from all processes
            merged_annotations = {}
            for proc_annotations in all_annotations:
                merged_annotations.update(proc_annotations)
            annotations = merged_annotations
    
    # Save only on main process
    if accelerator.is_main_process:
        npz_path = os.path.join(results_save_path, 'results.npz')
        annotations = {'results': annotations}
        np.savez(npz_path, **annotations)
        
        RH_Evaluation(npz_path, eval_dataloader.dataset.dataset_path, 'test')

# Modified from agora_evaluation
def evaluate_agora(model, eval_dataloader, conf_thresh,
                        vis = True, vis_step = 40, results_save_path = None,
                        distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None
    num_processes = accelerator.num_processes

    has_kid = ('train' in eval_dataloader.dataset.split and eval_dataloader.dataset.ds_name == 'agora')
    
    os.makedirs(results_save_path,exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    
    step = 0
    total_miss_count = 0
    total_count = 0
    total_fp = 0
    mve, mpjpe = [0.], [0.]
    # inference timing (ms) for the model forward call in agora
    inference_times = []

    if has_kid:
        kid_total_miss_count = 0
        kid_total_count = 0
        kid_mve, kid_mpjpe = [0.], [0.]

    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    body_verts_ind = smpl_layer.body_vertex_idx
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=80)
    progress_bar.set_description('evaluate')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        # time only the model forward call
        if cur_device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(samples, targets)
        if cur_device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        # record milliseconds
        inference_times.append((t1 - t0) * 1000.0)
        bs = len(targets)
        for idx in range(bs):
            #gt
            gt_j2ds = targets[idx]['j2ds'].cpu().numpy()[:,:24,:]
            gt_j3ds = targets[idx]['j3ds'].cpu().numpy()[:,:24,:]
            gt_verts = targets[idx]['verts'].cpu().numpy()

            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]
            pred_j3ds = outputs['pred_j3ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()


            matched_verts_idx = []
            assert len(gt_j2ds.shape) == 3 and len(pred_j2ds.shape) == 3
            #matching
            greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds) # tuples are (idx_pred_kps, idx_gt_kps)
            matchDict, falsePositive_count = get_matching_dict(greedy_match)

            #align with matching result
            gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
            gtIdxs = np.arange(len(gt_j3ds))
            miss_flag = []
            for gtIdx in gtIdxs:
                gt_verts_list.append(gt_verts[gtIdx])
                gt_joints_list.append(gt_j3ds[gtIdx])
                if matchDict[str(gtIdx)] == 'miss' or matchDict[str(
                        gtIdx)] == 'invalid':
                    miss_flag.append(1)
                    pred_verts_list.append([])
                    pred_joints_list.append([])
                else:
                    miss_flag.append(0)
                    pred_joints_list.append(pred_j3ds[matchDict[str(gtIdx)]])
                    pred_verts_list.append(pred_verts[matchDict[str(gtIdx)]])
                    matched_verts_idx.append(matchDict[str(gtIdx)])

            if has_kid:
                gt_kid_list = targets[idx]['kid']

            #calculating 3d errors
            for i, (gt3d, pred) in enumerate(zip(gt_joints_list, pred_joints_list)):
                total_count += 1
                if has_kid and gt_kid_list[i]:
                    kid_total_count += 1

                # Get corresponding ground truth and predicted 3d joints and verts
                if miss_flag[i] == 1:
                    total_miss_count += 1
                    if has_kid and gt_kid_list[i]:
                        kid_total_miss_count += 1
                    continue

                gt3d = gt3d.reshape(-1, 3)
                pred3d = pred.reshape(-1, 3)
                gt3d_verts = gt_verts_list[i].reshape(-1, 3)
                pred3d_verts = pred_verts_list[i].reshape(-1, 3)
                
                gt3d, gt3d_verts = select_and_align(gt3d, gt3d_verts, body_verts_ind)
                pred3d, pred3d_verts = select_and_align(pred3d, pred3d_verts, body_verts_ind)

                #joints
                error_j, pa_error_j = cal_3d_position_error(pred3d, gt3d)
                mpjpe.append(error_j)
                if has_kid and gt_kid_list[i]:
                    kid_mpjpe.append(error_j)
                #vertices
                error_v,pa_error_v = cal_3d_position_error(pred3d_verts, gt3d_verts)
                mve.append(error_v)
                if has_kid and gt_kid_list[i]:
                    kid_mve.append(error_v)


            #counting
            step += 1
            total_fp += falsePositive_count

            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            
            if vis and (img_idx%vis_step == 0):
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())

                # render mesh
                colors = [(1.0, 1.0, 0.9)] * len(gt_verts)
                gt_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = gt_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = targets[idx]['cam_intrinsics'].reshape(3,3).detach().cpu(),
                                            colors = colors)

                colors = [(1.0, 0.6, 0.6)] * len(pred_verts)   
                for i in matched_verts_idx:
                    colors[i] = (0.7, 1.0, 0.4)

                # colors = get_colors_rgb(len(pred_verts))
                pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = pred_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                            colors = colors,
                                            )


                if 'enc_outputs' not in outputs:
                    pred_scale_img = np.zeros_like(pred_mesh_img)
                else:
                    enc_out = outputs['enc_outputs']
                    h, w = enc_out['hw'][idx]
                    flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                    ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                    xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                    scale_map = torch.zeros((h,w,2))
                    scale_map[ys,xs] = flatten_map

                    pred_scale_img = vis_scale_img(img = ori_img.copy(),
                                                   scale_map = scale_map,
                                                   conf_thresh = model.sat_cfg['conf_thresh'],
                                                   patch_size=28)

                pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
                pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
                pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))

                # sat
                sat_img = vis_sat(ori_img.copy(),
                                    input_size = model.input_size,
                                    patch_size = 14,
                                    sat_dict = outputs['sat'],
                                    bid = idx)

                ori_img = pad_img(ori_img, model.input_size)

                full_img = np.vstack([np.hstack([ori_img, sat_img]),
                                      np.hstack([pred_scale_img, pred_box_img]),
                                      np.hstack([gt_mesh_img, pred_mesh_img])])

                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)
                
        progress_bar.update(1)

    if distributed:
        mve = accelerator.gather_for_metrics(mve)
        mpjpe = accelerator.gather_for_metrics(mpjpe)


        total_miss_count = sum(accelerator.gather_for_metrics([total_miss_count]))
        total_count = sum(accelerator.gather_for_metrics([total_count]))
        total_fp = sum(accelerator.gather_for_metrics([total_fp]))

        if has_kid:
            kid_mve = accelerator.gather_for_metrics(kid_mve)
            kid_mpjpe = accelerator.gather_for_metrics(kid_mpjpe)
            kid_total_miss_count = sum(accelerator.gather_for_metrics([kid_total_miss_count]))
            kid_total_count = sum(accelerator.gather_for_metrics([kid_total_count]))

    if len(mpjpe) <= num_processes:
        return "Failed to evaluate. Keep training!"
    if has_kid and len(kid_mpjpe) <= num_processes:
        return "Failed to evaluate. Keep training!"
    
    precision, recall, f1 = compute_prf1(total_count,total_miss_count,total_fp)
    error_dict = {}
    error_dict['precision'] = precision
    error_dict['recall'] = recall
    error_dict['f1'] = f1

    error_dict['MPJPE'] = round(float(sum(mpjpe)/(len(mpjpe)-num_processes)), 1)
    error_dict['NMJE'] = round(error_dict['MPJPE'] / (f1), 1)
    error_dict['MVE'] = round(float(sum(mve)/(len(mve)-num_processes)), 1)
    error_dict['NMVE'] = round(error_dict['MVE'] / (f1), 1)

    if has_kid:
        kid_precision, kid_recall, kid_f1 = compute_prf1(kid_total_count,kid_total_miss_count,total_fp)
        error_dict['kid_precision'] = kid_precision
        error_dict['kid_recall'] = kid_recall
        error_dict['kid_f1'] = kid_f1

        error_dict['kid-MPJPE'] = round(float(sum(kid_mpjpe)/(len(kid_mpjpe)-num_processes)), 1)
        error_dict['kid-NMJE'] = round(error_dict['kid-MPJPE'] / (kid_f1), 1)
        error_dict['kid-MVE'] = round(float(sum(kid_mve)/(len(kid_mve)-num_processes)), 1)
        error_dict['kid-NMVE'] = round(error_dict['kid-MVE'] / (kid_f1), 1)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path,'results.txt'),'w') as f:
            for k,v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict


def test_agora(model, eval_dataloader, conf_thresh, 
                vis = True, vis_step = 400, results_save_path = None,
                distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None

    os.makedirs(os.path.join(results_save_path,'predictions'),exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    step = 0
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=100)
    progress_bar.set_description('testing')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():    
           outputs = model(samples, targets)
        bs = len(targets)
        for idx in range(bs):
            #gt
            img_name = targets[idx]['img_name'].split('.')[0]
            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_j2ds = np.array(outputs['pred_j2ds'][idx][select_queries_idx].detach().to('cpu'))[:,:24,:]*(3840/model.input_size)
            pred_j3ds = np.array(outputs['pred_j3ds'][idx][select_queries_idx].detach().to('cpu'))[:,:24,:]
            pred_verts = np.array(outputs['pred_verts'][idx][select_queries_idx].detach().to('cpu'))
            pred_poses = np.array(outputs['pred_poses'][idx][select_queries_idx].detach().to('cpu'))
            pred_betas = np.array(outputs['pred_betas'][idx][select_queries_idx].detach().to('cpu'))

            #visualization
            step+=1
            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            if vis and (img_idx%vis_step == 0):
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                ori_img = pad_img(ori_img, model.input_size)

                sat_img = vis_sat(ori_img.copy(),
                                    input_size = model.input_size,
                                    patch_size = 14,
                                    sat_dict = outputs['sat'],
                                    bid = idx)
                
                colors = get_colors_rgb(len(pred_verts))
                mesh_img = vis_meshes_img(img = ori_img.copy(),
                                          verts = pred_verts,
                                          smpl_faces = smpl_layer.faces,
                                          colors = colors,
                                          cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())
                
                if 'enc_outputs' not in outputs:
                    pred_scale_img = np.zeros_like(ori_img)
                else:
                    enc_out = outputs['enc_outputs']
                    h, w = enc_out['hw'][idx]
                    flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                    ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                    xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                    scale_map = torch.zeros((h,w,2))
                    scale_map[ys,xs] = flatten_map
                    pred_scale_img = vis_scale_img(img = ori_img.copy(),
                                                   scale_map = scale_map,
                                                   conf_thresh = model.sat_cfg['conf_thresh'],
                                                   patch_size=28)

                full_img = np.vstack([np.hstack([ori_img, mesh_img]),
                                      np.hstack([pred_scale_img, sat_img])])
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

            
            # submit
            for pnum in range(len(pred_j2ds)):
                smpl_dict = {}
                # smpl_dict['age'] = 'kid'
                smpl_dict['joints'] = pred_j2ds[pnum].reshape(24,2)
                smpl_dict['params'] = {'transl': np.zeros((1,3)),
                                        'betas': pred_betas[pnum].reshape(1,10),
                                        'global_orient': pred_poses[pnum][:3].reshape(1,1,3),
                                        'body_pose': pred_poses[pnum][3:].reshape(1,23,3)}
                # smpl_dict['verts'] = pred_verts[pnum].reshape(6890,3)
                # smpl_dict['allSmplJoints3d'] = pred_j3ds[pnum].reshape(24,3)
                with open(os.path.join(results_save_path,'predictions',f'{img_name}_personId_{pnum}.pkl'), 'wb') as f:
                    pickle.dump(smpl_dict, f)
 
        progress_bar.update(1)

    accelerator.print('Packing...')

    folder_path = os.path.join(results_save_path,'predictions')
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_save_path,f'pred_{timestamp}.zip')
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname)


    return 'Results saved at: ' + os.path.join(results_save_path,'predictions')

def evaluate_3dpw(model, eval_dataloader, conf_thresh,
                        vis = True, vis_step = 40, results_save_path = None,
                        distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None
    num_processes = accelerator.num_processes
    
    os.makedirs(results_save_path,exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    
    step = 0
    total_miss_count = 0
    total_count = 0
    total_fp = 0

    mve, mpjpe, pa_mpjpe, pa_mve = [0.], [0.], [0.], [0.]
    # focal statistics: collect per-image predicted focal (first person), gt focal (first person), and absolute error
    pred_focals = []
    gt_focals = []
    focal_errors = []
    # height statistics: collect predicted and gt heights and absolute errors (per matched person)
    pred_heights_vals = []
    gt_heights_vals = []
    height_errors = []
    # inference timing (ms) for the model forward call: outputs = model(samples, targets)
    inference_times = []
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    smpl2h36m_regressor = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(cur_device)
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=80)
    progress_bar.set_description('evaluate')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]

        with torch.no_grad():    
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(samples, targets)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            # record milliseconds
            # print((t1 - t0) * 1000.0)
            inference_times.append((t1 - t0) * 1000.0)

        bs = len(targets)
        for idx in range(bs):
            #gt 
            gt_verts = targets[idx]['verts']
            gt_transl = targets[idx]['transl']
            gt_j3ds = torch.einsum('bik,ji->bjk', [gt_verts - gt_transl[:,None,:], smpl2h36m_regressor]) + gt_transl[:,None,:]

            gt_verts = gt_verts.cpu().numpy()
            gt_heights = targets[idx]['heights'].cpu().numpy()
            gt_j3ds = gt_j3ds.cpu().numpy()
            gt_j2ds = targets[idx]['j2ds'].cpu().numpy()[:,:24,:]
            gt_focal = targets[idx]['focals'][0].cpu().numpy()

            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
            pred_poses = outputs['pred_poses'][idx][select_queries_idx].detach()
            pred_betas = outputs['pred_betas'][idx][select_queries_idx].detach()
            try:
                pred_verts_t = smpl_layer(betas=pred_betas, poses=pred_poses*0)[0]
                pred_heights = torch.max(pred_verts_t[:, :, 1], dim=1).values - torch.min(pred_verts_t[:, :, 1], dim=1).values
            except Exception:
                pred_heights = torch.zeros(len(pred_betas)).to(pred_betas.device)

            pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
            pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:,None,:], smpl2h36m_regressor]) + pred_transl[:,None,:]
            
            pred_verts = pred_verts.cpu().numpy()
            pred_j3ds = pred_j3ds.cpu().numpy()
            pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]
            pred_focal = outputs['pred_intrinsics'][idx][0, 0, 0].detach().cpu().numpy()

            pred_focals.append(pred_focal)
            gt_focals.append(gt_focal)
            focal_errors.append(abs(pred_focal - gt_focal))

            matched_verts_idx = []
            assert len(gt_j2ds.shape) == 3 and len(pred_j2ds.shape) == 3
            #matching
            greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds) # tuples are (idx_pred_kps, idx_gt_kps)
            matchDict, falsePositive_count = get_matching_dict(greedy_match)

            #align with matching result
            gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
            gt_heights_list, pred_heights_list = [], []
            gtIdxs = np.arange(len(gt_j3ds))
            miss_flag = []
            for gtIdx in gtIdxs:
                gt_verts_list.append(gt_verts[gtIdx])
                gt_joints_list.append(gt_j3ds[gtIdx])
                gt_heights_list.append(gt_heights[gtIdx])
                if matchDict[str(gtIdx)] == 'miss' or matchDict[str(
                        gtIdx)] == 'invalid':
                    miss_flag.append(1)
                    pred_verts_list.append([])
                    pred_joints_list.append([])
                    pred_heights_list.append([])
                else:
                    miss_flag.append(0)
                    pred_joints_list.append(pred_j3ds[matchDict[str(gtIdx)]])
                    pred_verts_list.append(pred_verts[matchDict[str(gtIdx)]])
                    pred_heights_list.append(pred_heights[matchDict[str(gtIdx)]].cpu().numpy())
                    matched_verts_idx.append(matchDict[str(gtIdx)])

            #calculating 3d errors
            for i, (gt3d, pred) in enumerate(zip(gt_joints_list, pred_joints_list)):
                total_count += 1

                # Get corresponding ground truth and predicted 3d joints and verts
                if miss_flag[i] == 1:
                    total_miss_count += 1
                    continue

                gt3d = gt3d.reshape(-1, 3)
                pred3d = pred.reshape(-1, 3)
                gt3d_verts = gt_verts_list[i].reshape(-1, 3)
                pred3d_verts = pred_verts_list[i].reshape(-1, 3)

                gt_h = gt_heights_list[i]
                pred_h = pred_heights_list[i]

                height_errors.append(abs(pred_h - gt_h))

                gt_pelvis = gt3d[[0],:].copy()
                pred_pelvis = pred3d[[0],:].copy()

                gt3d = (gt3d - gt_pelvis)[H36M_EVAL_JOINTS, :].copy()
                gt3d_verts = (gt3d_verts - gt_pelvis).copy()
                
                pred3d = (pred3d - pred_pelvis)[H36M_EVAL_JOINTS, :].copy()
                pred3d_verts = (pred3d_verts - pred_pelvis).copy()

                #joints
                error_j, pa_error_j = cal_3d_position_error(pred3d, gt3d)
                mpjpe.append(error_j)
                pa_mpjpe.append(pa_error_j)
                #vertices
                error_v, pa_error_v = cal_3d_position_error(pred3d_verts, gt3d_verts)
                mve.append(error_v)
                pa_mve.append(pa_error_v)


            #counting
            step += 1
            total_fp += falsePositive_count

            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            
            if vis and (img_idx%vis_step == 0) and len(matched_verts_idx) > 0:
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                ori_img = pad_img(ori_img, model.input_size)

                # prepare matched pairs (gt_idx -> pred_idx)
                matched_pairs = []
                for gtIdx in gtIdxs:
                    if matchDict[str(gtIdx)] == 'miss' or matchDict[str(gtIdx)] == 'invalid':
                        continue
                    matched_pairs.append((int(gtIdx), int(matchDict[str(gtIdx)])))

                # build lists of verts and translations for gt and pred
                gt_list = [gt_verts[gt_i] for gt_i, _ in matched_pairs]
                pred_list = [pred_verts[p_i] for _, p_i in matched_pairs]

                gt_trans_list = [targets[idx]['transl'][gt_i].cpu().numpy() for gt_i, _ in matched_pairs]
                pred_trans_list = [pred_transl[p_i].detach().cpu().numpy() for _, p_i in matched_pairs]

                colors = get_colors_rgb(len(matched_verts_idx))
                colors_gt = colors
                colors_pred = colors

                # render projected meshes on image
                # gt_mesh_img = vis_meshes_img(img = ori_img.copy(),
                #                              verts = gt_list,
                #                              smpl_faces = smpl_layer.faces,
                #                              colors = colors_gt,
                #                              cam_intrinsics = targets[idx]['cam_intrinsics'].detach().cpu())

                # pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                #                                verts = pred_list,
                #                                smpl_faces = smpl_layer.faces,
                #                                colors = colors_pred,
                #                                cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())
                
                faces_list = [smpl_layer.faces] * max(1, len(pred_list))
                K_gt = targets[idx]['cam_intrinsics'][0].detach().cpu().numpy()
                K_vis = outputs['pred_intrinsics'][idx][0].detach().cpu().numpy()

                gt_mesh_img = render_meshes(img=ori_img.copy(), 
                                            l_mesh=gt_list, 
                                            l_face=faces_list, 
                                            cam_param={'focal': np.asarray([K_gt[0,0],K_gt[1,1]]), 'princpt': np.asarray([K_gt[0,-1],K_gt[1,-1]])}, 
                                            color=colors_gt,
                                            heights=gt_heights_list)
                pred_mesh_img = render_meshes(img=ori_img.copy(),
                                            l_mesh=pred_list, 
                                            l_face=faces_list, 
                                            cam_param={'focal': np.asarray([K_vis[0,0],K_vis[1,1]]), 'princpt': np.asarray([K_vis[0,-1],K_vis[1,-1]])}, 
                                            color=colors_pred,
                                            heights=pred_heights_list)

                # faces list repeated

                _, pred_sideview, pred_bev = render_side_views(ori_img.copy(), colors_pred, pred_list, faces_list, pred_trans_list, K_vis)
                _, gt_sideview, gt_bev = render_side_views(ori_img.copy(), colors_gt, gt_list, faces_list, gt_trans_list, K_gt)

                # compose final visualization: top row: ori | gt_proj | pred_proj ; bottom row: pred_side | gt_side | bev comparison
                top_row = np.hstack([ori_img, gt_mesh_img, pred_mesh_img])
                bottom_row = np.hstack([pred_sideview, gt_bev, pred_bev])
                # ensure same widths: if mismatched, resize bottom_row to match top_row width
                if bottom_row.shape[1] != top_row.shape[1]:
                    # simple horizontal tiling adjustment: resize bottom_row width via padding
                    pad_w = top_row.shape[1] - bottom_row.shape[1]
                    if pad_w > 0:
                        bottom_row = np.pad(bottom_row, ((0,0),(0,pad_w),(0,0)), mode='constant', constant_values=255)

                full_img = np.vstack([top_row, bottom_row])
                # downsample=2
                full_img = cv2.resize(full_img, (full_img.shape[1]//2, full_img.shape[0]//2))
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)
                
        progress_bar.update(1)

    if distributed:
        mve = accelerator.gather_for_metrics(mve)
        mpjpe = accelerator.gather_for_metrics(mpjpe)
        pa_mpjpe = accelerator.gather_for_metrics(pa_mpjpe)
        pa_mve = accelerator.gather_for_metrics(pa_mve)

        total_miss_count = sum(accelerator.gather_for_metrics([total_miss_count]))
        total_count = sum(accelerator.gather_for_metrics([total_count]))
        total_fp = sum(accelerator.gather_for_metrics([total_fp]))
        # gather focal stats lists
        pred_focals = accelerator.gather_for_metrics(pred_focals)
        gt_focals = accelerator.gather_for_metrics(gt_focals)
        focal_errors = accelerator.gather_for_metrics(focal_errors)
        # gather height stats lists
        height_errors = accelerator.gather_for_metrics(height_errors)
        # gather inference times
        inference_times = accelerator.gather_for_metrics(inference_times)

    if len(mpjpe) <= num_processes:
        return "Failed to evaluate. Keep training!"
    
    precision, recall, f1 = compute_prf1(total_count,total_miss_count,total_fp)
    error_dict = {}
    error_dict['recall'] = recall

    error_dict['MPJPE'] = round(float(sum(mpjpe)/(len(mpjpe)-num_processes)), 1)
    error_dict['PA-MPJPE'] = round(float(sum(pa_mpjpe)/(len(pa_mpjpe)-num_processes)), 1)
    error_dict['MVE'] = round(float(sum(mve)/(len(mve)-num_processes)), 1)
    error_dict['PA-MVE'] = round(float(sum(pa_mve)/(len(pa_mve)-num_processes)), 1)

    # focal statistics: compute mean absolute error, and mean/std for pred and gt focals
    # remove placeholder zeros added at initialization
    pf = np.array(pred_focals, dtype=float)
    gf = np.array(gt_focals, dtype=float)
    fe = np.array(focal_errors, dtype=float)

    focal_mae = float(np.mean(fe))
    pred_mean = float(np.mean(pf))
    pred_std = float(np.std(pf, ddof=0))
    gt_mean = float(np.mean(gf))
    gt_std = float(np.std(gf, ddof=0))

    error_dict['focal'] = round(focal_mae, 4)
    error_dict['pred_focal'] = {'mean': round(pred_mean, 4), 'std': round(pred_std, 4)}
    error_dict['gt_focal'] = {'mean': round(gt_mean, 4), 'std': round(gt_std, 4)}

    # height statistics
    error_dict['height_error'] = np.round(float(np.mean(np.array(height_errors, dtype=float))), 4)

    # inference time: compute mean across all recorded forward calls (ms)
    try:
        it = np.array(inference_times, dtype=float)
        # if list contains nested lists because of gather, flatten
        it = it.flatten()
        # remove zero or near-zero entries if any accidental placeholders
        if it.size == 0:
            mean_it = 0.0
        else:
            mean_it = float(np.mean(it))
    except Exception:
        mean_it = 0.0

    error_dict['inference_time_ms'] = round(mean_it, 4)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path,'results.txt'),'w') as f:
            for k,v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict


def evaluate_panoptic(model, eval_dataloader, conf_thresh,
                        vis = True, vis_step = 40, results_save_path = None,
                        distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None
    num_processes = accelerator.num_processes

    seqs = ['haggling', 'mafia', 'ultimatum', 'pizza']
    J24_TO_H36M = [14, 2, 1, 0, 3, 4, 5, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6] # Different SMPL2H36M regressor
    # J24_TO_H36M = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    os.makedirs(results_save_path,exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    
    step = 0

    seq_mpjpe = {'haggling': [0.], 'mafia':[0.], 'ultimatum':[0.], 'pizza':[0.]}
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    smpl2h36m_regressor = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(cur_device)
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=80)
    progress_bar.set_description('evaluate')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():    
           outputs = model(samples, targets)
        bs = len(targets)
        for idx in range(bs):
            #gt
            gt_j3ds = targets[idx]['j3ds'].cpu()
            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            while len(select_queries_idx) == 0:
                conf_thresh /= 2
                select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]

            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
            pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
            pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:,None,:], smpl2h36m_regressor]) + pred_transl[:,None,:]

            pred_verts = pred_verts.detach().cpu()
            pred_j3ds = pred_j3ds.detach().cpu()
            
            # following BMP
            gt_j3ds[:, 14, :] -= torch.tensor([0., 0.06, 0., 0.])
            gt_keypoints_3d = gt_j3ds.clone()
            gt_pelvis_smpl = gt_keypoints_3d[:, [14], :-1].clone()
            visible_kpts = gt_keypoints_3d[:, J24_TO_H36M, -1].clone()
            # visible_kpts = targets[idx]['vis'][:, J24_TO_H36M].cpu().clone()
            origin_gt_kpts3d = gt_j3ds.clone()
            origin_gt_kpts3d = origin_gt_kpts3d[:, J24_TO_H36M]
            origin_gt_kpts3d[:, :, :-1] -= gt_pelvis_smpl
            gt_keypoints_3d = gt_keypoints_3d[:, J24_TO_H36M, :-1].clone()
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis_smpl

            # Get 14 predicted joints from the SMPL mesh
            pred_keypoints_3d_smpl = pred_j3ds
            pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
            pred_keypoints_3d_smpl = pred_keypoints_3d_smpl - pred_pelvis_smpl

            # To select closest points
            glb_vis = (visible_kpts.sum(0) >= (
                    visible_kpts.shape[0] - 0.1)).float()[None, :, None]  # To avoid in-accuracy in float point number

            
            dist = vectorize_distance((glb_vis * gt_keypoints_3d).numpy(),
                                    (glb_vis * pred_keypoints_3d_smpl).numpy())
            paired_idxs = torch.from_numpy(dist.argmin(1))

            # is_mismatch = len(set(paired_idxs.tolist())) < len(paired_idxs)
            # if is_mismatch:
            #     self.mismatch_cnt += 1

            selected_prediction = pred_keypoints_3d_smpl[paired_idxs]

            # Compute error metrics
            # Absolute error (MPJPE)
            error_smpl = (torch.sqrt(((selected_prediction - gt_keypoints_3d) ** 2).sum(dim=-1)) * visible_kpts)
            mpjpes = error_smpl.mean(-1) * 1000
            mpjpes = mpjpes.tolist()
            
            for seq in seqs:
                if seq in targets[idx]['img_path']:
                    for mpjpe in mpjpes:
                        seq_mpjpe[seq].append(float(mpjpe))
                    break

            gt_keypoints_3d = gt_keypoints_3d.numpy()
            selected_prediction = selected_prediction.numpy()
            visible_kpts = visible_kpts.bool().numpy()

            #counting
            step+=1
            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            if vis and (img_idx%vis_step == 0):
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                ori_img = pad_img(ori_img, model.input_size)
                
                # selected predicted verts
                if len(paired_idxs) == 0:
                    continue
                selected_verts = pred_verts[paired_idxs].numpy()
                # convert to numpy list for render_meshes / render_side_views
                try:
                    selected_verts_np = [v.numpy() for v in selected_verts]
                except Exception:
                    selected_verts_np = [v for v in selected_verts]

                colors = get_colors_rgb(len(selected_verts_np))
                faces_list = [smpl_layer.faces] * max(1, len(selected_verts_np))

                # use predicted intrinsics for visualization
                K_vis = outputs['pred_intrinsics'][idx][0].detach().cpu().numpy()
                cam_param = {'focal': np.asarray([K_vis[0,0], K_vis[1,1]]), 'princpt': np.asarray([K_vis[0,2], K_vis[1,2]])}

                # render projected meshes on image (pred only)
                try:
                    pred_mesh_img = render_meshes(img=ori_img.copy(), l_mesh=selected_verts_np, l_face=faces_list, cam_param=cam_param, color=colors)
                except Exception:
                    # fallback to simple overlay
                    pred_mesh_img = vis_meshes_img(img = ori_img.copy(), verts = selected_verts, smpl_faces = smpl_layer.faces, colors = colors, cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())

                # translations for side view
                try:
                    pred_trans_list = [pred_transl[i].detach().cpu().numpy() for i in paired_idxs]
                except Exception:
                    pred_trans_list = [np.zeros(3) for _ in selected_verts_np]

                # render side and bird-eye views for predictions
                try:
                    _, pred_sideview, pred_bev = render_side_views(ori_img.copy(), colors, selected_verts_np, faces_list, pred_trans_list, K_vis)
                except Exception:
                    pred_sideview = np.ones_like(ori_img) * 255
                    pred_bev = np.ones_like(ori_img) * 255

                top_row = np.hstack([ori_img, pred_mesh_img])
                bottom_row = np.hstack([pred_sideview, pred_bev])
                if bottom_row.shape[1] != top_row.shape[1]:
                    pad_w = top_row.shape[1] - bottom_row.shape[1]
                    if pad_w > 0:
                        bottom_row = np.pad(bottom_row, ((0,0),(0,pad_w),(0,0)), mode='constant', constant_values=255)

                full_img = np.vstack([top_row, bottom_row])
                # downsample for saving
                full_img = cv2.resize(full_img, (full_img.shape[1]//2, full_img.shape[0]//2))
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

        progress_bar.update(1)

    if distributed:
        for seq in seqs:
            seq_mpjpe[seq] = accelerator.gather_for_metrics(seq_mpjpe[seq])
    
    error_dict = {}

    for seq in seqs:
        if len(seq_mpjpe[seq]) <= num_processes:
            return "Failed to evaluate. Keep training!"
        error_dict[seq] = round(sum(seq_mpjpe[seq])/(len(seq_mpjpe[seq])-num_processes), 1)

    all_mpjpe = []
    for seq in seqs:
        all_mpjpe += seq_mpjpe[seq]

    error_dict['mean'] = round(sum(all_mpjpe)/(len(all_mpjpe)-num_processes*4), 1)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path,'results.txt'),'w') as f:
            for k,v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict


# Modified from https://github.com/Arthur151/ROMP/tree/master/trace/lib/evaluation/mupots_util
def evaluate_mupots(model, eval_dataloader, conf_thresh,
                    vis = True, vis_step = 40, results_save_path = None,
                    distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None
    
    os.makedirs(results_save_path, exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)

    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    smpl2h36m_regressor = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(cur_device)
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=80)
    progress_bar.set_description('evaluate')

    step = 0
    mismatch_cnt = 0
    pje_all_dict = defaultdict(list)
    pje_all_mask = defaultdict(list)
    pje_match_dict = defaultdict(list)
    pje_match_mask = defaultdict(list)
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():
           outputs = model(samples, targets)
        bs = len(targets)
        for idx in range(bs):
            step += 1
            ts_name = targets[idx]['TS']

            # gt
            gt_j3ds = targets[idx]['j3ds'].cpu().numpy()
            gt_j2ds = targets[idx]['j2ds'].cpu().numpy()
            gt_jvis = targets[idx]['jvis'].cpu().numpy()
            img_size = targets[idx]['img_size'].cpu().numpy()
            resize_rate = targets[idx]['resize_rate'].cpu().numpy()
            gt_j2ds[:, :, 0] = gt_j2ds[:, :, 0] * img_size[1] / resize_rate
            gt_j2ds[:, :, 1] = gt_j2ds[:, :, 1] * img_size[0] / resize_rate
            gt_visibility = np.ones(gt_j2ds.shape[:2], dtype='bool')

            # pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
            pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
            pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:,None,:], smpl2h36m_regressor]) + pred_transl[:,None,:]

            pred_intrinsics = outputs['pred_intrinsics'][idx][0]
            pred_j2ds_homo = torch.einsum('bjc,cd->bjd', pred_j3ds, pred_intrinsics.transpose(0, 1))
            pred_j2ds = pred_j2ds_homo[..., :2] / (pred_j2ds_homo[..., 2:] + 1e-6)
            pred_j2ds = pred_j2ds.cpu().numpy() / resize_rate
            pred_j3ds = pred_j3ds.cpu().numpy() * 1000.
            pred_visibility = np.ones(pred_j2ds.shape[:2], dtype='bool')

            H36M_to_MPI = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]
            gt_j2ds = gt_j2ds[:, H36M_to_MPI]
            gt_j3ds = gt_j3ds[:, H36M_to_MPI]
            pred_j2ds = pred_j2ds[:, H36M_to_MPI]
            pred_j3ds = pred_j3ds[:, H36M_to_MPI]

            # matching
            joints_for_matching = np.arange(1, 14)
            pair_inds = _match_poses(
                gt_j2ds[:, joints_for_matching], gt_visibility[:, joints_for_matching], 
                pred_j2ds[:, joints_for_matching], pred_visibility[:, joints_for_matching], 40)
            
            for k in range(len(pair_inds)):
                jvis = gt_jvis[k]
                gtP = gt_j3ds[k, :] - gt_j3ds[k, 14:15]
                if pair_inds[k] != -1:
                    predP = pred_j3ds[pair_inds[k]]
                    predP = predP - predP[14:15]
                    predP = _scale_to_gt(predP, gtP)
                else:
                    mismatch_cnt += 1
                    predP = 100000 * np.ones(gtP.shape)

                errorP = np.sqrt(np.power(predP - gtP, 2).sum(axis=1))
                pje_all_dict[ts_name].append(errorP)
                pje_all_mask[ts_name].append(jvis)
                if pair_inds[k] != -1:
                    pje_match_dict[ts_name].append(errorP)
                    pje_match_mask[ts_name].append(jvis)

            img_idx = step + accelerator.process_index * len(eval_dataloader) * bs
            if vis and img_idx % vis_step == 0:
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                ori_img = pad_img(ori_img, model.input_size)
                
                matches = list(filter(lambda x: x != -1, pair_inds))
                if len(matches) == 0:
                    continue
                selected_verts = pred_verts[matches].detach().cpu()
                # convert to numpy list
                try:
                    selected_verts_np = [v.numpy() for v in selected_verts]
                except Exception:
                    selected_verts_np = [v for v in selected_verts]

                colors = get_colors_rgb(len(selected_verts_np))
                faces_list = [smpl_layer.faces] * max(1, len(selected_verts_np))

                K_vis = outputs['pred_intrinsics'][idx][0].detach().cpu().numpy()
                cam_param = {'focal': np.asarray([K_vis[0,0], K_vis[1,1]]), 'princpt': np.asarray([K_vis[0,2], K_vis[1,2]])}

                try:
                    pred_mesh_img = render_meshes(img=ori_img.copy(), l_mesh=selected_verts_np, l_face=faces_list, cam_param=cam_param, color=colors)
                except Exception:
                    pred_mesh_img = vis_meshes_img(img = ori_img.copy(), verts = selected_verts, smpl_faces = smpl_layer.faces, colors = colors, cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())

                try:
                    pred_trans_list = [pred_transl[i].detach().cpu().numpy() for i in matches]
                except Exception:
                    pred_trans_list = [np.zeros(3) for _ in selected_verts_np]

                try:
                    _, pred_sideview, pred_bev = render_side_views(ori_img.copy(), colors, selected_verts_np, faces_list, pred_trans_list, K_vis)
                except Exception:
                    pred_sideview = np.ones_like(ori_img) * 255
                    pred_bev = np.ones_like(ori_img) * 255

                top_row = np.hstack([ori_img, pred_mesh_img])
                bottom_row = np.hstack([pred_sideview, pred_bev])
                if bottom_row.shape[1] != top_row.shape[1]:
                    pad_w = top_row.shape[1] - bottom_row.shape[1]
                    if pad_w > 0:
                        bottom_row = np.pad(bottom_row, ((0,0),(0,pad_w),(0,0)), mode='constant', constant_values=255)

                full_img = np.vstack([top_row, bottom_row])
                full_img = cv2.resize(full_img, (full_img.shape[1]//2, full_img.shape[0]//2))
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

        progress_bar.update(1)

    if distributed:
        for k in pje_all_dict.keys():
            pje_all_dict[k] = accelerator.gather_for_metrics(pje_all_dict[k])
            pje_all_mask[k] = accelerator.gather_for_metrics(pje_all_mask[k])
            pje_match_dict[k] = accelerator.gather_for_metrics(pje_match_dict[k])
            pje_match_mask[k] = accelerator.gather_for_metrics(pje_match_mask[k])

    pck_all = mpii_compute_3d_pck(pje_all_dict, pje_all_mask)
    pck_match = mpii_compute_3d_pck(pje_match_dict, pje_match_mask)
    
    error_dict = {}
    error_dict['PCK (All)'] = round(float(pck_all * 100), 1)
    error_dict['PCK (Match)'] = round(float(pck_match * 100), 1)
    error_dict['mismatch'] = mismatch_cnt

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path, 'results.txt'),'w') as f:
            for k,v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict


def test_chi3d(model, eval_dataloader, conf_thresh, 
               vis = True, vis_step = 400, results_save_path = None,
               distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None

    os.makedirs(os.path.join(results_save_path, 'predictions'),exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    step = 0
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=100)
    progress_bar.set_description('testing')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():    
           outputs = model(samples, targets)
        bs = len(targets)
        for idx in range(bs):
            # gt
            img_name = targets[idx]['img_name'].split('.')[0]
            
            # pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_j2ds = np.array(outputs['pred_j2ds'][idx][select_queries_idx].detach().to('cpu'))[:,:24,:]*(3840/model.input_size)
            pred_j3ds = np.array(outputs['pred_j3ds'][idx][select_queries_idx].detach().to('cpu'))[:,:24,:]
            pred_verts = np.array(outputs['pred_verts'][idx][select_queries_idx].detach().to('cpu'))
            pred_poses = np.array(outputs['pred_poses'][idx][select_queries_idx].detach().to('cpu'))
            pred_betas = np.array(outputs['pred_betas'][idx][select_queries_idx].detach().to('cpu'))

            # visualization
            step+=1
            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            if vis and (img_idx%vis_step == 0):
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                ori_img = pad_img(ori_img, model.input_size)

                sat_img = vis_sat(ori_img.copy(),
                                    input_size = model.input_size,
                                    patch_size = 14,
                                    sat_dict = outputs['sat'],
                                    bid = idx)
                
                colors = get_colors_rgb(len(pred_verts))
                mesh_img = vis_meshes_img(img = ori_img.copy(),
                                          verts = pred_verts,
                                          smpl_faces = smpl_layer.faces,
                                          colors = colors,
                                          cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())
                
                if 'enc_outputs' not in outputs:
                    pred_scale_img = np.zeros_like(ori_img)
                else:
                    enc_out = outputs['enc_outputs']
                    h, w = enc_out['hw'][idx]
                    flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                    ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                    xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                    scale_map = torch.zeros((h,w,2))
                    scale_map[ys,xs] = flatten_map
                    pred_scale_img = vis_scale_img(img = ori_img.copy(),
                                                   scale_map = scale_map,
                                                   conf_thresh = model.sat_cfg['conf_thresh'],
                                                   patch_size=28)

                full_img = np.vstack([np.hstack([ori_img, mesh_img]),
                                      np.hstack([pred_scale_img, sat_img])])
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

            # TODO: modify submit items
            for pnum in range(len(pred_j2ds)):
                smpl_dict = {}
                smpl_dict['joints'] = pred_j2ds[pnum].reshape(24,2)
                smpl_dict['params'] = {
                    'transl': np.zeros((1,3)),
                    'betas': pred_betas[pnum].reshape(1,10),
                    'global_orient': pred_poses[pnum][:3].reshape(1,1,3),
                    'body_pose': pred_poses[pnum][3:].reshape(1,23,3)
                }
                save_path = os.path.join(results_save_path,'predictions',f'{img_name}_personId_{pnum}.pkl')
                os.makedirs(save_path, exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump(smpl_dict, f)
 
        progress_bar.update(1)

    accelerator.print('Packing...')

    folder_path = os.path.join(results_save_path,'predictions')
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_save_path,f'pred_{timestamp}.zip')
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname)

    return 'Results saved at: ' + os.path.join(results_save_path, 'predictions')
