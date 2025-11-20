import os
import re
import datetime
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import get_scheduler

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader

from .funcs.eval_funcs import *
from .funcs.infer_funcs import inference
from utils import misc
from utils.misc import get_world_size
from models import build_sat_model
from datasets.common import COMMON
from datasets.multiple_datasets import MultipleDatasets, datasets_dict


class Engine():
    def __init__(self, args, mode='train'): 
        self.exp_name = args.exp_name
        self.mode = mode
        assert mode in ['train','eval','infer']
        self.conf_thresh = args.conf_thresh
        self.eval_func_maps = {
            'agora_validation': evaluate_agora,
            'bedlam_validation_6fps': evaluate_agora,
            'agora_test': test_agora,
            'chi3d_test': test_chi3d,
            '3dpw_test': evaluate_3dpw,
            'rich_test': evaluate_3dpw,
            'harmony4d_test': evaluate_3dpw,
            'hi4d_test': evaluate_3dpw,
            'panoptic_test': evaluate_panoptic,
            'mupots_test': evaluate_mupots,
            'rh_test': evaluate_rh,
        }
        self.inference_func = inference

        if self.mode == 'train':
            self.output_dir = os.path.join('./outputs')
            self.log_dir = os.path.join(self.output_dir,'logs')
            self.ckpt_dir = os.path.join(self.output_dir,'ckpts')
            self.distributed_eval = args.distributed_eval
            self.eval_vis_num = args.eval_vis_num
        elif self.mode == 'eval':
            self.output_dir = os.path.join('./results')
            self.distributed_eval = args.distributed_eval
            self.eval_vis_num = args.eval_vis_num
        elif self.mode == 'infer':
            output_dir = getattr(args, 'output_dir', None)
            if output_dir is not None:
                self.output_dir = output_dir
            else:
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                self.output_dir = os.path.join('./results',f'{self.exp_name}_infer_{timestamp}')
            self.distributed_infer = args.distributed_infer

        self.prepare_accelerator()
        self.prepare_models(args)
        self.prepare_datas(args)
        if self.mode == 'train':
            self.prepare_training(args)

        total_cnt = sum(p.numel() for p in self.model.parameters())
        trainable_cnt = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.accelerator.print('Initialization finished')
        self.accelerator.print(f'{trainable_cnt / 1e6:.2f} M trainable parameters ({total_cnt / 1e6:.2f} M total).')

    def prepare_accelerator(self):
        if self.mode == 'train':
            self.accelerator = Accelerator(
                log_with="tensorboard",
                project_dir=os.path.join(self.log_dir)
            )
            if self.accelerator.is_main_process:
                os.makedirs(self.log_dir, exist_ok=True)
                os.makedirs(os.path.join(self.ckpt_dir,self.exp_name),exist_ok=True)
                self.accelerator.init_trackers(self.exp_name)
        else:
            self.accelerator = Accelerator()
            if self.accelerator.is_main_process:
                os.makedirs(self.output_dir, exist_ok=True)
        
    def prepare_models(self, args):
        # load model and criterion
        self.accelerator.print('Preparing models...')
        self.unwrapped_model, self.criterion = build_sat_model(args, set_criterion = (self.mode == 'train'))
        if self.criterion is not None:
            self.weight_dict = self.criterion.weight_dict
        
        # load weights
        if args.pretrain:
            self.accelerator.print(f'Loading pretrained weights: {args.pretrain_path}') 
            state_dict = torch.load(args.pretrain_path, weights_only=False)
            if args.use_kid:    
                # check shape dim
                pretrained_last_layer_key = 'shape_head.0.layers.{}.weight'.format(len(self.unwrapped_model.shape_head[0].layers)-1)
                pretrained_out_dim = state_dict[pretrained_last_layer_key].shape[0]
                if pretrained_out_dim == 10:
                    self.accelerator.print('Pretrained model is trained without kid shape parameter.')
                    # 1. 加载模型其他部分权重（排除shape_head）
                    other_state_dict = {
                        k: v for k, v in state_dict.items()
                        if not k.startswith('shape_head.') and not k.startswith('human_model.') and not k.startswith('mean_shape')
                    }
                    self.unwrapped_model.load_state_dict(other_state_dict, strict=False)

                    # 2. 逐个处理shape_head中的每个MLP
                    for mlp_idx, mlp in enumerate(self.unwrapped_model.shape_head):
                        last_layer_idx = len(mlp.layers) - 1
                        # 提取该MLP对应的预训练权重
                        pretrained_mlp_state = {
                            k: v for k, v in state_dict.items()
                            if k.startswith(f'shape_head.{mlp_idx}.')
                        }
                        
                        current_mlp_state = mlp.state_dict()
                        matched_state = {}
                        
                        for k, v in pretrained_mlp_state.items():
                            # 去掉前缀，仅保留MLP内部的键（如"layers.4.weight"）
                            inner_key = k.split(f'shape_head.{mlp_idx}.')[1]
                            if inner_key not in current_mlp_state:
                                continue
                            
                            current_shape = current_mlp_state[inner_key].shape
                            pretrained_shape = v.shape
                            
                            # 处理最后一层权重
                            if inner_key == f'layers.{last_layer_idx}.weight':
                                matched_state[inner_key] = torch.cat([v, torch.zeros(1, v.shape[1], dtype=v.dtype, device=v.device)], dim=0)  # 第十一维为0
                            elif inner_key == f'layers.{last_layer_idx}.bias':
                                matched_state[inner_key] = torch.cat([v, torch.zeros(1, dtype=v.dtype, device=v.device)], dim=0)
                            else:
                                if current_shape == pretrained_shape:
                                    matched_state[inner_key] = v
                        
                        current_mlp_state.update(matched_state)
                        mlp.load_state_dict(current_mlp_state)
                else:
                    self.unwrapped_model.load_state_dict(state_dict, strict=False)
            else:
                self.unwrapped_model.load_state_dict(state_dict, strict=False)

            if hasattr(args, 'freeze') and args.freeze:
                for n, p in self.unwrapped_model.named_parameters():
                    if n in state_dict.keys():
                        p.requires_grad_(False)

        # print computational cost information
        total_params = sum(p.numel() for p in self.unwrapped_model.parameters())
        self.accelerator.print('Total number of parameters: {:.2f} M'.format(total_params / 1e6))
        
        heads_params = [
            sum(p.numel() for p in self.unwrapped_model.bbox_embed.parameters()), 
            sum(p.numel() for p in self.unwrapped_model.conf_head.parameters()), 
            sum(p.numel() for p in self.unwrapped_model.pose_head.parameters()), 
            sum(p.numel() for p in self.unwrapped_model.shape_head.parameters()), 
            sum(p.numel() for p in self.unwrapped_model.cam_head.parameters())
        ]

        encoder_params = sum(p.numel() for p in self.unwrapped_model.encoder.parameters())
        self.accelerator.print("\tEncoder: {:.2f} M parameters".format(encoder_params / 1e6))
        decoder_params = sum(p.numel() for p in self.unwrapped_model.decoder.parameters()) - heads_params[0] - heads_params[1]
        self.accelerator.print("\tDecoder: {:.2f} M parameters".format(decoder_params / 1e6))
        self.accelerator.print("\tHeads: {:.2f} M parameters".format(sum(heads_params) / 1e6))
        self.accelerator.print("\tOthers: {:.2f} M parameters".format(
            (total_params - sum(heads_params) - encoder_params - decoder_params) / 1e6
        ))

        # to gpu
        self.model = self.accelerator.prepare(self.unwrapped_model)
        
    def prepare_datas(self, args):
        # load dataset and dataloader
        if self.mode == 'train':
            self.accelerator.print('Loading training datasets:\n',
                            [f'{d}_{s}' for d,s in zip(args.train_datasets_used, args.train_datasets_split)])
            self.train_batch_size = args.train_batch_size
            train_dataset = MultipleDatasets(args.train_datasets_used, args.train_datasets_split, args.train_datasets_ds_rate, 
                                        make_same_len=False, input_size=args.input_size, aug=True, 
                                        mode = 'train', sat_cfg=args.sat_cfg, use_kid=getattr(args,'use_kid',False),
                                        aug_cfg=args.aug_cfg)
            self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size,
                                                shuffle=True,collate_fn=misc.collate_fn, 
                                                num_workers=args.train_num_workers,pin_memory=True)
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)                                 

        if self.mode != 'infer':
            self.accelerator.print('Loading evaluation datasets:',
                                [f'{d}_{s}' for d,s in zip(args.eval_datasets_used, args.eval_datasets_split)])
            self.eval_batch_size = args.eval_batch_size
            eval_ds = {f'{ds}_{split}': datasets_dict[ds](split = split, 
                                                          mode = 'eval', 
                                                          input_size = args.input_size, 
                                                          aug = False,
                                                          sat_cfg=args.sat_cfg)\
                        for (ds, split) in zip(args.eval_datasets_used, args.eval_datasets_split)}
            self.eval_dataloaders = {k: DataLoader(dataset=v, batch_size=self.eval_batch_size,
                                        shuffle=False,collate_fn=misc.collate_fn, 
                                        num_workers=args.eval_num_workers,pin_memory=True)\
                                    for (k,v) in eval_ds.items()}
            if self.distributed_eval:
                for (k,v) in self.eval_dataloaders.items():
                    self.eval_dataloaders.update({k: self.accelerator.prepare(v)})
        
        else:
            img_folder = args.input_dir
            self.accelerator.print(f'Loading inference images from {img_folder}')
            self.infer_batch_size = args.infer_batch_size
            infer_ds = COMMON(img_folder = img_folder, input_size=args.input_size,aug=False,
                                mode = 'infer', sat_cfg=args.sat_cfg)
            self.infer_dataloader = DataLoader(dataset=infer_ds, batch_size=self.infer_batch_size,
                                        shuffle=False,collate_fn=misc.collate_fn, 
                                        num_workers=args.infer_num_workers,pin_memory=True)

            if self.distributed_infer:
                self.infer_dataloader = self.accelerator.prepare(self.infer_dataloader)

    def prepare_training(self, args):
        self.start_epoch = 0
        self.num_epochs = args.num_epochs
        self.global_step = 0
        if hasattr(args, 'sat_gt_epoch'):
            self.sat_gt_epoch = args.sat_gt_epoch
            self.accelerator.print(f'Use GT for the first {self.sat_gt_epoch} epoch(s)...')
        else:
            self.sat_gt_epoch = -1
        self.save_and_eval_epoch = args.save_and_eval_epoch
        self.least_eval_epoch = args.least_eval_epoch

        self.detach_j3ds = args.detach_j3ds

        self.accelerator.print('Preparing optimizer and lr_scheduler...')   
        param_dicts = [
            {
                "params":
                    [p for n, p in self.unwrapped_model.named_parameters()
                    if not misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": 
                    [p for n, p in self.unwrapped_model.named_parameters() 
                    if misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad],
                "lr": args.lr_encoder,
            }
        ]

        # optimizer
        if args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise NotImplementedError
        
        # lr_scheduler
        if args.lr_scheduler == 'cosine':
            self.lr_scheduler = get_scheduler(name="cosine", optimizer=self.optimizer, 
                                          num_warmup_steps=args.num_warmup_steps, 
                                          num_training_steps=get_world_size() * self.num_epochs * len(self.train_dataloader)) 
        elif args.lr_scheduler == 'multistep':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, args.milestones, gamma=args.gamma)  
        else:
            raise NotImplementedError      

        self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.optimizer, self.lr_scheduler)

        # resume
        if args.resume: #load model, optimizer, lr_scheduler and random_state
            if hasattr(args, 'ckpt_epoch'):
                self.load_ckpt(args.ckpt_epoch,args.ckpt_step)    
            else:
                self.accelerator.print('Auto resume from the latest ckpt...')
                epoch, step = -1, -1
                pattern = re.compile(r'epoch_(\d+)_step_(\d+)')
                for folder_name in os.listdir(os.path.join(self.output_dir,'ckpts',self.exp_name)):
                    match = pattern.match(folder_name)
                    if match:
                        i, j = int(match.group(1)), int(match.group(2))
                        if i > epoch:
                            epoch, step = i, j
                if epoch >= 0:
                    self.load_ckpt(epoch, step)    
                else:
                    self.accelerator.print('No existing ckpts! Train from scratch.')               

    def load_ckpt(self, epoch, step):   
        self.accelerator.print(f'Loading checkpoint: epoch_{epoch}_step_{step}') 
        ckpts_save_path = os.path.join(self.output_dir,'ckpts',self.exp_name, f'epoch_{epoch}_step_{step}')
        self.start_epoch = epoch + 1
        self.global_step = step + 1
        self.accelerator.load_state(ckpts_save_path)
    
    def train_step(self, samples, targets, sat_use_gt, step, progress_bar):
        outputs = self.model(samples, targets, sat_use_gt=sat_use_gt, detach_j3ds=self.detach_j3ds)
        loss_dict = self.criterion(outputs, targets)
        loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys())
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
    
        reduced_dict = self.accelerator.reduce(loss_dict,reduction='mean')
        simplified_logs = {k: v.item() for k, v in reduced_dict.items() if '.' not in k}
        
        if step % 10 == 0:
            self.accelerator.log({('train/'+k):v for k,v in simplified_logs.items()},
                                    step=self.global_step)

        progress_bar.update(1)
        progress_bar.set_postfix(**{"lr": self.lr_scheduler.get_last_lr()[0], "step": self.global_step})

        self.global_step += 1
        self.accelerator.wait_for_everyone()

    def train(self):
        self.accelerator.print('Start training!')
        for epoch in range(self.start_epoch, self.num_epochs):
            progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process, ncols=100)
            progress_bar.set_description(f"Epoch {epoch}")
            self.model.train()
            self.criterion.train()

            sat_use_gt = (epoch < self.sat_gt_epoch)
            for step, (samples, targets) in enumerate(self.train_dataloader):
                self.train_step(samples, targets, sat_use_gt, step, progress_bar)

            if epoch % self.save_and_eval_epoch == 0 or epoch == self.num_epochs-1:
                self.save_and_eval(epoch, save_ckpt=True)
        
        self.accelerator.end_training()

    def eval(self, results_save_path = None, epoch = -1):
        if results_save_path is None:
            results_save_path = os.path.join(self.output_dir,self.exp_name,'evaluation')
        # preparing
        self.model.eval()
        unwrapped_model = self.unwrapped_model # self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            os.makedirs(results_save_path,exist_ok=True)
        # evaluate
        for i, (key, eval_dataloader) in enumerate(self.eval_dataloaders.items()):
            assert key in self.eval_func_maps
            img_cnt = len(eval_dataloader) * self.eval_batch_size
            if self.distributed_eval:
                img_cnt *= self.accelerator.num_processes
            self.accelerator.print(f'Evaluate on {key}: {img_cnt} images')
            self.accelerator.print('Using following threshold(s): ', self.conf_thresh)
            conf_thresh = self.conf_thresh  # if 'agora' in key or 'bedlam' in key else [0.2]
            for thresh in conf_thresh:
                if self.accelerator.is_main_process or self.distributed_eval:
                    error_dict = self.eval_func_maps[key](model = unwrapped_model, 
                                    eval_dataloader = eval_dataloader, 
                                    conf_thresh = thresh,
                                    vis_step = img_cnt // self.eval_vis_num if self.eval_vis_num != 0 else None,
                                    results_save_path = os.path.join(results_save_path,key,f'thresh_{thresh}'),
                                    distributed = self.distributed_eval,
                                    accelerator = self.accelerator,
                                    vis = self.eval_vis_num != 0)
                    if isinstance(error_dict,dict) and self.mode == 'train':
                        log_dict = flatten_dict(error_dict)
                        self.accelerator.log({(f'{key}_thresh_{thresh}/'+k):v for k,v in log_dict.items()}, step=epoch)

                    self.accelerator.print(f'thresh_{thresh}: ',error_dict)
                self.accelerator.wait_for_everyone() 
  
    def save_and_eval(self, epoch, save_ckpt=False):
        torch.cuda.empty_cache()
        # save current state and model
        if self.accelerator.is_main_process and save_ckpt:
            ckpts_save_path = os.path.join(self.output_dir,'ckpts',self.exp_name, f'epoch_{epoch}_step_{self.global_step-1}')
            os.makedirs(ckpts_save_path,exist_ok=True)
            self.accelerator.save_state(ckpts_save_path, safe_serialization=False)
        self.accelerator.wait_for_everyone()
        
        if epoch < self.least_eval_epoch:
            return
        results_save_path = os.path.join(self.output_dir,'results',self.exp_name, f'epoch_{epoch}_step_{self.global_step-1}')        
        self.eval(results_save_path, epoch=epoch)

    def infer(self):
        self.model.eval()
        # unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model = self.unwrapped_model 
        
        results_save_path = self.output_dir
        if self.accelerator.is_main_process:
            os.makedirs(results_save_path,exist_ok=True)
        
        self.accelerator.print('Using following threshold(s): ', self.conf_thresh)
        for thresh in self.conf_thresh:
            if self.accelerator.is_main_process or self.distributed_infer:
                self.inference_func(model = unwrapped_model, 
                        infer_dataloader = self.infer_dataloader, 
                        conf_thresh = thresh,
                        results_save_path = os.path.join(results_save_path,f'thresh_{thresh}'),
                        distributed = self.distributed_infer,
                        accelerator = self.accelerator)
            self.accelerator.wait_for_everyone()


def flatten_dict(d, parent_key='', sep='-'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

