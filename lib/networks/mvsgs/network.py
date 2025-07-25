import torch
import torch.nn as nn
from torch.nn import functional as F
from .feature_net import FeatureNet
from .cost_reg_net import CostRegNet, MinCostRegNet
from . import utils
from lib.config import cfg
from .gs import GS
from lib.gaussian_renderer import render
import os
import imageio
import numpy as np
import PIL
import cv2
from .utils import write_cam, save_pfm, visualize_depth
import copy

from plyfile import PlyData, PlyElement

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class Network(nn.Module):
    
    def __init__(self,):
        super(Network, self).__init__()
        self.feature_net = FeatureNet()
        for i in range(cfg.mvsgs.cas_config.num):
            if i == 0:
                cost_reg_l = MinCostRegNet(int(32 * (2**(-i))*2))
            else:
                cost_reg_l = CostRegNet(int(32 * (2**(-i)*2)))
            setattr(self, f'cost_reg_{i}', cost_reg_l)
            gs_l = GS(feat_ch=cfg.mvsgs.cas_config.gs_model_feat_ch[i]+3)
            setattr(self, f'gs_{i}', gs_l)

    def render_rays(self, rays, **kwargs):
        level, batch, im_feat, feat_volume, gs_model, size = kwargs['level'], kwargs['batch'], kwargs['im_feat'], kwargs['feature_volume'], kwargs['gs_model'], kwargs['size']
        world_xyz, uvd, z_vals = utils.sample_along_depth(rays, N_samples=cfg.mvsgs.cas_config.num_samples[level], level=level)
        # uvd就是像素坐标的基础上加了一个维度的深度，其中num_samples决定了有多少个深度
        B, N_rays, N_samples = world_xyz.shape[:3]
        rgbs = utils.unpreprocess(batch['src_inps'], render_scale=cfg.mvsgs.cas_config.render_scale[level])
        # 反预处理，原本预处理到-1 - 1 的，现在回来
        up_feat_scale = cfg.mvsgs.cas_config.render_scale[level] / cfg.mvsgs.cas_config.im_ibr_scale[level]
        if up_feat_scale != 1.:
            B, S, C, H, W = im_feat.shape
            im_feat = F.interpolate(im_feat.reshape(B*S, C, H, W), None, scale_factor=up_feat_scale, align_corners=True, mode='bilinear').view(B, S, C, int(H*up_feat_scale), int(W*up_feat_scale))
        # im_feat高维特征8维，rgbs3维，合成11维
        img_feat_rgb = torch.cat((im_feat, rgbs), dim=2)
        H_O, W_O = kwargs['batch']['src_inps'].shape[-2:]
        B, H, W = len(uvd), int(H_O * cfg.mvsgs.cas_config.render_scale[level]), int(W_O * cfg.mvsgs.cas_config.render_scale[level])
        uvd[..., 0], uvd[..., 1] = (uvd[..., 0]) / (W-1), (uvd[..., 1]) / (H-1)
        # 归一化一下，像素坐标变成0-1
        vox_feat = utils.get_vox_feat(uvd.reshape(B, -1, 3), feat_volume)
        # feat_volume特征体，这里的主要目的是得到“体素”的特征，也就是每一个像素坐标（带深度）的的特征值
        # 这个应该只有目标视图的特征，后面一步应该就是获得了参考视图的特征
        img_feat_rgb_dir = utils.get_img_feat(world_xyz, img_feat_rgb, batch, self.training, level)
        # feat:插值得到在参考视图下每一个像素点的特征和rgb 8+3
        # ray_diff:每个点的差距除以均值；每个单独相乘之后三个加起来 3+1
        net_output = gs_model(vox_feat, img_feat_rgb_dir, z_vals, batch, size, level)
        return net_output


    def batchify_rays(self, rays, **kwargs):
        ret = self.render_rays(rays, **kwargs)
        return ret


    def forward_feat(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        feat2, feat1, feat0 = self.feature_net(x)
        feats = {
                'level_2': feat0.reshape((B, S, feat0.shape[1], H, W)),
                'level_1': feat1.reshape((B, S, feat1.shape[1], H//2, W//2)),
                'level_0': feat2.reshape((B, S, feat2.shape[1], H//4, W//4)),
                }
        return feats

    def forward_render(self, ret, batch):
        B, _, _, H, W = batch['src_inps'].shape
        rgb = ret['rgb'].reshape(B, H, W, 3).permute(0, 3, 1, 2)
        rgb = self.cnn_renderer(rgb)
        ret['rgb'] = rgb.permute(0, 2, 3, 1).reshape(B, H*W, 3)


    def forward(self, batch,idx):
        B, _, _, H_img, W_img = batch['src_inps'].shape
        if not cfg.save_video:
            feats = self.forward_feat(batch['src_inps'])
            # feats: level_0,level_1，level_2分别是：下采样4倍32维，下采样2倍16维，不下采样8维
            ret = {}
            depth, std, near_far = None, None, None
            for i in range(cfg.mvsgs.cas_config.num):
                H, W = int(H_img*cfg.mvsgs.cas_config.render_scale[i]), int(W_img*cfg.mvsgs.cas_config.render_scale[i])
                try:
                    feature_volume, depth_values, near_far = utils.build_feature_volume(
                            feats[f'level_{i}'],
                            batch,
                            D=cfg.mvsgs.cas_config.volume_planes[i],
                            depth=depth,
                            std=std,
                            near_far=near_far,
                            level=i)
                except:
                    feature_volume, depth_values, near_far = utils.build_feature_volume(
                            feats[f'level_{i}'],
                            batch,
                            D=cfg.mvsgs.cas_config.volume_planes[i],
                            depth=std,
                            std=std,
                            near_far=near_far,
                            level=i)
                feature_volume, depth_prob = getattr(self, f'cost_reg_{i}')(feature_volume)
                # 进入一个CostRegNet得到可能体depth_prob
                depth, std = utils.depth_regression(depth_prob, depth_values, i, batch)
                # 回归得到深度
                img_origin = cv2.cvtColor(np.uint8((batch['tar_img'][0].cpu().detach().numpy()) * 255),cv2.COLOR_RGB2BGR)
                img_origin = cv2.resize(img_origin,(depth.shape[2],depth.shape[1]))


                # depth_path = f"data/SPARSE/scan24/depth_dust3r/{idx:04d}.tiff"
                # depth_dust3r = cv2.imread(depth_path,cv2.IMREAD_ANYDEPTH)
                # img = cv2.resize(img, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)
                depth_dust3r = batch['depth_dust3r'][0].cpu().detach().numpy()
                original_width , original_height = depth_dust3r.shape[1] , depth_dust3r.shape[0]
                target_width , target_height = depth.shape[2] , depth.shape[1]

                original_ratio = original_width / original_height
                target_ratio = target_width / target_height
                if original_ratio > target_ratio:
                    # 原始图像更宽，按高度缩放
                    scale = target_height / original_height
                    new_width = int(original_width * scale)
                    new_height = target_height
                else:
                    # 原始图像更高，按宽度缩放
                    scale = target_width / original_width
                    new_width = target_width
                    new_height = int(original_height * scale)
                depth_dust3r = cv2.resize(
                    depth_dust3r, 
                    (new_width, new_height), 
                    interpolation=cv2.INTER_AREA  # 缩小用INTER_AREA效果好
                )

                start_x = (new_width - target_width) // 2
                start_y = (new_height - target_height) // 2

                depth_dust3r = depth_dust3r[
                    start_y:start_y + target_height,
                    start_x:start_x + target_width
                ]
                depth_dust3r[depth_dust3r == 0] = 10
                depth_dust3r = torch.tensor(depth_dust3r)
                if cfg.mvsgs.cas_config.depth_inv[i]:
                    depth_dust3r = 1. / torch.clamp_min(depth_dust3r, 1e-6)
                depth_dust3r_vis = np.uint8(np.clip(cv2.cvtColor(depth_dust3r.cpu().detach().numpy() * 100, cv2.COLOR_GRAY2BGR),0,255))

                depth_vis = cv2.cvtColor(depth[0].cpu().detach().numpy() * 100*cfg.mvsgs.scale_factor , cv2.COLOR_GRAY2BGR)
                depth_vis = np.uint8(np.clip(depth_vis,0,255))
                vis_add_img_dust3r = cv2.addWeighted(img_origin, 0.1, depth_dust3r_vis, 0.9, 0)
                vis_add_img = cv2.addWeighted(img_origin, 0.1, depth_vis, 0.9, 0)
                if cfg.mvsgs.cas_config.depth_inv[i]:
                    depth_dust3r_torch_align = (depth_dust3r / cfg.mvsgs.scale_factor).unsqueeze(0).cuda()
                    std_dust3r = torch.ones_like(std)*10000
                else:
                    depth_dust3r_torch_align = (depth_dust3r * cfg.mvsgs.scale_factor).unsqueeze(0).cuda()
                    std_dust3r = torch.ones_like(std)/10000
                
                if not cfg.mvsgs.cas_config.render_if[i]:
                    continue

                # depth = depth_dust3r_torch_align
                # std = std_dust3r


                rays = utils.build_rays(depth, std, batch, self.training, near_far, i)
                # rays_dust3r = utils.build_rays(depth_dust3r_torch_align, std_dust3r, batch, self.training, near_far, i)

                
                # 在原本ray的基础上增加了rays_near_far和near_far
                im_feat_level = cfg.mvsgs.cas_config.render_im_feat_level[i]
                output = self.batchify_rays(
                        rays=rays,
                        feature_volume=feature_volume,
                        batch=batch,
                        im_feat=feats[f'level_{im_feat_level}'],
                        gs_model=getattr(self, f'gs_{i}'),
                        level=i,
                        size=(H,W)
                        )
                ret_i = {}
                world_xyz, rot_out, scale_out, opacity_out, color_out, rgb_vr = output

                xyz = world_xyz.squeeze().detach().cpu().numpy()
                normals = np.zeros_like(xyz)
                color = color_out.squeeze().detach().cpu().numpy()
                C0 = 0.28209479177387814
                color = (color - 0.5) / C0
                opacities = inverse_sigmoid(opacity_out).squeeze(0).detach().cpu().numpy()
                scale = torch.log(scale_out).squeeze().detach().cpu().numpy()
                rotation = rot_out.squeeze().detach().cpu().numpy()

                l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
                # All channels except the 3 DC
                for ii in range(3):
                    l.append('f_dc_{}'.format(ii))
                l.append('opacity')
                for ii in range(3):
                    l.append('scale_{}'.format(ii))
                for ii in range(4):
                    l.append('rot_{}'.format(ii))

                dtype_full = [(attribute, 'f4') for attribute in l]


                elements = np.empty(xyz.shape[0], dtype=dtype_full)
                scale[scale > -5] = -5
                attributes = np.concatenate((xyz, normals, color, opacities, scale, rotation), axis=1)
                # elements[:] = list(map(tuple, attributes))
                # el = PlyElement.describe(elements, 'vertex')
                # PlyData([el]).write(f"test_{idx}.ply")

                GS_scene = {}
                GS_scene["data"] = attributes
                GS_scene['dtype_full'] = dtype_full






                render_novel = []
                for b_i in range(B):
                    render_novel_i_0 = render(batch[f'novel_view{i}'], b_i, world_xyz[b_i], color_out[b_i], rot_out[b_i], scale_out[b_i], opacity_out[b_i], bg_color=cfg.mvsgs.bg_color)
                    if cfg.mvsgs.reweighting: 
                        render_novel_i = (render_novel_i_0 + rgb_vr[b_i]*4) / 5
                    else:
                        render_novel_i = (render_novel_i_0 + rgb_vr[b_i]) / 2
                    render_novel.append(render_novel_i)
                render_novel = torch.stack(render_novel)
                ret_i.update({'rgb': render_novel.flatten(2).permute(0,2,1)})
                if cfg.mvsgs.cas_config.depth_inv[i]:
                    ret_i.update({'depth_mvs': 1./depth})
                else:
                    ret_i.update({'depth_mvs': depth})
                ret_i.update({'std': std})
                if ret_i['rgb'].isnan().any():
                    __import__('ipdb').set_trace()
                ret.update({key+f'_level{i}': ret_i[key] for key in ret_i})
                
                if cfg.save_ply:
                    result_dir = cfg.dir_ply
                    os.makedirs(result_dir, exist_ok = True)
                    depth_origin = copy.deepcopy(depth)
                    depth = F.interpolate(depth.unsqueeze(1),size=(H,W)).squeeze(1)
                    for b_i in range(B):
                        scan_dir = os.path.join(result_dir, batch['meta']['scene'][b_i])
                        os.makedirs(scan_dir, exist_ok = True)
                        img_dir = os.path.join(scan_dir, 'images')
                        os.makedirs(img_dir, exist_ok = True)
                        img_path = os.path.join(img_dir, '{}_{}_{}.png'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item()))
                        img = render_novel[b_i].permute(1,2,0).detach().cpu().numpy()
                        img = (img*255).astype(np.uint8)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(img_path, img)
                        cam_dir = os.path.join(scan_dir, 'cam')
                        os.makedirs(cam_dir, exist_ok = True)
                        cam_path = os.path.join(cam_dir, '{}_{}_{}.txt'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item()))
                        ixt = batch['tar_ixt'].detach().cpu().numpy()[b_i]
                        ext = batch['tar_ext'].detach().cpu().numpy()[b_i]
                        ext[:3,3] /= cfg.mvsgs.scale_factor
                        write_cam(cam_path, ixt, ext)

                        depth /= cfg.mvsgs.scale_factor
                        depth_dir = os.path.join(scan_dir, 'depth')
                        os.makedirs(depth_dir, exist_ok = True)
                        depth_path = os.path.join(depth_dir, '{}_{}_{}.pfm'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item()))
                        depth_vis = depth[b_i].detach().cpu().numpy()
                        save_pfm(depth_path, depth_vis)
                        
                        depth_minmax = [
                            batch["near_far"].min().detach().cpu().numpy()/cfg.mvsgs.scale_factor,
                            batch["near_far"].max().detach().cpu().numpy()/cfg.mvsgs.scale_factor,
                        ]
                        rendered_depth_vis, _ = visualize_depth(depth_vis, depth_minmax)
                        rendered_depth_vis = rendered_depth_vis.permute(1,2,0).detach().cpu().numpy()
                        depth_vis_path = os.path.join(depth_dir, '{}_{}_{}.png'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item()))
                        imageio.imwrite(depth_vis_path, (rendered_depth_vis*255.).astype(np.uint8))
            return ret , GS_scene
        else:
            pred_rgb_nb_list = []
            for v_i, meta in enumerate(batch['rendering_video_meta']):
                batch['tar_ext'][:,:3] = meta['tar_ext'][:,:3]
                batch['rays_0'] = meta['rays_0']
                batch['rays_1'] = meta['rays_1']
                feats = self.forward_feat(batch['src_inps'])
                depth, std, near_far = None, None, None
                for i in range(cfg.mvsgs.cas_config.num):
                    H, W = int(H_img*cfg.mvsgs.cas_config.render_scale[i]), int(W_img*cfg.mvsgs.cas_config.render_scale[i])
                    feature_volume, depth_values, near_far = utils.build_feature_volume(
                            feats[f'level_{i}'],
                            batch,
                            D=cfg.mvsgs.cas_config.volume_planes[i],
                            depth=depth,
                            std=std,
                            near_far=near_far,
                            level=i)
                    feature_volume, depth_prob = getattr(self, f'cost_reg_{i}')(feature_volume)
                    depth, std = utils.depth_regression(depth_prob, depth_values, i, batch)
                    if not cfg.mvsgs.cas_config.render_if[i]:
                        continue
                    rays = utils.build_rays(depth, std, batch, self.training, near_far, i)
                    im_feat_level = cfg.mvsgs.cas_config.render_im_feat_level[i]
                    output = self.batchify_rays(
                            rays=rays,
                            feature_volume=feature_volume,
                            batch=batch,
                            im_feat=feats[f'level_{im_feat_level}'],
                            gs_model=getattr(self, f'gs_{i}'),
                            level=i,
                            size=(H,W)
                            )
                    if i == cfg.mvsgs.cas_config.num-1:
                        world_xyz, rot_out, scale_out, opacity_out, color_out, rgb_vr = output
                        for b_i in range(B):
                            render_novel_i_0 = render(meta, b_i, world_xyz[b_i], color_out[b_i], rot_out[b_i], scale_out[b_i], opacity_out[b_i], bg_color=cfg.mvsgs.bg_color)
                            if cfg.mvsgs.reweighting: 
                                render_novel_i = (render_novel_i_0 + rgb_vr*4) / 5
                            else:
                                render_novel_i = (render_novel_i_0 + rgb_vr) / 2
                            render_novel_i = render_novel_i[b_i].permute(1,2,0)
                            if cfg.mvsgs.eval_center:
                                H_crop, W_crop = int(H_img*0.1), int(W_img*0.1)
                                render_novel_i = render_novel_i[H_crop:-H_crop, W_crop:-W_crop,:]
                            if v_i == 0:
                                pred_rgb_nb_list.append([(render_novel_i.data.cpu().numpy()*255).astype(np.uint8)])
                            else:
                                pred_rgb_nb_list[b_i].append((render_novel_i.data.cpu().numpy()*255).astype(np.uint8))
                            img_dir = os.path.join(cfg.result_dir, '{}_{}'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item()))
                            os.makedirs(img_dir,exist_ok=True)
                            save_path = os.path.join(img_dir,f'{len(pred_rgb_nb_list[b_i])}.png')
                            PIL.Image.fromarray((render_novel_i.data.cpu().numpy()*255).astype(np.uint8)).save(save_path)
            for b_i in range(B):
                video_path = os.path.join(cfg.result_dir, '{}_{}_{}.mp4'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item()))
                imageio.mimwrite(video_path, np.stack(pred_rgb_nb_list[b_i]), fps=10, quality=10)

