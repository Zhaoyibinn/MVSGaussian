import numpy as np
import os
from lib.config import cfg
import imageio
import cv2
import random
from lib.utils import data_utils
import torch
from lib.datasets import mvsgs_utils
from lib.utils.video_utils import *

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        self.input_h_w = kwargs['input_h_w']
        if 'scene' in kwargs and kwargs['scene']:
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.scale_factor = cfg.mvsgs.scale_factor
        self.build_metas()
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = [0.0, 0.0, 0.0]
        self.scale = 1.0

    def build_metas(self):
        if len(self.scenes) == 0:
            ### set your own scenes
            scenes = ['room']
        else:
            scenes = self.scenes
        self.scene_infos = {}
        self.metas = []
        
        for scene in scenes:

            pose_bounds = np.load(os.path.join(self.data_root, scene, 'poses_bounds.npy')) # c2w, -u, r, -t
            poses = pose_bounds[:, :15].reshape((-1, 3, 5))
            c2ws = np.eye(4)[None].repeat(len(poses), 0)
            c2ws[:, :3, 0], c2ws[:, :3, 1], c2ws[:, :3, 2], c2ws[:, :3, 3] = poses[:, :3, 1], poses[:, :3, 0], -poses[:, :3, 2], poses[:, :3, 3]
            ixts = np.eye(3)[None].repeat(len(poses), 0)
            ixts[:, 0, 0], ixts[:, 1, 1] = poses[:, 2, 4], poses[:, 2, 4]
            ixts[:, 0, 2], ixts[:, 1, 2] = poses[:, 1, 4]/2., poses[:, 0, 4]/2.
            # c2ws位姿，ixts内参
            img_paths = sorted([item for item in os.listdir(os.path.join(self.data_root, scene, 'images')) if ('.png' in item) or ('.jpg' in item) or ('.JPG' in item)])
            depth_ranges = pose_bounds[:, -2:]
            scene_info = {'ixts': ixts.astype(np.float32), 'c2ws': c2ws.astype(np.float32), 'image_names': img_paths, 'depth_ranges': depth_ranges.astype(np.float32)}
            scene_info['scene_name'] = scene


            depth_dust3r_path = os.path.join(self.data_root,scene,'depth_dust3r')
            depth_dust3r_imgs = []
            if os.path.exists(depth_dust3r_path):
                for depth_dust3r_img in sorted(os.listdir(depth_dust3r_path)):
                    depth_dust3r_imgs.append(cv2.imread(os.path.join(depth_dust3r_path,depth_dust3r_img),cv2.IMREAD_ANYDEPTH))
            assert len(depth_dust3r_imgs) == len(img_paths) ,"图像数量和Dust3R深度图数量不一致 请检查"
            
            self.depth_dust3r_imgs = depth_dust3r_imgs

            self.scene_infos[scene] = scene_info
            img_len = len(img_paths)
            ### set your own train_ids and render_ids
            # train_ids = [j for j in range(img_len) if j not in render_ids]
            train_ids = [j for j in range(img_len)]
            if self.split == 'train':
                render_ids = train_ids
            else:
                # render_ids = [j for j in range(img_len//16, img_len, img_len//8)]
                render_ids = train_ids
            # zyb标记，全部渲染
            #
            c2ws = c2ws[train_ids]
            for i in render_ids:
                c2w = scene_info['c2ws'][i]
                distance = np.linalg.norm((c2w[:3, 3][None] - c2ws[:, :3, 3]), axis=-1)
                argsorts = distance.argsort()
                argsorts = argsorts[1:] if i in train_ids else argsorts
                if self.split == 'train':
                    src_views = [train_ids[i] for i in argsorts[:cfg.mvsgs.train_input_views[1]+1]]
                else:
                    src_views = [train_ids[i] for i in argsorts[:cfg.mvsgs.test_input_views]]
                self.metas += [(scene, i, src_views)]
    
    def get_video_rendering_path(self, ref_poses, mode, near_far, train_c2w_all, n_frames=60, rads_scale = 1.25):
        # loop over batch
        poses_paths = []
        ref_poses = ref_poses[None]
        for batch_idx, cur_src_poses in enumerate(ref_poses):
            if mode == 'interpolate':
                # convert to c2ws
                pose_square = torch.eye(4).unsqueeze(0).repeat(cur_src_poses.shape[0], 1, 1)
                cur_src_poses = torch.from_numpy(cur_src_poses)
                pose_square[:, :3, :] = cur_src_poses[:,:3]
                cur_c2ws = pose_square.double().inverse()[:, :3, :].to(torch.float32).cpu().detach().numpy()
                cur_path = get_interpolate_render_path(cur_c2ws, n_frames)
            elif mode == 'spiral':
                cur_c2ws_all = train_c2w_all
                cur_near_far = near_far.tolist()
                cur_path = get_spiral_render_path(cur_c2ws_all, cur_near_far, rads_scale=rads_scale, N_views=n_frames)
            else:
                raise Exception(f'Unknown video rendering path mode {mode}')

            # convert back to extrinsics tensor
            cur_w2cs = torch.tensor(cur_path).inverse()[:, :3].to(torch.float32)
            poses_paths.append(cur_w2cs)

        poses_paths = torch.stack(poses_paths, dim=0)
        return poses_paths

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views = self.metas[index]
        if self.split == 'train':
            if np.random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views, input_views_num)
        scene_info = self.scene_infos[scene]
        # ext，ixt内参
        tar_img, tar_mask, tar_ext, tar_ixt = self.read_tar(scene_info, tar_view)
        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)
        # src_inps就是imgs的stack
        ret = {'src_inps': src_inps.transpose(0, 3, 1, 2),
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        # ret.update({'img_name': self.scene_infos[scene]['image_names'][index]})



        if self.split != 'train':
            ret.update({'tar_img': tar_img,
                        'tar_mask': tar_mask})

        H, W = tar_img.shape[:2]
        depth_ranges = np.array(scene_info['depth_ranges'])
        near_far = np.array([depth_ranges[:, 0].min().item()*self.scale_factor, depth_ranges[:, 1].max().item()*self.scale_factor]).astype(np.float32)
        # near_far = scene_info['depth_ranges'][tar_view]
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view, 'frame_id': 0}})
        ret.update({'depth_dust3r': self.depth_dust3r_imgs[index]})
        for i in range(cfg.mvsgs.cas_config.num):
            rays, rgb, msk = mvsgs_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk})
            s = cfg.mvsgs.cas_config.volume_scale[i]
            ret['meta'].update({f'h_{i}': int(H*s), f'w_{i}': int(W*s)})
            
        R = np.array(tar_ext[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(tar_ext[:3, 3], np.float32)
        for i in range(cfg.mvsgs.cas_config.num):
            h, w = H*cfg.mvsgs.cas_config.render_scale[i], W*cfg.mvsgs.cas_config.render_scale[i]
            tar_ixt_ = tar_ixt.copy()
            tar_ixt_[:2,:] *= cfg.mvsgs.cas_config.render_scale[i]
            FovX = data_utils.focal2fov(tar_ixt_[0, 0], w)
            FovY = data_utils.focal2fov(tar_ixt_[1, 1], h)
            projection_matrix = data_utils.getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=tar_ixt_, h=h, w=w).transpose(0, 1)
            world_view_transform = torch.tensor(data_utils.getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]
            novel_view_data = {
                'FovX':  torch.FloatTensor([FovX]),
                'FovY':  torch.FloatTensor([FovY]),
                'width': w,
                'height': h,
                'world_view_transform': world_view_transform,
                'full_proj_transform': full_proj_transform,
                'camera_center': camera_center
            }
            ret[f'novel_view{i}'] = novel_view_data    
        
        if cfg.save_video:
            rendering_video_meta = []
            render_path_mode = 'spiral'
            train_c2w_all = np.linalg.inv(src_exts)
            poses_paths = self.get_video_rendering_path(src_exts, render_path_mode, near_far, train_c2w_all, n_frames=60)
            for pose in poses_paths[0]:
                R = np.array(pose[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
                T = np.array(pose[:3, 3], np.float32)
                FovX = data_utils.focal2fov(tar_ixt[0, 0], W)
                FovY = data_utils.focal2fov(tar_ixt[1, 1], H)
                projection_matrix = data_utils.getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=tar_ixt, h=H, w=W).transpose(0, 1)
                world_view_transform = torch.tensor(data_utils.getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]
                rendering_meta = {
                    'FovX':  torch.FloatTensor([FovX]),
                    'FovY':  torch.FloatTensor([FovY]),
                    'width': W,
                    'height': H,
                    'world_view_transform': world_view_transform,
                    'full_proj_transform': full_proj_transform,
                    'camera_center': camera_center,
                    'tar_ext': pose
                }
                for i in range(cfg.mvsgs.cas_config.num):
                    tar_ext[:3] = pose
                    rays, _, _ = mvsgs_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
                    rendering_meta.update({f'rays_{i}': rays})
                rendering_video_meta.append(rendering_meta)
            ret['rendering_video_meta'] = rendering_video_meta
        #     ret中的0和1就是下采样了的和没有下采样的
        return ret

    def read_src(self, scene, src_views):
        '''
        类似的，就是把多个图像的每个东西stack之后一起返回
        '''
        src_ids = src_views
        ixts, exts, imgs = [], [], []
        for idx in src_ids:
            img, orig_size = self.read_image(scene, idx)
            imgs.append(((img/255.)*2-1).astype(np.float32))
            ixt, ext, _ = self.read_cam(scene, idx, orig_size)
            ixts.append(ixt)
            exts.append(ext)
        return np.stack(imgs), np.stack(exts), np.stack(ixts)

    def read_tar(self, scene, view_idx):
        '''
        return:
        img 缩放成640*960的RBG图
        mask
        ext 改正后的pose
        ixt 改正后的内参
        '''
        img, orig_size = self.read_image(scene, view_idx)
        img = (img/255.).astype(np.float32)
        ixt, ext, _ = self.read_cam(scene, view_idx, orig_size)
        mask = np.ones_like(img[..., 0]).astype(np.uint8)
        return img, mask, ext, ixt

    def read_cam(self, scene, view_idx, orig_size):
        ext = scene['c2ws'][view_idx].astype(np.float32)
        ixt = scene['ixts'][view_idx].copy()
        ixt[0] *= self.input_h_w[1] / orig_size[0]
        ixt[1] *= self.input_h_w[0] / orig_size[1]
        w2c = np.linalg.inv(ext)
        w2c[:3,3] *= self.scale_factor
        return ixt, w2c, 1

    def read_image(self, scene, view_idx):
        image_path = os.path.join(self.data_root, scene['scene_name'], 'images', scene['image_names'][view_idx])
        img = (np.array(imageio.imread(image_path))).astype(np.float32)
        img = img[:,:,:3]
        orig_size = img.shape[:2][::-1]
        # img = cv2.resize(img, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)

        original_width , original_height = img.shape[1] , img.shape[0]
        target_width , target_height = self.input_h_w[::-1][0] , self.input_h_w[::-1][1]

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
        img = cv2.resize(
            img, 
            (new_width, new_height), 
            interpolation=cv2.INTER_AREA  # 缩小用INTER_AREA效果好
        )

        start_x = (new_width - target_width) // 2
        start_y = (new_height - target_height) // 2

        img = img[
            start_y:start_y + target_height,
            start_x:start_x + target_width
        ]


        return np.array(img), orig_size

    def __len__(self):
        return len(self.metas)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K
