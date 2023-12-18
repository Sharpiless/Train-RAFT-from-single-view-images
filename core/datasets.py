# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from utils.mpi.homography_sampler import HomographySample
from utils.mpi import mpi_rendering
from geometry import transformation_from_parameters

import os
import cv2
import math
import random
from glob import glob
import os.path as osp

from model.AdaMPI import MPIPredictor

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

from utils.utils import (
    image_to_tensor,
    disparity_to_tensor,
    render_3dphoto_dynamic,
)


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def gen_swing_path(num_frames=90, r_x=0.14, r_y=0., r_z=0.10):
    "Return a list of matrix [4, 4]"
    t = torch.arange(num_frames) / (num_frames - 1)
    poses = torch.eye(4).repeat(num_frames, 1, 1)
    poses[:, 0, 3] = r_x * torch.sin(2. * math.pi * t)
    poses[:, 1, 3] = r_y * torch.cos(2. * math.pi * t)
    poses[:, 2, 3] = r_z * (torch.cos(2. * math.pi * t) - 1.)
    return poses.unbind()

def generate_random_pose(base_motions=[0.05, 0.05, 0.05]):
    scx = ((-1)**random.randrange(2))
    scy = ((-1)**random.randrange(2))
    scz = ((-1)**random.randrange(2))
    if base_motions[0] == 0.05:
        scz = -1 # most cameras move forward in kitti
    else:
        scx = scx * 0.5 # object motion
        scy = scy * 0.5
        scz = scz * 0.5
        
    # Random scalars excluding zeros / very small motions
    cx = (random.random()*0.1+base_motions[0]) * scx
    cy = (random.random()*0.1+base_motions[1]) * scy
    cz = (random.random()*0.15+base_motions[2]) * scz
    camera_mot = [cx*0.5, cy*0.5, cz]

    # generate random triplet of Euler angles
    # Random sign
    sax = ((-1)**random.randrange(2))
    say = ((-1)**random.randrange(2))
    saz = ((-1)**random.randrange(2))
    if not base_motions[0] == 0.05:
        sax = sax * 0.5
        say = say * 0.5
        saz = saz * 0.5
    # Random angles in -pi/18,pi/18, excluding -pi/36,pi/36 to avoid zeros / very small rotations
    ax = (random.random()*math.pi / 36.0) * sax
    ay = (random.random()*math.pi / 36.0) * say
    az = (random.random()*math.pi / 36.0) * saz
    camera_ang = [ax*0.2, ay*0.2, az*0.2]

    axisangle = torch.from_numpy(
        np.array([[camera_ang]], dtype=np.float32)).cuda().float()
    translation = torch.from_numpy(
        np.array([[camera_mot]])).cuda().float()

    cam_ext = transformation_from_parameters(
        axisangle, translation)[0]
    return cam_ext

def generate_random_pose_train():
    scx = ((-1)**random.randrange(2))
    scy = ((-1)**random.randrange(2))
    scz = ((-1)**random.randrange(2))
    # Random scalars in -0.2,0.2, excluding -0.1,0.1 to avoid zeros / very small motions
    cx = (random.random()*0.1+0.1) * scx
    cy = (random.random()*0.1+0.1) * scy
    cz = (random.random()*0.1+0.1) * scz
    # cz = (random.random()*0.2)
    camera_mot = [cx*0.5, cy*0.5, cz]

    # generate random triplet of Euler angles
    # Random sign
    sax = ((-1)**random.randrange(2))
    say = ((-1)**random.randrange(2))
    saz = ((-1)**random.randrange(2))
    # Random angles in -pi/18,pi/18, excluding -pi/36,pi/36 to avoid zeros / very small rotations
    ax = (random.random()*math.pi / 36.0 + math.pi / 36.0) * sax
    ay = (random.random()*math.pi / 36.0 + math.pi / 36.0) * say
    az = (random.random()*math.pi / 36.0 + math.pi / 36.0) * saz
    camera_ang = [ax*0.2, ay*0.2, az*0.2]

    axisangle = torch.from_numpy(
        np.array([[camera_ang]], dtype=np.float32)).cuda()
    translation = torch.from_numpy(np.array([[camera_mot]])).cuda()

    # Compute (R|t)
    cam_ext = transformation_from_parameters(axisangle, translation)[0]
    return cam_ext

def render_novel_view_dynamic(
    obj_mask,
    mpi_all_rgb_src,
    mpi_all_sigma_src,
    disparity_all_src,
    G_tgt_src,
    K_src_inv,
    K_tgt,
    K_src,
    src_pose,
    homography_sampler,
    hard_flow=False
):
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid.to(K_src.dtype),
        disparity_all_src,
        K_src_inv
    )

    xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
        xyz_src_BS3HW.to(K_src.dtype),
        G_tgt_src.to(K_src.dtype)
    )

    mpi_depth_src = torch.reciprocal(disparity_all_src)
    B, S = disparity_all_src.size()
    xyz_tgt = xyz_tgt_BS3HW.reshape(
        B * S, 3, -1) / mpi_depth_src[0].unsqueeze(1).unsqueeze(2)
    # BSx3xHW torch.Size([64, 3, 98304])
    meshgrid_tgt = torch.matmul(K_tgt, xyz_tgt)
    meshgrid_src = homography_sampler.meshgrid.unsqueeze(
        0).unsqueeze(1).repeat(B, S, 1, 1, 1).reshape(B * S, 3, -1)
    mpi_flow_src = meshgrid_src - meshgrid_tgt
    H, W = mpi_all_rgb_src.shape[-2:]
    mpi_flow_src = mpi_flow_src.reshape(B, S, 3, H, W)[:, :, :2]
    obj_mask = obj_mask.unsqueeze(1).repeat(B, S, 1, 1, 1)

    tgt_imgs_syn, tgt_depth_syn, _, flow_syn, obj_mask = mpi_rendering.render_tgt_rgb_depth(
        homography_sampler,
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        disparity_all_src,
        xyz_tgt_BS3HW,
        xyz_src_BS3HW,
        G_tgt_src,
        K_src_inv,
        K_tgt,
        mpi_flow_src,
        use_alpha=False,
        is_bg_depth_inf=False,
        hard_flow=hard_flow,
        obj_mask=obj_mask
    )
    flow_syn = torch.clip(flow_syn, -512, 512)
    return tgt_imgs_syn, tgt_depth_syn, flow_syn, obj_mask

class MPIFlowDataset(data.Dataset):
    def __init__(self, args, aug_params=None, sparse=False, image_root='datasets/custom'):
        self.args = args
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
                
        self.image_list = sorted(os.listdir(os.path.join(image_root, "images")))
        self.image_root = image_root

        # load pretrained model
        ckpt = torch.load(args.ckpt_path)
        self.model = MPIPredictor(
            width=args.image_size[0],
            height=args.image_size[1],
            num_planes=ckpt['num_planes'],
        )
        self.model.load_state_dict(ckpt['weight'])
        self.model = self.model.cuda().half()
        self.model = self.model.eval()

    def __getitem__(self, index):

        index = index % len(self.image_list)

        # img1 = frame_utils.read_gen(
        #     os.path.join(self.image_root, "images", self.image_list[index])
        # )
        flow, img1, img2 = self.construct_mpiflow(index)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()

    def construct_mpiflow(self, index, mask_thresh=0.99):
        
        # render 3D photo
        K = torch.tensor([
            [0.58, 0, 0.5],
            [0, 0.58, 0.5],
            [0, 0, 1]
        ]).cuda().half()
        K[0, :] *= self.args.image_size[1] # width
        K[1, :] *= self.args.image_size[0] # height
        K = K.unsqueeze(0)
        
        image = image_to_tensor(os.path.join(self.image_root, "images", self.image_list[index])).cuda().half()  # [1,3,h,w]
        obj_mask = image_to_tensor(os.path.join(self.image_root, "masks", self.image_list[index])).cuda().half()  # [1,3,h,w]
        disp = disparity_to_tensor(os.path.join(self.image_root, "disps", self.image_list[index])).cuda().half() # [1,1,h,w]
        image = F.interpolate(image, size=(self.args.image_size[0], self.args.image_size[1]),
                            mode='bilinear', align_corners=True)
        obj_mask = F.interpolate(obj_mask, size=(self.args.image_size[0], self.args.image_size[1]),
                            mode='bilinear', align_corners=True)
        disp = F.interpolate(disp, size=(self.args.image_size[0], self.args.image_size[1]),
                            mode='bilinear', align_corners=True)
        with torch.no_grad():
            mpi_all_src, disparity_all_src = self.model(image, disp)  # [b,s,4,h,w]

        h, w = mpi_all_src.shape[-2:]
        swing_path_list = gen_swing_path()
        src_pose = swing_path_list[0]
        obj_mask_np = obj_mask.squeeze().cpu().numpy()
        # preprocess the predict MPI
        device = mpi_all_src.device
        homography_sampler = HomographySample(h, w, device)
        k_src_inv = torch.inverse(K.to(torch.float64).cpu())
        k_src_inv = k_src_inv.cuda().to(K.dtype)
        mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
        mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
        xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
            homography_sampler.meshgrid.to(K.dtype),
            disparity_all_src,
            k_src_inv,
        )
        _, _, blend_weights, _, _, _ = mpi_rendering.render(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            xyz_src_BS3HW,
            use_alpha=False,
            is_bg_depth_inf=False,
        )
        cam_ext_dynamic = generate_random_pose_train()
        cam_ext = generate_random_pose(base_motions=[0, 0, 0])

        frame, depth, flowA2B, mask = render_novel_view_dynamic(
            obj_mask,
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext.cuda(),
            k_src_inv,
            K,
            K,
            src_pose,
            homography_sampler
        )

        frame_dync, depth_dync, flowA2B_dync, mask_dync = render_novel_view_dynamic(
            1 - obj_mask,
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext_dynamic.cuda(),
            k_src_inv,
            K,
            K,
            src_pose,
            homography_sampler
        )
        frame_np = frame[0].permute(
            1, 2, 0).cpu().numpy().astype(np.float32)   # [b,h,w,3]
        frame_np = np.clip(np.round(frame_np * 255),
                           a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]

        frame_dync_np = frame_dync[0].permute(
            1, 2, 0).cpu().numpy().astype(np.float32)   # [b,h,w,3]
        frame_dync_np = np.clip(np.round(frame_dync_np * 255),
                                a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]
        mask = mask[0].permute(
            1, 2, 0).cpu().squeeze().numpy().astype(np.float32)   # [b,h,w,3]
        mask_dync = mask_dync[0].permute(
            1, 2, 0).cpu().squeeze().numpy().astype(np.float32)   # [b,h,w,3]

        flow_np = flowA2B[0].permute(
            1, 2, 0).contiguous().cpu().numpy().astype(np.float32)   # [b,h,w,3]
        flow_dync_np = flowA2B_dync[0].permute(
            1, 2, 0).contiguous().cpu().numpy().astype(np.float32)  # [b,h,w,3]

        # A2B 光流 mask
        flow_np[obj_mask_np < mask_thresh] = 0
        flow_dync_np[obj_mask_np >= mask_thresh] = 0

        frame_np[mask < mask_thresh] = 255
        frame_dync_np[mask_dync < mask_thresh] = 255
        frame_mix = frame_dync_np.copy()
        frame_mix[mask >= mask_thresh] = frame_np[mask >= mask_thresh]
        flow_mix = flow_dync_np.copy()
        flow_mix[obj_mask_np >= mask_thresh] = flow_np[obj_mask_np >= mask_thresh]

        fill_mask = mask_dync.copy()
        fill_mask[mask >= mask_thresh] = 1

        fill_mask = (fill_mask < mask_thresh).astype(np.int32)

        inpainted = cv2.inpaint(
            frame_mix, fill_mask.astype(np.uint8), 3, cv2.INPAINT_NS)
        src_np = image[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        src_np = np.clip(np.round(src_np * 255),
                        a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]
        return flow_mix, src_np[:, :, ::-1], inpainted[:, :, ::-1]


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    elif args.stage == 'MPI-Flow':
        aug_params = {'spatial_aug_prob': 1.0, 'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': -0.1, 'do_flip': False}
        train_dataset = MPIFlowDataset(args, aug_params)
        # # debug
        # from tqdm import tqdm
        # for i in tqdm(range(5000)):
        #     train_dataset.__getitem__(i)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=1, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader




if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--ckpt_path', type=str,
                        default='adampiweight/adampi_64p.pth')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[512, 768])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    train_loader = fetch_dataloader(args)