import torch
import torch.nn as nn
import numpy as np
from bisect import bisect_right
from utils import temporal_smoothness_theta_penalty, rotation_6d_to_matrix
from lbs import batch_rodrigues
import smplx
import os


class GeodesicLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(GeodesicLoss, self).__init__()
        self.reduction = reduction
        self.eps = 1e-6
        

    def bgdR(self, m1, m2):
        m1 = m1.reshape(-1, 3, 3)
        m2 = m2.reshape(-1, 3, 3)
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
        # cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        # cos = torch.min(cos, m1.new(np.ones(batch)))
        # cos = torch.max(cos, m1.new(np.ones(batch)) * -1)
        # return torch.acos(cos)
        traces = m.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        return dists

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred, ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        if self.reduction == 'batchmean':
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))
        else:
            return theta
        

class Loss(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.device = device
        self.optimize_beta = config['optimization']['opti_beta']
        self.optimize_trans = config['optimization']['opti_trans']
        self.factor = 3 if config['model_architecture']['_type'] == "3D" else 6
        self.use_geodesic = config['loss']['use_geodesic']
        self.include_theta_smoothness = 'theta_smooth' in config['loss']['components']
        self.weights = config['loss']['weights']
        self.components = config['loss']['components']
        self.smpl_gender = config['smpl_gender']        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.geodesic_loss = GeodesicLoss(reduction='mean')

    def forward(self, _input, output, gender):
        batch_size = output.shape[0]
        
        # Handle translation
        if self.optimize_trans:
            pred_trans = output[:, :3]
            start_rot = 3
        else:
            pred_trans = _input[:, :3]
            start_rot = 0

        # Handle beta
        if self.optimize_beta:
            pred_betas = output[:, -10:]
        else:
            pred_betas = _input[:, -10:]
    
        pred_rot_mat = rotation_6d_to_matrix(output[:, start_rot:start_rot+self.factor*24].contiguous().view(-1, 6)).view(batch_size, -1, 3, 3)
        real_rot_mat = batch_rodrigues(_input[:, 3:3+3*24].contiguous().view(-1, 3)).view(batch_size, -1, 3, 3)
        
        # Calculate rotation loss
        if 'geo_loss' in self.components:
            if self.use_geodesic:
                geo_loss = self.geodesic_loss(pred_rot_mat, real_rot_mat)
            else:
                geo_loss = self.mse_loss(pred_rot_mat, real_rot_mat)
        else:
            geo_loss = torch.tensor(0.0).to(self.device)
        
        file_path = os.path.dirname(os.path.abspath(__file__))

        #print("file path ", file_path)
        human_model = smplx.create(f"{os.path.dirname(file_path)}/lidar2mesh-v1/",
                                    gender=self.smpl_gender, 
                                    use_face_contour=False,
                                    ext="npz").to(self.device)
        bpose_real = real_rot_mat[:, 1:, :, :]
        orient_real = real_rot_mat[:, :1, :, :]
        real_smpl = human_model(betas=_input[:, -10:], 
                                return_verts=True, 
                                pose2rot=False,
                                body_pose=bpose_real,
                                global_orient=orient_real,
                                transl=_input[:, :3])
        
        bpose_pred = pred_rot_mat[:, 1:, :, :]
        orient_pred = pred_rot_mat[:, :1, :, :]
        pred_smpl = human_model(betas=pred_betas, 
                                return_verts=True, 
                                pose2rot=False,
                                body_pose=bpose_pred,
                                global_orient=orient_pred,
                                transl=pred_trans)
        
        real_verts, real_joints = real_smpl.vertices, real_smpl.joints
        pred_verts, pred_joints = pred_smpl.vertices, pred_smpl.joints

        # Calculate losses
        joint_loss = self.l1_loss(pred_joints, real_joints) if 'joint_loss' in self.components else torch.tensor(0.0).to(self.device)
        vert_loss = self.l1_loss(pred_verts, real_verts) if 'vert_loss' in self.components else torch.tensor(0.0).to(self.device)
        
        beta_loss = self.mse_loss(pred_betas, _input[:, -10:]) if self.optimize_beta and 'beta_loss' in self.components else torch.tensor(0.0).to(self.device)
        trans_loss = self.mse_loss(pred_trans, _input[:, :3]) if self.optimize_trans and 'trans_loss' in self.components else torch.tensor(0.0).to(self.device)
        
        theta_smooth = temporal_smoothness_theta_penalty(pred_rot_mat) if self.include_theta_smoothness else torch.tensor(0.0).to(self.device)
        
        total_loss = self.weights['geo_loss'] * geo_loss + \
                     self.weights['joint_loss'] * joint_loss + \
                     self.weights['vert_loss'] * vert_loss + \
                     self.weights['beta_loss'] * beta_loss + \
                     self.weights['trans_loss'] * trans_loss + \
                     self.weights['theta_smooth'] * theta_smooth

        # Create a dictionary of loss components
        loss_components = {
            'geo_loss': self.weights['geo_loss'] * geo_loss,
            'joint_loss': self.weights['joint_loss'] * joint_loss,
            'vert_loss': self.weights['vert_loss'] * vert_loss,
            'beta_loss': self.weights['beta_loss'] * beta_loss,
            'trans_loss': self.weights['trans_loss'] * trans_loss,
            'theta_smooth': self.weights['theta_smooth'] * theta_smooth,
            'total_loss': total_loss
        }

        return total_loss, loss_components, real_verts, pred_verts, real_joints, pred_joints, _input[:, -10:], pred_betas
    
    
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0/3,
                 warmup_iters=500, warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]