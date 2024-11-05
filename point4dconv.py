import torch
from torch import nn
import numpy as np
import random

try :
    from .pointnet_utils import *
except:
    from pointnet_utils import *


class P4DConv(nn.Module):
    def __init__(self, in_planes, mlp_planes, mlp_batch_norm, mlp_activation,
                 spatial_kernel_size, spatial_stride, temporal_kernel_size,
                 temporal_stride=1, temporal_padding=[0, 0],
                 temporal_padding_mode='replicate', operator='*',
                 spatial_pooling='max', temporal_pooling='sum', bias=False,
                 debug_print=False, visu=False, seed=None):
        super().__init__()
        
        self.seed = seed
        if self.seed is not None:
            self.set_seed(self.seed)
        self.visu = visu
        self.debug_print = debug_print
        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_activation = mlp_activation

        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.temporal_padding_mode = temporal_padding_mode

        self.operator = operator
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling

        conv_d = [nn.Conv2d(in_channels=4, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
        if mlp_batch_norm[0]:
            conv_d.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
        if mlp_activation[0]:
            conv_d.append(nn.ReLU(inplace=True))
        self.conv_d = nn.Sequential(*conv_d)

        if in_planes != 0:
            conv_f = [nn.Conv2d(in_channels=in_planes, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
            if mlp_batch_norm[0]:
                conv_f.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
            if mlp_activation[0]:
                conv_f.append(nn.ReLU(inplace=True))
            self.conv_f = nn.Sequential(*conv_f)

        mlp = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=mlp_planes[i]))
            if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)
        
        if self.debug_print:
            print(f"P4DConv initialized with: in_planes={in_planes}, "
                  f"mlp_planes={mlp_planes}, spatial_kernel_size={spatial_kernel_size}, "
                  f"temporal_kernel_size={temporal_kernel_size}")
            
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, points: torch.Tensor, features: torch.Tensor = None):
        
        if self.seed is not None:
            self.set_seed(self.seed)
            
        if self.debug_print:
            print(f"P4DConv forward pass started. Input xyzs shape: {xyzs.shape}, "
                  f"features shape: {features.shape if features is not None else None}")
            
        xyzs = points[:, :, :, :3]  # [B, T, N, 3]
        rgbs = points[:, :, :, 3:]  # [B, T, N, 3]
    

        device = xyzs.get_device()
        nframes = xyzs.size(1)
        npoints = xyzs.size(2)

        if self.debug_print:
            print(f"Processing {nframes} frames with {npoints} points each")

        assert (self.temporal_kernel_size % 2 == 1), "P4DConv: Temporal kernel size should be odd!"
        assert ((nframes + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride == 0), "P4DConv: Temporal length error!"

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs] # List of length T of (B, N, 3) tensors
        
        rgbs = torch.split(tensor=rgbs, split_size_or_sections=1, dim=1)
        rgbs = [torch.squeeze(input=rgb, dim=1).contiguous() for rgb in rgbs] # List of length T of (B, N, 3) tensors

        if self.temporal_padding_mode == 'zeros':
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            rgb_padding = torch.zeros(rgbs[0].size(), dtype=torch.float32, device=device)   
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
                rgbs = [rgb_padding] + rgbs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]
                rgbs = rgbs + [rgb_padding]
        else:
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
                rgbs = [rgbs[0]] + rgbs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]
                rgbs = rgbs + [rgbs[-1]]
                
        # xyzs is a list of length T+sum(temporal_padding) of (B, N, 3) tensors

        if self.in_planes != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

            if self.temporal_padding_mode == 'zeros':
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
            else:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]

        if self.debug_print:
            print(f"After temporal padding: {len(xyzs)} frames")

        new_xyzs = []
        new_features = []
        for t in range(self.temporal_kernel_size//2, len(xyzs)-self.temporal_kernel_size//2, self.temporal_stride):
            if self.debug_print:
                print(f"Processing temporal anchor frame {t}")

            anchor_idx = farthest_point_sample(xyzs[t], npoints//self.spatial_stride) # [B, npoints//spatial_stride]
            anchor_xyz_flipped = gather(xyzs[t].transpose(1, 2).contiguous(), anchor_idx) # [B, 3, npoints//spatial_stride]
            anchor_xyz_expanded = anchor_xyz_flipped.unsqueeze(3) # [B, 3, npoints//spatial_stride, 1]
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous() # [B, npoints//spatial_stride, 3]

            if self.debug_print:
                print(f"Anchor points sampled. Shape: {anchor_xyz.shape}")

            new_feature = []
            query_balls = []
            query_balls_rgb = []
            for i in range(t-self.temporal_kernel_size//2, t+self.temporal_kernel_size//2+1):
                if self.debug_print:
                    print(f"  Processing neighbor frame {i}")

                neighbor_xyz = xyzs[i] # [B, N, 3] 
                neighbor_rgb = rgbs[i] # [B, N, 3]
                
                # TODO : Change query ball to only select balls with nb of neighbors > 0 so that 
                # we can avoid to iteratively augment the radius size
                idx, radius = query_ball_point_adaptive(self.r, self.k, neighbor_xyz, anchor_xyz) # [B, npoints//spatial_stride, k]
                
                
                if self.debug_print:
                    print(f"  Shape of idx: {idx.shape}")
                    print(f"  Shape of neighbor_xyz: {neighbor_xyz.transpose(1, 2).contiguous().shape}")
                    
                neighbor_xyz_grouped = grouping(neighbor_xyz.transpose(1, 2).contiguous(), idx) # [B, 3, npoints//spatial_stride, k]
                neighbor_rgb_grouped = grouping(neighbor_rgb.transpose(1, 2).contiguous(), idx) # [B, 3, npoints//spatial_stride, k]
                
                if self.debug_print:
                    print(f"  Shape of neighbor_xyz_grouped: {neighbor_xyz_grouped.shape}")
                    
                if self.visu:
                    query_balls.append(neighbor_xyz_grouped.permute(0, 2, 3, 1).cpu().numpy()[0]) # List of 3 elements of shape (npoints//spatial_stride, k, 3)
                    query_balls_rgb.append(neighbor_rgb_grouped.permute(0, 2, 3, 1).cpu().numpy()[0]) # List of 3 elements of shape (npoints//spatial_stride, k, 3)
                    
                xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded # [B, 3, npoints//spatial_stride, k]
                #xyz_displacement = xyz_displacement.permute(0, 3, 1, 2)
                t_displacement = torch.ones((xyz_displacement.size()[0], 1, xyz_displacement.size()[2], xyz_displacement.size()[3]), dtype=torch.float32, device=device) * (i-t)
                displacement = torch.cat(tensors=(xyz_displacement, t_displacement), dim=1, out=None)

                if self.debug_print:
                    print(f"  Displacement tensor shape: {displacement.shape}")

                displacement = self.conv_d(displacement)
                
                if self.debug_print:
                    print(f"  Displacement tensor shape after conv_d: {displacement.shape}")

                if self.in_planes != 0:
                    neighbor_feature_grouped = grouping(features[i], idx)
                    feature = self.conv_f(neighbor_feature_grouped)
                    if self.operator == '+':
                        feature = feature + displacement
                    else:
                        feature = feature * displacement
                else:
                    feature = displacement

                if self.debug_print:
                    print(f"  Feature tensor shape after conv_f: {feature.shape}")

                feature = self.mlp(feature)
                
                if self.debug_print:
                    print(f"  Feature tensor shape after mlp: {feature.shape}")
                
                if self.spatial_pooling == 'max':
                    feature = torch.max(input=feature, dim=-1, keepdim=False)[0]
                elif self.spatial_pooling == 'sum':
                    feature = torch.sum(input=feature, dim=-1, keepdim=False)
                else:
                    feature = torch.mean(input=feature, dim=-1, keepdim=False)

                if self.debug_print:
                    print(f"  Feature tensor shape after spatial pooling: {feature.shape}")

                new_feature.append(feature)
                
            if self.visu:
                prev_frame = xyzs[t-1].cpu().numpy()[0]
                prev_rgb = rgbs[t-1].cpu().numpy()[0]
                anchor_frame = xyzs[t].cpu().numpy()[0]
                anchor_rgb = rgbs[t].cpu().numpy()[0]
                next_frame = xyzs[t+1].cpu().numpy()[0]
                next_rgb = rgbs[t+1].cpu().numpy()[0]
                anchor_xyz_np = anchor_xyz.cpu().numpy()[0]
                # visualize_frames(radius.cpu().numpy(), 
                #                  prev_frame, 
                #                  prev_rgb, 
                #                  anchor_frame, 
                #                  anchor_rgb, 
                #                  next_frame, 
                #                  next_rgb,
                #                  anchor_xyz_np, 
                #                  query_balls,
                #                  query_balls_rgb)
                

            new_feature = torch.stack(tensors=new_feature, dim=1)
            if self.temporal_pooling == 'max':
                new_feature = torch.max(input=new_feature, dim=1, keepdim=False)[0]
            elif self.temporal_pooling == 'sum':
                new_feature = torch.sum(input=new_feature, dim=1, keepdim=False)
            else:
                new_feature = torch.mean(input=new_feature, dim=1, keepdim=False)

            if self.debug_print:
                print(f"Feature tensor shape after temporal pooling: {new_feature.shape}")

            new_xyzs.append(anchor_xyz)
            new_features.append(new_feature)

        new_xyzs = torch.stack(tensors=new_xyzs, dim=1)
        new_features = torch.stack(tensors=new_features, dim=1)

        if self.debug_print:
            print(f"P4DConv forward pass completed. Output shapes: "
                  f"new_xyzs: {new_xyzs.shape}, new_features: {new_features.shape}")

        return new_xyzs, new_features

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    B, T, N, C = 1, 4, 1024, 3
    xyzs = torch.randn(B, T, N, 3).to(device)
    features = torch.randn(B, T, C, N).to(device)
    
    model = P4DConv(in_planes=C, mlp_planes=[64, 32], mlp_batch_norm=[False, False], mlp_activation=[False, False],
                        spatial_kernel_size=[0.5, 128], spatial_stride=1,
                        temporal_kernel_size=3, temporal_stride=2, temporal_padding=[1, 1],
                        temporal_padding_mode='replicate', operator='+', spatial_pooling='max',
                        temporal_pooling='max', bias=False)
    
    model.to(device)
    output = model(xyzs, features)
    
    print("Final output tensor shapes:")
    print(f"new_xyzs: {output[0].shape}")
    print(f"new_features: {output[1].shape}")
    
    