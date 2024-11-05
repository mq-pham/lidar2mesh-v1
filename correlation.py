import torch
import torch.nn as nn
from pointnet_utils import query_ball_point_adaptive, grouping
import matplotlib.pyplot as plt
from utils import plot_sphere

class PointSpatioTemporalCorrelation(nn.Module):
    def __init__(
            self,
            radius: float,
            nsamples: int,
            in_channels: int,
            out_channels: int,
            visu = False,
            debug_print = False,
    ):
        super().__init__()

        self.radius = radius
        self.nsamples = nsamples
        self.in_channels = in_channels
        self.visu = visu
        self.debug_print = debug_print

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels+out_channels+3, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, P1: torch.Tensor, P2: torch.Tensor, X1: torch.Tensor, S2: torch.Tensor) -> (torch.Tensor):
        r"""
        Parameters
        ----------
        P1:     (B, N, 3)
        P2:     (B, N, 3)
        X1:     (B, C, N)
        S2:     (B, C, N)

        Returns
        -------
        S1:     (B, C, N)
        """
        if self.debug_print:
            print(f"Input shapes:")
            print(f"P1: {P1.shape}, P2: {P2.shape}, X1: {X1}, S2: {S2.shape}")
            
        # 1. Sample points
        idx, r = query_ball_point_adaptive(self.radius, self.nsamples, P2, P1)    # [B, N, k]
        
        if self.debug_print:
            print(f"\nAfter query_ball_point_adaptive:")
            print(f"idx shape: {idx.shape}, r shape: {r.shape}")
            print(f"r min: {r.min().item():.4f}, r max: {r.max().item():.4f}")

        # 2.1 Group P2 points
        P2_flipped = P2.transpose(1, 2).contiguous()    # (B, 3, N)
        P2_grouped = grouping(P2_flipped, idx)    # (B, 3, N, k)
        
        if self.debug_print:
            print(f"\nAfter grouping P2:")
            print(f"P2_grouped shape: {P2_grouped.shape}")
            
        # 2.2 Group P2 states
        S2_grouped = grouping(S2, idx)  # (B, C, N, k)
        
        if self.debug_print:
            print(f"\nAfter grouping S2:")
            print(f"S2_grouped shape: {S2_grouped.shape}")
            print(f"S2_grouped min: {S2_grouped.min().item():.4f}, max: {S2_grouped.max().item():.4f}")

        # 3. Calcaulate displacements
        P1_flipped = P1.transpose(1, 2).contiguous()    # (B, 3, N)
        P1_expanded = torch.unsqueeze(P1_flipped, 3)    # (B, 3, N, 1)
        displacement = P2_grouped - P1_expanded # (B, 3, N, k) -> Difference between each nsamples from P2 in query balls versus its centroid computed from P1
        
        if self.debug_print:
            print(f"\nAfter calculating displacements:")
            print(f"displacement shape: {displacement.shape}")
            print(f"displacement min: {displacement.min().item():.4f}, max: {displacement.max().item():.4f}")

        
        if self.in_channels != 0:
            X1_expanded = torch.unsqueeze(X1, 3)    # (B, C, N, 1)
            X1_repeated = X1_expanded.repeat(1, 1, 1, self.nsamples) # [B, C, N, k]
            correlation = torch.cat(tensors=(S2_grouped, X1_repeated, displacement), dim=1)  # (B, 2*C + 3, N, k)
        else:
            correlation = torch.cat(tensors=(S2_grouped, displacement), dim=1)   # (B, C + 3, N, k)
        
        if self.debug_print:
            print(f"\nAfter creating correlation tensor:")
            print(f"correlation shape: {correlation.shape}")
            print(f"correlation min: {correlation.min().item():.4f}, max: {correlation.max().item():.4f}")
            

        # 5. Fully-connected layer (the only parameters)
        S1 = self.fc(correlation)
        
        if self.debug_print:
            print(f"\nAfter fully-connected layer:")
            print(f"S1 shape: {S1.shape}")
            print(f"S1 min: {S1.min().item():.4f}, max: {S1.max().item():.4f}")

        # 6. Pooling
        S1 = torch.max(input=S1, dim=-1, keepdim=False)[0] # (B, out_channels, N)
        
        if self.debug_print:
            print(f"\nAfter pooling:")
            print(f"S1 shape: {S1.shape}")
            print(f"S1 min: {S1.min().item():.4f}, max: {S1.max().item():.4f}")
        
        if self.visu:
            P1_np = P1.cpu().numpy()[0] # (N, 3)
            P2_np = P2.cpu().numpy()[0] # (N, 3)
            P2_grouped_np = P2_grouped.permute(0, 2, 3, 1).cpu().numpy()[0] # (N, k, 3)
            S2_np = S2.detach().cpu().numpy()[0] # (C, N)
            S1_np = S1.detach().cpu().numpy()[0] # (C, N)
            r = r.cpu().numpy()[0] # (N, k)
            
            fig = plt.figure(figsize=(10, 10))
            ax1 = fig.add_subplot(111, projection='3d')
            
            ax1.scatter(P1_np[:, 0], P1_np[:, 1] - 1.0, P1_np[:, 2], color="red", s=0.1)
            ax1.scatter(P2_np[:, 0], P2_np[:, 1], P2_np[:, 2], color="blue", s=0.1)
            
            for i in range(min(10,P1_np.shape[0])): # How much sampled FPS centroid to plot
                sampled_center = P1_np[i]
                
                ball = P2_grouped_np[i]
                ax1.scatter(ball[:, 0], ball[:, 1], ball[:, 2], color="yellow", s=50, edgecolors='black')
                ax1.scatter(sampled_center[0], sampled_center[1], sampled_center[2], color="black", s=50, edgecolors='black')
                #ax.plot([sampled_center[0], ball[:, 0]], [sampled_center[1], ball[:, 1]], [sampled_center[2], ball[:, 2]], color='gray', alpha=0.5)
                plot_sphere(ax1, sampled_center[0], sampled_center[1], sampled_center[2], r[i], color='green', alpha=0.1)
                
                # Draw lines from sampled center to each point in the ball
                for j in range(P2_grouped_np.shape[1]):
                    ax1.plot([sampled_center[0], ball[j, 0]], [sampled_center[1], ball[j, 1]], [sampled_center[2], ball[j, 2]], color='gray', alpha=0.5)
                
            
            ax1.set_axis_off()
            ax1.set_xlim(-1, 1)
            ax1.set_ylim(-1, 1)
            ax1.set_zlim(-1, 1)
            ax1.view_init(elev=45, azim=-30)
            ax1.set_title("P4DConv Visualization: Previous, Anchor, and Next Frames")
            
            
                
            plt.tight_layout()
            plt.show()
            
            # Print some statistical information
            print("S2 (Input States) Statistics:")
            print(f"Mean: {S2_np.mean():.4f}")
            print(f"Std Dev: {S2_np.std():.4f}")
            print(f"Min: {S2_np.min():.4f}")
            print(f"Max: {S2_np.max():.4f}")
            print("\nS1 (Output States) Statistics:")
            print(f"Mean: {S1_np.mean():.4f}")
            print(f"Std Dev: {S1_np.std():.4f}")
            print(f"Min: {S1_np.min():.4f}")
            print(f"Max: {S1_np.max():.4f}")

        return S1 # (B, out_channels, N)
    

if __name__ == '__main__':
    # Test the module
    
    from P4D.smplx_model import SMPLX
    from lidar import LiDAR
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    smplx = SMPLX().to(device)
    lidar = LiDAR(
        elev=0.0,
        dist=6.0,
        image_size=(256, 192),
        min_depth=0.0,
        max_depth=1.0,
        device=device,
        return_normals=False,
        subsample=1024
    )
    
    p1 = torch.tensor([
            1.22162998e+00,   1.17162502e+00,   1.16706634e+00,
            -1.20581151e-03,   8.60930011e-02,   4.45963144e-02,
            -1.52801601e-02,  -1.16911056e-02,  -6.02894090e-03,
            1.62427306e-01,   4.26302850e-02,  -1.55304456e-02,
            2.58729942e-02,  -2.15941742e-01,  -6.59851432e-02,
            7.79098943e-02,   1.96353287e-01,   6.44420758e-02,
            -5.43042570e-02,  -3.45508829e-02,   1.13200583e-02,
            -5.60734887e-04,   3.21716577e-01,  -2.18840033e-01,
            -7.61821344e-02,  -3.64610642e-01,   2.97633410e-01,
            9.65453908e-02,  -5.54007106e-03,   2.83410680e-02,
            -9.57194716e-02,   9.02515948e-02,   3.31488043e-01,
            -1.18847653e-01,   2.96623230e-01,  -4.76809204e-01,
            -1.53382001e-02,   1.72342166e-01,  -1.44332021e-01,
            -8.10869411e-02,   4.68325168e-02,   1.42248288e-01,
            -4.60898802e-02,  -4.05981280e-02,   5.28727695e-02,
            3.20133418e-02,  -5.23784310e-02,   2.41559884e-03,
            -3.08033824e-01,   2.31431410e-01,   1.62540793e-01,
            6.28208935e-01,  -1.94355965e-01,   7.23800480e-01,
            -6.49612308e-01,  -4.07179184e-02,  -1.46422181e-02,
            4.51475441e-01,   1.59122205e+00,   2.70355493e-01,
            2.04248756e-01,  -6.33800551e-02,  -5.50178960e-02,
            -1.00920045e+00,   2.39532292e-01,   3.62904727e-01]).to(device)
    
    p2 = torch.tensor([
            1.22162998e+00,   1.17162502e+00,   1.16706634e+00,
            -1.20581151e-03,   8.60930011e-02,   4.45963144e-02,
            -1.52801601e-02,  -1.16911056e-02,  -6.02894090e-03,
            1.62427306e-01,   5.0,  -2.0,
            2.0,  -2.0,  -2.0,
            7.79098943e-02,   1.96353287e-01,   6.44420758e-02,
            -5.43042570e-02,  -3.45508829e-02,   1.13200583e-02,
            -5.60734887e-04,   3.21716577e-01,  -2.18840033e-01,
            -7.61821344e-02,  -3.64610642e-01,   2.97633410e-01,
            9.65453908e-02,  -5.54007106e-03,   2.83410680e-02,
            -9.57194716e-02,   9.02515948e-02,   3.31488043e-01,
            -1.18847653e-01,   2.96623230e-01,  -4.76809204e-01,
            -1.53382001e-02,   1.72342166e-01,  -1.44332021e-01,
            -8.10869411e-02,   4.68325168e-02,   1.42248288e-01,
            -4.60898802e-02,  -4.05981280e-02,   5.28727695e-02,
            3.20133418e-02,  -5.23784310e-02,   2.41559884e-03,
            -3.08033824e-01,   2.31431410e-01,   1.62540793e-01,
            6.28208935e-01,  -1.94355965e-01,   7.23800480e-01,
            -6.49612308e-01,  -4.07179184e-02,  -1.46422181e-02,
            4.51475441e-01,   1.59122205e+00,   2.70355493e-01,
            2.04248756e-01,  -6.33800551e-02,  -5.50178960e-02,
            -1.00920045e+00,   2.39532292e-01,   3.62904727e-01]).to(device)
    
    _input = torch.cat([p1.unsqueeze(0), p2.unsqueeze(0)], dim=0) # (2, 66)
    betas = torch.zeros(1, 16).repeat(2, 1).to(device)
    target = smplx(pose_body=_input, betas=betas, use_rodrigues=True)  
    pc = lidar(target['meshes'], torch.tensor([0.0, 0.5]).to(device))

    B, N, C = 1, 1024, 64
    P1 = pc.points_padded()[0, :, :3].unsqueeze(0)
    P2 = pc.points_padded()[1, :, :3].unsqueeze(0)
    X1 = None
    S2 = torch.zeros(B, C, N).to(device)
    
    model = PointSpatioTemporalCorrelation(radius=0.1, nsamples=16, in_channels=0, out_channels=64, visu=True, debug_print=True)
    S1 = model(P1, P2, X1, S2)
    print(f"S1 shape: {S1.shape}")
    
