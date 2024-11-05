import torch
import torch.nn as nn
from correlation import PointSpatioTemporalCorrelation
from pointnet_utils import gather, farthest_point_sample, query_ball_point_adaptive, grouping
from transformer import Transformer
from lbs import batch_rodrigues
from pytorch3d.transforms import matrix_to_rotation_6d

class PointGRU(nn.Module):
    def __init__(
            self,
            radius: float,
            nsamples: int,
            in_channels: int,
            out_channels: int,
            visu = False,
            debug_print = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.visu = visu
        self.debug_print = debug_print
        self.z_corr = PointSpatioTemporalCorrelation(radius, nsamples, in_channels, out_channels, visu, debug_print)
        self.r_corr = PointSpatioTemporalCorrelation(radius, nsamples, in_channels, out_channels, visu, debug_print)
        self.s_corr = PointSpatioTemporalCorrelation(radius, nsamples, 0, out_channels, visu, debug_print)

        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels+out_channels, out_channels, kernel_size=1, bias=True)
        )
        
        self.tanh = nn.Tanh()

    def init_state(self, inputs):
        P, _ = inputs

        inferred_batch_size = P.size(0)
        inferred_npoints = P.size(1)

        #device = P.get_device()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        P = torch.zeros([inferred_batch_size, inferred_npoints, 3], dtype=torch.float32, device=device)
        S = torch.zeros([inferred_batch_size, self.out_channels, inferred_npoints], dtype=torch.float32, device=device)

        return P, S

    def forward(self, inputs, states=None):  # Inputs: (P1, X1), States: (P2, S2)
        if states is None:
            states = self.init_state(inputs)

        P1, X1 = inputs
        P2, S2 = states

        Z = self.z_corr(P1, P2, X1, S2)
        R = self.r_corr(P1, P2, X1, S2)
        Z = self.sigmoid(Z)
        R = self.sigmoid(R)

        S_old = self.s_corr(P1, P2, None, S2)

        if self.in_channels == 0:
            S_new = R*S_old
        else:
            S_new = torch.cat(tensors=[X1, R*S_old], dim=1)

        S_new = self.fc(S_new)

        S_new = self.tanh(S_new)

        S1 = Z * S_old + (1 - Z) * S_new

        return P1, S1
    
    
class P4GRU(nn.Module):
    def __init__(self, config, output_dim, debug_print=False, visu=False):
        super().__init__()
        
        self.visu = visu
        self.output_dim = output_dim
        self.debug_print = debug_print
        self.radius, self.nsamples = config['spatial_kernel_size']
        self.spatial_stride = config['spatial_stride']
        self.emb_relu = nn.ReLU() if config['emb_relu'] else nn.Identity()
        
        # GRU layers
        self.gru_layers = nn.ModuleList()
        num_gru_layers = config['num_gru_layers']
        gru_channels = config['gru_channels']
        
        for i in range(num_gru_layers):
            in_channels = 0 if i == 0 else gru_channels[i-1]
            out_channels = gru_channels[i]
            radius_factor = i + 1
            nsamples_factor = 4 - i if 4 - i > 0 else 1
            
            self.gru_layers.append(
                PointGRU(
                    radius=radius_factor * self.radius,
                    nsamples=nsamples_factor * self.nsamples,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    visu=self.visu,
                    debug_print=self.debug_print if i == 0 else False
                )
            )
        
        self.use_pos_embedding = config['use_pos_embedding']
        if self.use_pos_embedding:
            self.pos_embedding = nn.Conv1d(in_channels=3, out_channels=gru_channels[-1], kernel_size=1, stride=1, padding=0, bias=True)
        
        self.use_transformer = config['use_transformer']
        if self.use_transformer:
            transformer_config = config['transformer']
            self.transformer = Transformer(
                dim=transformer_config['dim'],
                depth=transformer_config['depth'],
                heads=transformer_config['heads'],
                dim_head=transformer_config['dim_head'],
                mlp_dim=transformer_config['mlp_dim']
            )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(gru_channels[-1]),
            nn.Linear(gru_channels[-1], transformer_config['mlp_dim']),
            #nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(transformer_config['mlp_dim'], output_dim),
        )
         
    def forward(self, points: torch.Tensor):
        if self.debug_print:
            print(f"P4DConv forward pass started. Input xyzs shape: {points.shape}")
            
        xyzs = points[:, :, :, :3]  # [B, T, N, 3]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #device = xyzs.get_device()
        B = xyzs.size(0)
        T = xyzs.size(1)
        N = xyzs.size(2)
        
        if self.debug_print:
            print(f"Processing {T} frames with {N} points each")
            
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs] # List of length T of (B, N, 3) tensors
        
        features = []
        pos = []
        
        states = [None] * len(self.gru_layers)
        
        for t in range(T):
            if self.debug_print:
                print(f"\nProcessing temporal frame {t}")
            
            xyz = xyzs[t]
            for i, gru_layer in enumerate(self.gru_layers):
                if self.debug_print:
                    print(f"\n=== GRU{i+1} ===")
                
                if i == 0:
                    xyz_idx = farthest_point_sample(xyz, N // self.spatial_stride)
                    xyz = gather(xyz.transpose(1, 2).contiguous(), xyz_idx).transpose(1, 2).contiguous()
                    
                    if t == 0:
                        S_init = torch.zeros([B, gru_layer.out_channels, N // self.spatial_stride], dtype=torch.float32, device=device)
                        states[i] = gru_layer((xyz, None), (xyz, S_init))
                    else:
                        states[i] = gru_layer((xyz, None), states[i])
                    
                    s_xyz, s_feat = states[i]
                else:
                    prev_xyz, prev_feat = states[i-1]
                    xyz_idx = farthest_point_sample(prev_xyz, prev_xyz.size(1) // self.spatial_stride)
                    xyz = gather(prev_xyz.transpose(1, 2).contiguous(), xyz_idx).transpose(1, 2).contiguous()
                    
                    idx, r = query_ball_point_adaptive(self.radius * (i+1) / 2, self.nsamples, prev_xyz, xyz)
                    feat = grouping(prev_feat, idx)
                    feat = torch.max(input=feat, dim=-1, keepdim=False)[0]
                    
                    states[i] = gru_layer((xyz, feat), states[i])
                    s_xyz, s_feat = states[i]
                
                if self.debug_print:
                    print(f"Shape of xyz: {s_xyz.shape}")
                    print(f"Shape of features: {s_feat.shape}")
            
            features.append(s_feat) 
            pos.append(s_xyz)
        
        features = torch.stack(features, dim=1) # [B, T, C, N]
        features = torch.reshape(features, (B, T * features.size(3), features.size(2))) # [B, T * N, C]
        
        pos = torch.stack(pos, dim=1) # [B, T, N, 3]
        pos = torch.reshape(pos, (B, T * pos.size(2), pos.size(3))) # [B, T * N, 3]
        
        if self.use_pos_embedding:
            pos_embedding = self.pos_embedding(pos.permute(0, 2, 1)).permute(0, 2, 1) # [B, T * N, C]
            embedding = pos_embedding + features
        else:
            embedding = features
        
        if self.use_transformer:
            embedding = self.transformer(embedding)
        
        embedding = self.emb_relu(embedding) # [B, T * N, C]
        
        embedding = torch.max(input=embedding, dim=1, keepdim=False)[0] # [B, C]
        output = self.mlp_head(embedding) # [B, output_dim]
        output = output.view(T, -1) # [T, output_dim]
        
        return output
    
