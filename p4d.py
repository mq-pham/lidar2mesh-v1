import torch
from torch import nn

try : 
    from point4dconv import P4DConv
    from transformer import Transformer
except :
    from .point4dconv import P4DConv
    from .transformer import Transformer

class P4D(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride, # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,  # P4DConv: temporal
                 emb_relu,                               # embedding: relu
                 dim, depth, heads, dim_head,            # transformer
                 mlp_dim, output_dim):                  # output
        super().__init__()
        
        self.p4dconv = P4DConv(in_planes=0, mlp_planes=[128, 256, dim], mlp_batch_norm=[True, True, True], mlp_activation=[True, True, True],
                                spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                temporal_padding_mode='replicate', operator='+', spatial_pooling='max', temporal_pooling='max', bias=True, debug_print=False)
        
        
        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)
        self.emb_relu = nn.ReLU() if emb_relu else False
        self.mlp_head = nn.Sequential(
            #nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            #nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )
        
        
   
    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None):
        
        #device = xyzs.device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # P4DConv forward pass
        xyzs, features = self.p4dconv(xyzs, features) # [B, T, n, 3], [B, T, dim, n]
        
        B = xyzs.shape[0]
        T = xyzs.shape[1]
        
        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]
        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)
        #output = embedding # [B, L*n, dim]
        output = torch.mean(input=output, dim=1, keepdim=False, out=None)[0] # [B, dim]
        output = self.mlp_head(output)  # [B, output_dim] => [1, 22*6*T] => [1, 132]
        output = output.view(T, -1) # [T, 132]
    
        return output
    