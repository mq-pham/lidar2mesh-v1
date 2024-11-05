import torch

from body_model import BodyModel
#from utils import rodrigues_2_rot_mat, rotation6d_2_rot_mat
from utils import rodrigues_2_rot_mat
from lbs import lbs, batch_rodrigues

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import os

 

# Define the path to SMPLX_MALE.npz using an absolute path
SMPL_MODEL_MALE_PATH = "./smpl/SMPL_NEUTRAL.pkl"

class SMPL(BodyModel):
    def __init__(self, num_betas=10, **kwargs):
        super().__init__(bm_fname=SMPL_MODEL_MALE_PATH, num_betas=num_betas, num_expressions=0, **kwargs)

    def forward(self, pose_body, betas, use_rodrigues=True, opti_trans=False):
        
        device = pose_body.device
        for name in ['init_pose_hand', 'init_v_template', 'shapedirs', 'posedirs', 'J_regressor', 'kintree_table', 'weights', 'f']:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(device))

        batch_size = pose_body.shape[0]
        
        if opti_trans:
            trans = pose_body[:, :3]
        else:
            trans = torch.zeros(batch_size, 3, device=device)
        pose_hand = self.init_pose_hand.expand(batch_size, -1)
        v_template = self.init_v_template.expand(batch_size, -1, -1)
        init_pose = pose_hand   
        if not use_rodrigues:
            init_pose = batch_rodrigues(init_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
        if opti_trans:
            full_pose = torch.cat([pose_body[:, 3:], init_pose], dim=1)
        else:
            full_pose = torch.cat([pose_body, init_pose], dim=1)
        
        shape_components = betas
        shapedirs = self.shapedirs
        

        verts, joints = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
                        shapedirs=shapedirs, posedirs=self.posedirs, J_regressor=self.J_regressor,
                        parents=self.kintree_table[0].long(), lbs_weights=self.weights, pose2rot=use_rodrigues)
        
        faces = self.f.expand(batch_size, -1, -1)
        textures = torch.ones_like(verts)
        
        meshes = Meshes(
            verts=verts,
            faces=faces,
            textures=TexturesVertex(verts_features=textures)
        )
        
        joints = joints + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)

        
        return dict(meshes=meshes, verts=verts, trans=trans, joints=joints)