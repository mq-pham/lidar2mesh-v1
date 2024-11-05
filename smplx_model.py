import torch

from body_model import BodyModel
from lbs import lbs, batch_rodrigues

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import os


# Define the path to SMPLX_MALE.npz using an absolute path
SMPLX_MODEL_MALE_PATH = "./smplx/SMPLX_NEUTRAL.npz"



class SMPLX(BodyModel):
    def __init__(self, num_betas=16, **kwargs):
        super().__init__(bm_fname=SMPLX_MODEL_MALE_PATH, num_betas=num_betas, num_expressions=0, **kwargs)

    def forward(self, pose_body, betas, use_rodrigues=True, opti_trans=True):
        
        device = pose_body.device
        for name in ['init_pose_hand', 'init_pose_jaw','init_pose_eye', 'init_v_template', 'init_expression', 
                    'shapedirs', 'exprdirs', 'posedirs', 'J_regressor', 'kintree_table', 'weights', 'f']:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(device))

        batch_size = pose_body.shape[0]
        
        if opti_trans:
            trans = pose_body[:, :3]
        else:
            trans = torch.zeros(batch_size, 3, device=device)
        pose_hand = self.init_pose_hand.expand(batch_size, -1)
        pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
        pose_eye = self.init_pose_eye.expand(batch_size, -1)
        v_template = self.init_v_template.expand(batch_size, -1, -1)
        expression = self.init_expression.expand(batch_size, -1)

        init_pose = torch.cat([pose_jaw, pose_eye, pose_hand], dim=-1) # [B, 99] full of zeros
        if not use_rodrigues:
            init_pose = batch_rodrigues(init_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
        if opti_trans:
            full_pose = torch.cat([pose_body[:, 3:], init_pose], dim=1)
        else:
            full_pose = torch.cat([pose_body, init_pose], dim=1) # No translation, concat of Pose body [B, 66] and init_pose [B, 99] = [B, 165]
        
        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)
        

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
