import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch3d.structures import Pointclouds
import gc
import matplotlib.pyplot as plt

def visualize_results(real_verts, pred_verts, pcs, loss_components):
    with torch.no_grad():
        print("Real verts shape: ", real_verts.shape)
        print("Pred verts shape: ", pred_verts.shape)
        print("PCs shape: ", pcs.shape)
        
        for key, value in loss_components.items():
            print(f"{key}: {value.item()}")
        
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        
        pred_verts = pred_verts[0].detach().cpu().numpy()
        real_verts = real_verts[0].detach().cpu().numpy()
        pcs = pcs.squeeze(0)[0].detach().cpu().numpy()
        
        ax1.scatter(real_verts[:, 0], real_verts[:, 1], real_verts[:, 2], c='r', s=0.05)
        ax2.scatter(pred_verts[:, 0], pred_verts[:, 1], pred_verts[:, 2], c='b', s=0.05)
        ax3.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c='black', s=3)
        
        for ax, data in zip([ax1, ax2, ax3], [real_verts, pred_verts, pcs]):
            center = data.mean(axis=0)
            radius = np.max(np.linalg.norm(data - center, axis=1))
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show()

def optimize_memory():
    # Clear PyTorch CUDA cache
    torch.cuda.empty_cache()
    
    # Garbage collect Python objects
    gc.collect()

optimize_memory()


def temporal_smoothness_theta_penalty(theta):
    """
    Computes the temporal smoothness penalty for theta across frames.
    Args:
    theta (torch.Tensor): Tensor of shape (num_frames, -1) representing the theta values for each frame.
    Returns:
    torch.Tensor: Scalar tensor representing the temporal smoothness penalty for theta.
    """
    device = theta.device
    smoothness_loss = torch.zeros(1, device=device)
    num_frames = theta.shape[0]

    # Compute the first-order difference penalty
    for i in range(1, num_frames):
        smoothness_loss += torch.norm(theta[i] - theta[i-1], p=2) ** 2
    smoothness_loss /= (num_frames - 1)

    # Compute the second-order difference penalty
    for i in range(1, num_frames - 1):
        smoothness_loss += torch.norm(theta[i+1] - 2 * theta[i] + theta[i-1], p=2) ** 2
    smoothness_loss /= (num_frames - 2)

    return smoothness_loss



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def plot_sphere(ax, x, y, z, r, color='r', alpha=0.1):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = r * np.cos(u) * np.sin(v) + x
    ys = r * np.sin(u) * np.sin(v) + y
    zs = r * np.cos(v) + z
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha)
    

def load_data(device, smplx, lidar, dataset="CNRS/283/-01_L_1_stageii.npz", step=1, opti_trans=True):
    
    # Load NPZ file
    npz_file_path = f"../data/{dataset}"
    data = np.load(npz_file_path, allow_pickle=True)
    
    # Extract relevant data
    pose_body = torch.tensor(data['pose_body'], dtype=torch.float32).to(device)
    root_orient = torch.tensor(data['root_orient'], dtype=torch.float32).to(device)
    trans = torch.tensor(data['trans'], dtype=torch.float32).to(device)
    betas = torch.tensor(data['betas'], dtype=torch.float32).to(device)
    
    num_total_frames = pose_body.shape[0]
    frame_indices = range(0, num_total_frames, step)
    num_frames = len(frame_indices)

    azimuth_step = 360 / (num_total_frames - 1) if num_total_frames > 1 else 0
    azimuths = torch.tensor([i * azimuth_step for i in frame_indices], device=device)

    if opti_trans:
        pose_input = torch.cat([trans[frame_indices], root_orient[frame_indices], pose_body[frame_indices]], dim=-1) # (T, 69) (trans: 3, root_orient: 3, pose_body: 63)
    else:
        pose_input = torch.cat([root_orient[frame_indices], pose_body[frame_indices]], dim=-1)
    beta_input = betas.unsqueeze(0).repeat(num_frames, 1) # (T, 16)
    
    _input = torch.cat([pose_input, beta_input], dim=-1) # (T, 69 + 16) = (T, 85)
    
    output = smplx(pose_body=_input[:, :-16], betas=_input[:, -16:], use_rodrigues=True, opti_trans=opti_trans) # Input : (T, 1*3 + 1*3 + 21*3) = (T, 69), (T, 16) Output : Meshes, Verts = (T, 10475, 3), Joints = (T, 55, 3)
    meshes = output['meshes']
    verts = output['verts']
    joints = output['joints']
    pc = lidar(meshes, azimuths) # Input : Meshes, Azimuths = (T, 1) Output : Pointclouds = (T, 1024, 3)
    
    # Apply trans
    if opti_trans:
        pc_offset = pc.points_padded() + output['trans'].unsqueeze(dim=1)
        pc = Pointclouds(points=pc_offset, features=None)

    return _input, verts, joints, pc, azimuths, dataset

def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def rot_mat_2_rodrigues(rotation_matrix):
    batch_dim = rotation_matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        rotation_matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    
    out_standardized = torch.where(out[..., 0:1] < 0, -out, out)
    
    norms = torch.norm(out_standardized[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, out_standardized[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return out_standardized[..., 1:] / sin_half_angles_over_angles


# def rotation6d_2_rot_mat(rotation6d):
#     batch_size = rotation6d.shape[0]
#     pose6d = rotation6d.reshape(-1, 6)
    
#     a1, a2 = pose6d[..., :3], pose6d[..., 3:]
#     b1 = F.normalize(a1, dim=-1)
#     b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
#     b2 = F.normalize(b2, dim=-1)
#     b3 = torch.cross(b1, b2, dim=-1)
#     output = torch.stack((b1, b2, b3), dim=-2)
    
#     return output.reshape(batch_size, -1)

def rot6d_to_rotmat(x):
    x = x.view(-1,3,2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats

def rotation_6d_to_matrix(x):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    # a1, a2 = d6[..., :3], d6[..., 3:]
    # b1 = F.normalize(a1, dim=-1)
    # b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    # b2 = F.normalize(b2, dim=-1, eps=1e-8)
    # b3 = torch.cross(b1, b2, dim=-1)
    # return torch.stack((b1, b2, b3), dim=-2)

    x = x.view(-1,3,2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats



def rodrigues_2_rot_mat(rvecs):
    # batch_size = rvecs.shape[0]
    # r_vecs = rvecs.reshape(-1, 3)
    # total_size = r_vecs.shape[0]
    # thetas = torch.norm(r_vecs, dim=1, keepdim=True)
    # is_zero = torch.eq(torch.squeeze(thetas), torch.tensor(0.0))
    # u = r_vecs / thetas

    # # Each K is the cross product matrix of unit axis vectors
    # # pyformat: disable
    # zero = torch.autograd.Variable(torch.zeros([total_size], device="cuda"))  # for broadcasting
    # Ks_1 = torch.stack([  zero   , -u[:, 2],  u[:, 1] ], axis=1)  # row 1
    # Ks_2 = torch.stack([  u[:, 2],  zero   , -u[:, 0] ], axis=1)  # row 2
    # Ks_3 = torch.stack([ -u[:, 1],  u[:, 0],  zero    ], axis=1)  # row 3
    # # pyformat: enable
    # Ks = torch.stack([Ks_1, Ks_2, Ks_3], axis=1)                  # stack rows

    # identity_mat = torch.autograd.Variable(torch.eye(3, device="cuda").repeat(total_size,1,1))
    # Rs = identity_mat + torch.sin(thetas).unsqueeze(-1) * Ks + \
    #      (1 - torch.cos(thetas).unsqueeze(-1)) * torch.matmul(Ks, Ks)
    # # Avoid returning NaNs where division by zero happened
    # R = torch.where(is_zero[:,None,None], identity_mat, Rs)

    # return R.reshape(batch_size, -1)
    
    batch_size = rvecs.shape[0]
    r_vecs = rvecs.reshape(-1, 3)
    angles = torch.norm(r_vecs, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), r_vecs * sin_half_angles_over_angles], dim=-1
    )
    
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    output = o.reshape(quaternions.shape[:-1] + (3, 3))
    
    return output.reshape(batch_size, -1)
    


def random_point_dropout(points, max_dropout_ratio=0.875):
    '''
    points: Tensor of shape (T, N, 3)
    '''
    T, N, _ = points.shape
    #device = points.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the same dropout ratio for all point clouds in the batch
    dropout_ratio = np.random.random() * max_dropout_ratio
    keep_num = int(N * (1 - dropout_ratio))
    
    # Generate a single random permutation for all point clouds
    idx = torch.randperm(N, device=device)[:keep_num]
    
    # Apply the same dropout to all point clouds in the batch
    dropped_points = points[:, idx, :]
    
    return dropped_points

def random_scale_point_cloud(points, scale_low=0.8, scale_high=1.25):
    '''
    points: Tensor of shape (T, N, 3)
    '''
    T, N, _ = points.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the same scale for all point clouds in the batch
    scale = torch.rand(1, device=device) * (scale_high - scale_low) + scale_low
    
    return points * scale

def shift_point_cloud(points, shift_range=0.1):
    '''
    points: Tensor of shape (T, N, 3)
    '''
    T, N, C = points.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the same shift for all point clouds in the batch
    shift = torch.rand((1, C), device=device) * shift_range
    
    return points + shift