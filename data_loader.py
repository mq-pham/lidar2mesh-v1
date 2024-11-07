import os
import argparse

import pickle
import torch
import smplx
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
import random

    

def camera_to_pixel(X, intrinsics, distortion_coefficients):
    # focal length
    f = intrinsics[:2]
    # center principal point
    c = intrinsics[2:]
    k = np.array([distortion_coefficients[0],
                 distortion_coefficients[1], distortion_coefficients[4]])
    p = np.array([distortion_coefficients[2], distortion_coefficients[3]])
    XX = X[..., :2] / (X[..., 2:])
    # XX = pd.to_numeric(XX, errors='coere')
    r2 = np.sum(XX[..., :2]**2, axis=-1, keepdims=True)

    radial = 1 + np.sum(k * np.concatenate((r2, r2**2, r2**3),
                        axis=-1), axis=-1, keepdims=True)

    tan = 2 * np.sum(p * XX[..., ::-1], axis=-1, keepdims=True)
    XXX = XX * (radial + tan) + r2 * p[..., ::-1]
    return f * XXX + c

def world_to_pixels(X, extrinsic_matrix, cam):
    B, N, dim = X.shape
    X = np.concatenate((X, np.ones((B, N, 1))), axis=-1).transpose(0, 2, 1)
    X = (extrinsic_matrix @ X).transpose(0, 2, 1)
    X = camera_to_pixel(X[..., :3].reshape(B*N, dim), cam['intrinsics'], [0]*5)
    X = X.reshape(B, N, -1)
    
    def check_pix(p):
        rule1 = p[:, 0] > 0
        rule2 = p[:, 0] < cam['width']
        rule3 = p[:, 1] > 0
        rule4 = p[:, 1] < cam['height']
        rule  = [a and b and c and d for a, b, c, d in zip(rule1, rule2, rule3, rule4)]
        return p[rule] if len(rule) > 50 else []
    
    X = [check_pix(xx) for xx in X]

    return X

def get_bool_from_coordinates(coordinates, shape=(1080, 1920)):
    bool_arr = np.zeros(shape, dtype=bool)
    if len(coordinates) > 0:
        bool_arr[coordinates[:, 0], coordinates[:, 1]] = True

    return bool_arr

""" 
def voxel_downsample(points: torch.Tensor, voxel_size: float):
    
    #Downsamples points using a voxel grid approach.
    
    #Args:
    #    points (torch.Tensor): A tensor containing 3D points of shape (N, 3).
    #    voxel_size (float): The size of each voxel.
        
    #Returns:
    #    torch.Tensor: A tensor of downsampled points.
   
    # Scale points to voxel space
    voxel_indices = (points / voxel_size).floor().long()
    unique_voxels, inverse_indices = torch.unique(voxel_indices, return_inverse=True, dim=0)
    
    # Average the points within each voxel
    downsampled_points = []
    for voxel in unique_voxels:
        mask = (inverse_indices == voxel[0])
        if mask.any():
            avg_point = points[mask].mean(dim=0)
            downsampled_points.append(avg_point)
    
    return torch.stack(downsampled_points)

def fix_points_num(points: np.array, num_points: int):
    
    #Downs samples the points using voxel and uniform downsampling,
    #and either repeats or randomly selects points to reach the desired number.
    
    #Args:
    #  points (np.array): a numpy array containing 3D points.
    #  num_points (int): the desired number of points 
    
    #Returns:
    #  a numpy array `(num_points, 3)`
    
    if len(points) == 0:
        return np.zeros((num_points, 3))
    
    # Remove NaN points
    points = points[~np.isnan(points).any(axis=-1)]
    
    # Convert points to a torch tensor
    points_tensor = torch.tensor(points, dtype=torch.float32)

    origin_num_points = points_tensor.shape[0]

    if origin_num_points < num_points:
        # Repeat points to reach the desired number
        num_whole_repeat = num_points // origin_num_points
        res = points_tensor.repeat(num_whole_repeat, 1)
        num_remain = num_points % origin_num_points
        if num_remain > 0:
            res = torch.cat((res, points_tensor[:num_remain]), dim=0)
    else:
        
        # Perform voxel downsampling
        downsampled_points = voxel_downsample(points_tensor, voxel_size=0.05)
        # Randomly sample from downsampled points
        indices = torch.randint(0, origin_num_points, (num_points,))
        res = downsampled_points[indices]

    return res.numpy()  # Convert back to numpy array

"""

##########
# TOBE UPDATE 


def fix_points_num(points: np.array, num_points: int):
  
    #downsamples the points using voxel and uniform downsampling, 
    #and either repeats or randomly selects points to reach the desired number.
    
    #Args:
    #  points (np.array): a numpy array containing 3D points.
    #  num_points (int): the desired number of points 
    
    #Returns:
    #  a numpy array `(num_points, 3)`
    
    if len(points) == 0:
        return np.zeros((num_points, 3))
    points = points[~np.isnan(points).any(axis=-1)]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc = pc.voxel_down_sample(voxel_size=0.05)
    ratio = int(len(pc.points) / num_points + 0.05)
    if ratio > 1:
        pc = pc.uniform_down_sample(ratio)

    points = np.asarray(pc.points)
    origin_num_points = points.shape[0]

    if origin_num_points < num_points:
        num_whole_repeat = num_points // origin_num_points
        res = points.repeat(num_whole_repeat, axis=0)
        num_remain = num_points % origin_num_points
        res = np.vstack((res, res[:num_remain]))
    else:
        res = points[np.random.choice(origin_num_points, num_points)]
    return res


INTRINSICS = [599.628, 599.466, 971.613, 540.258]
DIST       = [0.003, -0.003, -0.001, 0.004, 0.0]
LIDAR2CAM  = [[[-0.0355545576, -0.999323133, -0.0094419378, -0.00330376451], 
              [0.00117895777, 0.00940596282, -0.999955068, -0.0498469479], 
              [0.999367041, -0.0355640917, 0.00084373493, -0.0994979365], 
              [0.0, 0.0, 0.0, 1.0]]]

class SLOPER4D_Dataset(Dataset):
    def __init__(self, pkl_file, 
                 device='cpu', 
                 return_torch:bool=True, 
                 fix_pts_num:int=1024,
                 print_info:bool=True,
                 return_smpl:bool=True):
        
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        self.data         = data
        self.pkl_file     = pkl_file
        self.device       = device
        self.return_torch = return_torch
        self.print_info   = print_info
        self.fix_pts_num  = fix_pts_num
        self.return_smpl  = return_smpl
        
        self.framerate = data['framerate'] # scalar
        self.length    = data['total_frames'] if 'total_frames' in data else len(data['frame_num'])

        self.world2lidar, self.lidar_tstamps = self.get_lidar_data()
        self.load_3d_data(data)    
        self.load_rgb_data(data)
        self.load_mask(pkl_file)

        self.check_length()

    def get_lidar_data(self, is_inv=True):
        lidar_traj    = self.data['first_person']['lidar_traj'].copy()
        lidar_tstamps = lidar_traj[:self.length, -1]
        world2lidar   = np.array([np.eye(4)] * self.length)
        world2lidar[:, :3, :3] = R.from_quat(lidar_traj[:self.length, 4: 8]).inv().as_matrix()
        world2lidar[:, :3, 3:] = -world2lidar[:, :3, :3] @ lidar_traj[:self.length, 1:4].reshape(-1, 3, 1)

        return world2lidar, lidar_tstamps
    
    def load_rgb_data(self, data):
        try:
            self.cam = data['RGB_info']     
        except:
            print('=====> Load default camera parameters.')
            self.cam = {'fps':20, 'width': 1920, 'height':1080, 
                        'intrinsics':INTRINSICS, 'lidar2cam':LIDAR2CAM, 'dist':DIST}
            
        if 'RGB_frames' not in data:
            data['RGB_frames'] = {}
            world2lidar, lidar_tstamps = self.get_lidar_data()
            data['RGB_frames']['file_basename'] = [''] * self.length
            data['RGB_frames']['lidar_tstamps'] = lidar_tstamps[:self.length]
            data['RGB_frames']['bbox']          = [[]] * self.length
            data['RGB_frames']['skel_2d']       = [[]] * self.length
            data['RGB_frames']['cam_pose']      = self.cam['lidar2cam'] @ world2lidar
            self.save_pkl(overwrite=True)

        self.file_basename = data['RGB_frames']['file_basename'] # synchronized img file names
        self.lidar_tstamps = data['RGB_frames']['lidar_tstamps'] # synchronized ldiar timestamps
        self.bbox          = data['RGB_frames']['bbox']          # 2D bbox of the tracked human (N, [x1, y1, x2, y2])
        self.skel_2d       = data['RGB_frames']['skel_2d']       # 2D keypoints (N, [17, 3]), every joint is (x, y, probability)
        self.cam_pose      = data['RGB_frames']['cam_pose']      # extrinsic, world to camera (N, [4, 4])

        if self.return_smpl:
            self.smpl_verts, self.smpl_joints = self.return_smpl_verts()
            self.smpl_mask = world_to_pixels(self.smpl_verts, self.cam_pose, self.cam)

    def load_mask(self, pkl_file):
        mask_pkl = pkl_file[:-4] + "_mask.pkl"
        if os.path.exists(mask_pkl):
            with open(mask_pkl, 'rb') as f:
                print(f'Loading: {mask_pkl}')
                self.masks = pickle.load(f)['masks']
        else:
            self.masks = [[]]*self.length

    def load_3d_data(self, data, person='second_person'):
        assert self.length <= len(data['frame_num']), f"RGB length must be less than point cloud length"
        point_clouds = [[]] * self.length
        if 'point_clouds' in data[person]:
            for i, pf in enumerate(data[person]['point_frame']):
                index = data['frame_num'].index(pf)
                if index < self.length:
                    point_clouds[index] = data[person]['point_clouds'][i]
        if self.fix_pts_num is not None:
            point_clouds = np.array([fix_points_num(pts, self.fix_pts_num) for pts in point_clouds])

        sp = data['second_person']
        self.smpl_pose    = sp['opt_pose'][:self.length].astype(np.float32)  # n x 72 array of scalars
        #self.global_trans = sp['opt_trans'][:self.length].astype(np.float32) # n x 3 array of scalars
        self.global_trans = np.zeros((self.length, 3)).astype(np.float32) # n x 3 array of scalars
        self.betas        = sp['beta']                                       # n x 10 array of scalars
        self.smpl_gender  = sp['gender']                                     # male/female/neutral    

        
        # Ensure point_clouds is a list of numpy arrays
        self.human_points = []

        for i, pc in enumerate(point_clouds):
            # Subtract the corresponding translation for each point cloud
            pc -= sp['opt_trans'][i].astype(np.float32)
            self.human_points.append(pc)
        #self.human_points = point_clouds                                     # list of n arrays, each of shape (x_i, 3)

    def updata_pkl(self, img_name, 
                   bbox=None, 
                   cam_pose=None, 
                   keypoints=None):
        if img_name in self.file_basename:
            index = self.file_basename.index(img_name)
            if bbox is not None:
                self.data['RGB_frames']['bbox'][index] = bbox
            if keypoints is not None:
                self.data['RGB_frames']['skel_2d'][index] = keypoints
            if cam_pose is not None:
                self.data['RGB_frames']['cam_pose'][index] = cam_pose
        else:
            print(f"{img_name} is not in the synchronized labels file")
    
    def get_rgb_frames(self, ):
        return self.data['RGB_frames']

    def save_pkl(self, overwrite=False):
        
        save_path = self.pkl_file if overwrite else self.pkl_file[:-4] + '_updated.pkl' 
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"{save_path} saved")

    def check_length(self):
        # Check if all the lists inside rgb_frames have the same length
        assert all(len(lst) == self.length for lst in [self.bbox, self.skel_2d,  
                                                       self.lidar_tstamps, self.masks, 
                                                       self.smpl_pose, self.global_trans, 
                                                       self.human_points])

        print(f'Data length: {self.length}')
        
    def get_cam_params(self): 
        return torch.from_numpy(np.array(self.cam['lidar2cam']).astype(np.float32)).to(self.device), \
               torch.from_numpy(np.array(self.cam['intrinsics']).astype(np.float32)).to(self.device), \
               torch.from_numpy(np.array(self.cam['dist']).astype(np.float32)).to(self.device)
            
    def get_img_shape(self):
        return self.cam['width'], self.cam['height']

    def return_smpl_verts(self, ):
        file_path = os.path.dirname(os.path.abspath(__file__))
        with torch.no_grad():
            human_model = smplx.create(f"{os.path.dirname(file_path)}/lidar2mesh-v1/",
                                    gender=self.smpl_gender, 
                                    use_face_contour=False,
                                    ext="npz")
            orient = torch.tensor(self.smpl_pose).float()[:, :3]
            bpose  = torch.tensor(self.smpl_pose).float()[:, 3:]
            transl = torch.tensor(self.global_trans).float()
            smpl_md = human_model(betas=torch.tensor(self.betas).reshape(-1, 10).float(), 
                                    return_verts=True, 
                                    pose2rot=True,
                                    body_pose=bpose,
                                    global_orient=orient,
                                    transl=transl)
            
        return smpl_md.vertices.numpy(), smpl_md.joints.numpy()
            
    def __getitem__(self, index):
        sample = {
            'smpl_pose'    : torch.tensor(self.smpl_pose[index]).float().to(self.device),
            'global_trans' : torch.tensor(self.global_trans[index]).float().to(self.device),
            'betas'        : torch.tensor(self.betas).float().to(self.device), 
            'pcs'          : self.human_points[index],     
            'smpl_verts'   : self.smpl_verts[index],        
            'joints'       : self.smpl_joints[index],
            'gender'       : self.smpl_gender
        }

        if self.return_torch:
            for k, v in sample.items():
                if type(v) != str and type(v) != torch.Tensor:
                    sample[k] = torch.tensor(v).float().to(self.device)

        mispart = ''
        mispart += 'pts ' if len(sample['pcs']) < 1 else ''
           
        if len(mispart) > 0 and self.print_info:
            print(f'Missing {mispart} in: {index} ')

        return sample
    
    def __len__(self):
        return self.length
    
    
class ConsecutiveFrameBatchSampler(Sampler):
    def __init__(self, indices, batch_size, shuffle=False):
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices into batches of consecutive frames
        self.batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
        
        # Remove the last batch if it's not full
        if len(self.batches[-1]) != batch_size:
            self.batches.pop()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

def create_dataloaders(dataset, batch_size, test_frames=range(3000, 3100)):
    
    print("Creating dataloaders...")
    print(f"Total frames: {len(dataset)}")
    print(f"Bath size: {batch_size}")
    
    total_frames = len(dataset)
    
    # Create indices for all frames
    all_indices = set(range(total_frames))
    
    # Create test dataset
    test_indices = set(test_frames)
    test_dataset = Subset(dataset, sorted(list(test_indices)))
    
    # Create train and val datasets
    remaining_indices = sorted(list(all_indices - test_indices))
    num_train = int(0.9 * len(remaining_indices))
    
    train_indices = remaining_indices[:num_train]
    val_indices = remaining_indices[num_train:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Train frames: {len(train_dataset)}")
    print(f"Val frames: {len(val_dataset)}")
    print(f"Test frames: {len(test_dataset)}")

    train_sampler = ConsecutiveFrameBatchSampler(train_indices, batch_size, shuffle=True)
    val_sampler = ConsecutiveFrameBatchSampler(val_indices, batch_size, shuffle=True)
    test_sampler = ConsecutiveFrameBatchSampler(sorted(list(test_indices)), batch_size, shuffle=False)

    train_loader = DataLoader(dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_sampler=test_sampler)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


# def create_dataloaders(dataset, batch_size, test_frames=100, sample_size=200):
#     total_frames = len(dataset)
    
#     if sample_size and sample_size < total_frames:
#         # Si un sample_size est spécifié et qu'il est inférieur au nombre total de frames
#         indices = random.sample(range(total_frames), sample_size)
#         test_frames = min(test_frames, int(0.1 * sample_size))  # Ajuster test_frames si nécessaire
#     else:
#         indices = list(range(total_frames))
#         sample_size = total_frames
    
#     # Calculer les tailles des ensembles
#     train_val_frames = sample_size - test_frames
#     train_frames = int(0.9 * train_val_frames)
#     val_frames = train_val_frames - train_frames
    
#     # Mélanger les indices si un échantillon est utilisé
#     if sample_size < total_frames:
#         random.shuffle(indices)
    
#     # Créer les sous-ensembles
#     train_dataset = Subset(dataset, indices[:train_frames])
#     val_dataset = Subset(dataset, indices[train_frames:train_frames+val_frames])
#     test_dataset = Subset(dataset, indices[-test_frames:])
    
#     # Créer les samplers
#     train_sampler = BatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
#     val_sampler = BatchSampler(val_dataset, batch_size=batch_size, shuffle=True)
#     test_sampler = BatchSampler(test_dataset, batch_size=batch_size, shuffle=False)
    
#     # Créer les loaders
#     train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
#     val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
#     test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
    
#     return train_loader, val_loader, test_loader
    
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLOPER4D dataset')
    parser.add_argument('--pkl_file', type=str, 
                        default='./data/SLOPER4D/seq002_football_001/seq002_football_001_labels.pkl',  
                        help='Path to the pkl file')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='The batch size of the data loader')
    parser.add_argument('--index', type=int, default=-1,
                        help='the index frame to be saved to a image')
    args = parser.parse_args()
    
    dataset = SLOPER4D_Dataset(args.pkl_file, 
                               return_torch=True, 
                               fix_pts_num=True)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    root_folder = os.path.dirname(args.pkl_file)
    
    print(f"Total frames: {len(dataset)}")
    print("First sample of the dataloader:", next(iter(dataloader)))

    for index, sample in enumerate(dataloader):
        for i in range(args.batch_size):
            pcd_name  = f"{sample['lidar_tstamps'][i]:.03f}".replace('.', '_') + '.pcd'
            img_path  = os.path.join(root_folder, 'rgb_data', sample['file_basename'][i])
            pcd_path  = os.path.join(root_folder, 'lidar_data', 'lidar_frames_rot', pcd_name)
            extrinsic = sample['cam_pose'][i]      # 4x4 lidar to camera transformation
            keypoints = sample['skel_2d']       # 2D keypoints, coco17 style
            if index == args.index:
                print(f"{index} pcd path: {pcd_path}")
                print(f"{index} img path: {img_path}")