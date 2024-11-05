import math
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from pytorch3d.structures import Meshes, Pointclouds
from torch.utils.data import random_split
import numpy as np
from utils import random_point_dropout, random_scale_point_cloud, shift_point_cloud, load_data

class MeshPointCloudDataset(Dataset):
    def __init__(self, _input, verts, joints, pcs, azimuths, device, batchSize, dataset_name):
        # self.poses = poses
        # self.betas = betas
        self.input = _input  
        self.verts = verts
        self.joints = joints
        self.pcs = pcs
        
        self.azimuths = azimuths
        self.device = device
        self.batchSize = batchSize
        self.dataset_name = dataset_name

    def __len__(self):
        return math.ceil(len(self.pcs)/self.batchSize)

    def __getitem__(self, idx):
        start_index = idx * self.batchSize
        end_index = min((idx + 1) * self.batchSize, len(self.pcs))

        # Fetch and process batch point clouds
        batch_pcs = [item.to(self.device) for item in self.pcs[start_index:end_index]]
        batch_pcs = Pointclouds(
            points=[pc.points_packed() for pc in batch_pcs], 
            features=None
        )
        batch_pcs = batch_pcs.points_padded()

        # Fetch and process other attributes
        # batch_poses = torch.stack([item.to(self.device) for item in self.poses[start_index:end_index]])
        # batch_betas = torch.stack([item.to(self.device) for item in self.betas[start_index:end_index]])
        batch_input = torch.stack([item.to(self.device) for item in self.input[start_index:end_index]])
        batch_verts = torch.stack([item.to(self.device) for item in self.verts[start_index:end_index]])
        batch_joints = torch.stack([item.to(self.device) for item in self.joints[start_index:end_index]])
        batch_azimuths = torch.stack([item.to(self.device) for item in self.azimuths[start_index:end_index]])

        return batch_input, batch_verts, batch_joints, batch_pcs, batch_azimuths, self.dataset_name



def batchframe(index, batchSize, allframes):
    batchf = []
    for ind in index:
        start = ind * batchSize
        end = start + batchSize
        if end <= len(allframes):
            batchf.extend(allframes[start:end])
    return batchf


def split_single_dataset(_input, verts, joints, pcs, azimuths, device, batchSize, dataset_name, train_size=0.5, val_size=0.25, test_size=0.25, augment=1):
    nb_frame = len(pcs)
    nb_complete_batches = nb_frame // batchSize
    
    
    # Adjust split sizes to ensure they sum to 1
    total = train_size + val_size + test_size
    train_size, val_size, test_size = train_size/total, val_size/total, test_size/total
    
    # Calculate the number of batches for each split
    train_batches = math.floor(nb_complete_batches * train_size)
    val_batches = math.floor(nb_complete_batches * val_size)
    test_batches = nb_complete_batches - train_batches - val_batches
    
    
    # Create batch indices
    batch_indices = list(range(nb_complete_batches))
    train_ind, val_ind, test_ind = random_split(batch_indices, [train_batches, val_batches, test_batches])

    
    # Prepare data for each split
    #real_poses_train = batchframe(train_ind, batchSize, poses)
    #real_beta_train = batchframe(train_ind, batchSize, betas)
    real_input_train = batchframe(train_ind, batchSize, _input)
    real_point_clouds_train = batchframe(train_ind, batchSize, pcs)
    real_verts_train = batchframe(train_ind, batchSize, verts)
    real_joints_train = batchframe(train_ind, batchSize, joints)
    azimuths_train = batchframe(train_ind, batchSize, azimuths)
    
    
    if augment > 1:
        augmented_point_clouds = []
        for _ in range(augment - 1):  # Subtract 1 because we already have the original data
            for pc in real_point_clouds_train:
                points = pc.points_padded()
                
                # Apply augmentations
                augmented_points = random_scale_point_cloud(points, scale_low=0.8, scale_high=1.25)
                augmented_points = shift_point_cloud(augmented_points, shift_range=0.1)
                augmented_points = random_point_dropout(augmented_points, max_dropout_ratio=0.875)
                
                # Create new Pointclouds object with augmented data
                augmented_pc = Pointclouds(points=augmented_points, features=None)
                augmented_point_clouds.append(augmented_pc)
        
        # Combine original and augmented point clouds
        real_point_clouds_train.extend(augmented_point_clouds)

        # Double other training data
        #real_poses_train = real_poses_train * augment
        #real_beta_train = real_beta_train * augment
        real_input_train = real_input_train * augment
        real_verts_train = real_verts_train * augment
        real_joints_train = real_joints_train * augment
        azimuths_train = azimuths_train * augment
        

    # real_poses_val = batchframe(val_ind, batchSize, poses)
    # real_beta_val = batchframe(val_ind, batchSize, betas)
    real_input_val = batchframe(val_ind, batchSize, _input)
    real_point_clouds_val = batchframe(val_ind, batchSize, pcs)
    real_verts_val = batchframe(val_ind, batchSize, verts)
    real_joints_val = batchframe(val_ind, batchSize, joints)
    azimuths_val = batchframe(val_ind, batchSize, azimuths)

    # real_poses_test = batchframe(test_ind, batchSize, poses)
    # real_beta_test = batchframe(test_ind, batchSize, betas)
    real_input_test = batchframe(test_ind, batchSize, _input)
    real_point_clouds_test = batchframe(test_ind, batchSize, pcs)
    real_verts_test = batchframe(test_ind, batchSize, verts)
    real_joints_test = batchframe(test_ind, batchSize, joints)
    azimuths_test = batchframe(test_ind, batchSize, azimuths)

    # Create datasets
    train_dataset = MeshPointCloudDataset(real_input_train, real_verts_train, real_joints_train, real_point_clouds_train, azimuths_train, device, batchSize, dataset_name)
    val_dataset = MeshPointCloudDataset(real_input_val, real_verts_val, real_joints_val, real_point_clouds_val, azimuths_val, device, batchSize, dataset_name)
    test_dataset = MeshPointCloudDataset(real_input_test, real_verts_test, real_joints_test, real_point_clouds_test, azimuths_test, device, batchSize, dataset_name)

    return train_dataset, val_dataset, test_dataset


def shuffle_dataset(dataset, seed=42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    return Subset(dataset, indices)


def load_and_split_all_datasets(list_data_names, device, smpl, lidar, batchSize, train_size, val_size, test_size, step, augment, opti_trans):
    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []

    for data_name in list_data_names:
        print(f"Processing {data_name}")
        
        # Use the updated load_data function
        _input, verts, joints, pcs, azimuths, _ = load_data(device, smpl, lidar, dataset=data_name, step=step, opti_trans=opti_trans)

        train_dataset, val_dataset, test_dataset = split_single_dataset(
            _input, verts, joints, pcs, azimuths,
            device, batchSize, data_name, train_size=train_size, val_size=val_size, test_size=test_size, augment=augment)
        
        all_train_datasets.append(train_dataset)
        all_val_datasets.append(val_dataset)
        all_test_datasets.append(test_dataset)

    combined_train_dataset = ConcatDataset(all_train_datasets)
    combined_val_dataset = ConcatDataset(all_val_datasets)
    combined_test_dataset = ConcatDataset(all_test_datasets)

    # Shuffle the combined datasets
    combined_train_dataset = shuffle_dataset(combined_train_dataset)
    combined_val_dataset = shuffle_dataset(combined_val_dataset)
    combined_test_dataset = shuffle_dataset(combined_test_dataset)

    return combined_train_dataset, combined_val_dataset, combined_test_dataset

