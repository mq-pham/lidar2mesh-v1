import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

def train_one_epoch(model, optimizer, criterion, train_loader, epoch, total_epochs):
    model.train()
    total_loss = 0
    loss_components_sum = {comp: 0 for comp in criterion.components}
    loss_components_sum['total_loss'] = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch")
    for i, batch in enumerate(pbar, 1):
        smpl_pose = batch['smpl_pose']
        trans = batch['global_trans']
        betas = batch['betas']
        pcs = batch['pcs']
        gender = batch['gender'][0]  # Assuming gender is the same for the whole batch
        _input = torch.cat([trans, smpl_pose, betas], dim=1)
        pcs = pcs.unsqueeze(0)
        output = model(pcs)
        loss, loss_components, _, _, _, _, _, _ = criterion(_input, output, gender)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        for comp in loss_components:
            loss_components_sum[comp] += loss_components[comp].item()
        pbar.set_description(f"Epoch {epoch}/{total_epochs} Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    avg_loss_components = {comp: loss_components_sum[comp] / len(train_loader) for comp in loss_components_sum}
    return avg_loss, avg_loss_components

def evaluate(model, criterion, val_loader, epoch, total_epochs):
    model.eval()
    total_loss = 0
    loss_components_sum = {comp: 0 for comp in criterion.components}
    loss_components_sum['total_loss'] = 0
    pbar = tqdm(val_loader, desc=f"Validation {epoch}/{total_epochs}", unit="batch")
    with torch.no_grad():
        for i, batch in enumerate(pbar, 1):
            smpl_pose = batch['smpl_pose']
            trans = batch['global_trans']
            betas = batch['betas']
            pcs = batch['pcs']
            gender = batch['gender'][0]  # Assuming gender is the same for the whole batch
            _input = torch.cat([trans, smpl_pose, betas], dim=1)
            pcs = pcs.unsqueeze(0)
            output = model(pcs)
            loss, loss_components, _, _, _, _, _, _ = criterion(_input, output, gender)
            total_loss += loss.item()
            for comp in loss_components:
                loss_components_sum[comp] += loss_components[comp].item()
            pbar.set_description(f"Validation {epoch}/{total_epochs} Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(val_loader)
    avg_loss_components = {comp: loss_components_sum[comp] / len(val_loader) for comp in loss_components_sum}
    return avg_loss, avg_loss_components

def compute_mpvpe(pred_verts, real_verts):
    """Mean Per Vertex Position Error (MPVPE)"""
    return torch.mean(torch.norm(pred_verts - real_verts, dim=-1))

def compute_mrvpv(pred_verts, p=2):
    """Mean Running Vertex Position Variation (MRVPV)"""
    return torch.mean(torch.norm(pred_verts[1:] - pred_verts[:-1], dim=-1, p=p))

def compute_mrsv(pred_betas, p=2):
    """Mean Running Shape Variation (MRSV)"""
    return torch.mean(torch.norm(pred_betas[1:] - pred_betas[:-1], dim=-1, p=p))


def visualize_results(real_verts, pred_verts, pcs, start_frame_num, output_dir):
    batch_size = real_verts.shape[0]
    for i in range(batch_size):
        fig = plt.figure(figsize=(20, 10))
        
        # Real mesh
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(real_verts[i, :, 0], real_verts[i, :, 1], real_verts[i, :, 2], c='b', s=0.05)
        ax1.set_title('Real Mesh')
        
        # Reconstructed mesh
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(pred_verts[i, :, 0], pred_verts[i, :, 1], pred_verts[i, :, 2], c='r', s=0.05)
        ax2.set_title('Reconstructed Mesh')
        
        # Real point cloud
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(pcs[i, :, 0], pcs[i, :, 1], pcs[i, :, 2], c='g', s=1)
        ax3.set_title('Real Point Cloud')
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.view_init(elev=20, azim=30)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'frame_{start_frame_num + i:04d}.png'))
        plt.close()

def test(model, criterion, test_loader, experiment_dir, run_dir):
    model.eval()
    total_loss = 0
    pbar = tqdm(test_loader, desc="Testing", unit="batch")
    
    mpvpe_sum = 0
    mrvpv_1_sum = 0
    mrvpv_2_sum = 0
    mrsv_sum = 0
    num_batches = 0
    total_frames = 0
    
    # Create a subdirectory for frame images in the experiment directory
    frames_dir = os.path.join(experiment_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            smpl_pose = batch['smpl_pose']
            trans = batch['global_trans']
            betas = batch['betas']
            pcs = batch['pcs']
            gender = batch['gender'][0]  # Assuming gender is the same for the whole batch
            
            batch_size = smpl_pose.shape[0]
            _input = torch.cat([trans, smpl_pose, betas], dim=1)
            pcs = pcs.unsqueeze(0)
            
            output = model(pcs)
            loss, loss_components, real_verts, pred_verts, real_joints, pred_joints, real_betas, pred_betas = criterion(_input, output, gender)
            
            total_loss += loss.item()
            
            # Compute metrics for this batch
            mpvpe = compute_mpvpe(pred_verts, real_verts)
            mrvpv_1 = compute_mrvpv(pred_verts, p=1)
            mrvpv_2 = compute_mrvpv(pred_verts, p=2)
            mrsv = compute_mrsv(pred_betas)
            
            # Accumulate metrics
            mpvpe_sum += mpvpe.item()
            mrvpv_1_sum += mrvpv_1.item()
            mrvpv_2_sum += mrvpv_2.item()
            mrsv_sum += mrsv.item()
            num_batches += 1
            total_frames += batch_size
            
            pbar.set_description(f"Testing Loss: {loss.item():.4f}, MPVPE: {mpvpe.item():.4f}")
            
            # Visualize all frames in the batch
            visualize_results(real_verts.cpu(), pred_verts.cpu(), pcs[0].cpu(), total_frames - batch_size, frames_dir)

    # Create gif
    images = []
    for filename in sorted(os.listdir(frames_dir)):
        if filename.endswith('.png'):
            images.append(imageio.imread(os.path.join(frames_dir, filename)))
    
    if images:
        imageio.mimsave(os.path.join(run_dir, 'visualization.gif'), images, fps=5)
    else:
        print("No images found for GIF creation")
    
    # Delete the .png files
    for filename in os.listdir(frames_dir):
        if filename.endswith('.png'):
            os.remove(os.path.join(frames_dir, filename))

    # Compute average metrics
    avg_loss = total_loss / num_batches
    avg_mpvpe = mpvpe_sum / num_batches
    avg_mrvpv_1 = mrvpv_1_sum / num_batches
    avg_mrvpv_2 = mrvpv_2_sum / num_batches
    avg_mrsv = mrsv_sum / num_batches

    print(f"Test results:")
    print(f"    Average Loss: {avg_loss:.4f}")
    print(f"    Average MPVPE: {avg_mpvpe:.4f}")
    print(f"    Average MRVPV (L1): {avg_mrvpv_1:.4f}")
    print(f"    Average MRVPV (L2): {avg_mrvpv_2:.4f}")
    print(f"    Average MRSV: {avg_mrsv:.4f}")

    return {
        'test_loss': avg_loss,
        'mpvpe': avg_mpvpe,
        'mrvpv_1': avg_mrvpv_1,
        'mrvpv_2': avg_mrvpv_2,
        'mrsv': avg_mrsv
    }