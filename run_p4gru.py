import yaml
import torch
import copy
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from pathlib import Path
from trainer import train_one_epoch, evaluate, test
from loss import Loss, WarmupMultiStepLR
from p4gru import P4GRU
from utils import init_weights, optimize_memory
from data_loader import SLOPER4D_Dataset, create_dataloaders
from torch.utils.data import DataLoader
import psutil
import GPUtil
import time
import threading

def monitor_resources(interval=5):
    while True:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        gpus = GPUtil.getGPUs()
        gpu_load = gpus[0].load * 100 if gpus else 0
        gpu_memory = gpus[0].memoryUtil * 100 if gpus else 0
        
        print(f"CPU: {cpu_percent}% | RAM: {memory_percent}% | GPU Load: {gpu_load:.2f}% | GPU Memory: {gpu_memory:.2f}%")
        
        time.sleep(interval)

def load_config(config_filename: str = "config.yaml") -> Dict[str, Any]:
    current_dir = Path(__file__).parent.absolute()
    config_path = current_dir / config_filename

    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, is_best, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    checkpoint_path = checkpoint_dir / 'latest_checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_checkpoint_path = checkpoint_dir / 'best_checkpoint.pth'
        torch.save(checkpoint, best_checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_dir, best=False):
    checkpoint_path = checkpoint_dir / ('best_checkpoint.pth' if best else 'latest_checkpoint.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Loaded {'best' if best else 'latest'} checkpoint from epoch {start_epoch - 1}")
        return start_epoch, best_val_loss
    else:
        print("No checkpoint found, starting from scratch")
        return 0, float('inf')
    
    
def plot_losses(loss_data, save_path):
    epochs = loss_data['Epoch']
    train_loss = loss_data['Train Loss']
    val_loss = loss_data['Val Loss']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_aggregated_losses(experiment_dir: Path):
    run_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith('run')]
    
    all_train_losses = []
    all_val_losses = []
    epochs = None

    for run_dir in run_dirs:
        with open(run_dir / 'loss_values.csv', 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            if epochs is None:
                epochs = [int(row['Epoch']) for row in data]
            all_train_losses.append([float(row['Train Loss']) for row in data])
            all_val_losses.append([float(row['Val Loss']) for row in data])

    mean_train_loss = np.mean(all_train_losses, axis=0)
    std_train_loss = np.std(all_train_losses, axis=0)
    mean_val_loss = np.mean(all_val_losses, axis=0)
    std_val_loss = np.std(all_val_losses, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_train_loss, label='Mean Train Loss')
    plt.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.3)
    plt.plot(epochs, mean_val_loss, label='Mean Validation Loss')
    plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Aggregated Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(experiment_dir / 'aggregated_losses.png')
    plt.close()
    
    
def adjust_loss_config(config, components):
    new_config = copy.deepcopy(config)
    new_config['loss']['components'] = components
    return new_config

def run_simulation(config: Dict[str, Any], experiment_dir: Path, run_dir: Path,
                   train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model setup
    factor = 3 if config['model_architecture']['_type'] == "3D" else 6
    output_dim = (factor * 24 + (10 if config['optimization']['opti_beta'] else 0) + (3 if config['optimization']['opti_trans'] else 0)) * config['training']['batch_size']
    print(f"Output dim: {output_dim}")
    model = P4GRU(
        config=config['model_architecture'],
        output_dim=output_dim,
        debug_print=config['debug']['print'],
        visu=config['debug']['visualize']
    ).to(device)
    
    if config['model_architecture']['init_layers']:
        model.apply(init_weights)
    
    optimize_memory()
    
    # Loss and optimizer setup
    criterion = Loss(device=device, config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    # Load latest checkpoint
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, run_dir)
    
    # LR scheduler setup
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=config['lr_scheduler']['milestones'],
        gamma=config['lr_scheduler']['gamma'],
        warmup_factor=config['lr_scheduler']['warmup_factor'],
        warmup_iters=config['lr_scheduler']['warmup_iters'] * len(train_loader),
        warmup_method=config['lr_scheduler']['warmup_method'],
        last_epoch=start_epoch - 1
    )
    
    # CSV file for storing loss values
    csv_path = run_dir / 'loss_values.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write header with all loss components
    header = ['Epoch', 'Train Loss', 'Val Loss']
    header.extend([f'Train_{comp}' for comp in config['loss']['components']])
    header.extend([f'Val_{comp}' for comp in config['loss']['components']])
    csv_writer.writerow(header)
    
    # Start resource monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_resources, args=(30,))  # Update every 10 seconds
    monitor_thread.daemon = True  # This ensures the thread will exit when the main program exits
    monitor_thread.start()
    
    loss_data = {h: [] for h in header}
    
    # Training loop
    try:
        for epoch in range(start_epoch+1, config['training']['num_epochs']):
            train_loss, train_loss_components = train_one_epoch(model, optimizer, criterion, train_loader, epoch, config['training']['num_epochs'])
            val_loss, val_loss_components = evaluate(model, criterion, val_loader, epoch, config['training']['num_epochs'])
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Write loss values to CSV and store for plotting
            row = [epoch, train_loss, val_loss]
            row.extend([train_loss_components[comp] for comp in config['loss']['components']])
            row.extend([val_loss_components[comp] for comp in config['loss']['components']])
            csv_writer.writerow(row)
            csv_file.flush()  # Ensure data is written to file
            

            for h, v in zip(header, row):
                loss_data[h].append(v)
            
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
                    
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, is_best, run_dir)
            
            lr_scheduler.step()
        
        csv_file.close()
        print("Training completed")
        optimize_memory()
        
        # Plot losses
        plot_losses(loss_data, run_dir / 'loss_plot.png')
        
        # Load best checkpoint for testing
        load_checkpoint(model, optimizer, run_dir, best=True)
        
        # Testing and final evaluation
        eval_results = test(model, criterion, test_loader, experiment_dir, run_dir)
        
        # Save evaluation results
        with open(run_dir / 'eval_results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=eval_results.keys())
            writer.writeheader()
            writer.writerow(eval_results)
        
        optimize_memory()
        
        return eval_results

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        # You might want to save a checkpoint here
        return None
        
    finally:
        # This ensures that resources are properly released even if an exception occurs
        csv_file.close()

def aggregate_results(experiment_dir: Path):
    run_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith('run')]
    
    all_results = []
    for run_dir in run_dirs:
        with open(run_dir / 'eval_results.csv', 'r') as f:
            reader = csv.DictReader(f)
            all_results.append(next(reader))
    
    mean_results = {}
    std_results = {}
    for key in all_results[0].keys():
        values = [float(result[key]) for result in all_results]
        mean_results[key] = np.mean(values)
        std_results[key] = np.std(values)
    
    with open(experiment_dir / 'aggregated_results.csv', 'w', newline='') as f:
        fieldnames = [f'{key}_mean' for key in mean_results.keys()] + [f'{key}_std' for key in std_results.keys()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({**{f'{k}_mean': v for k, v in mean_results.items()},
                         **{f'{k}_std': v for k, v in std_results.items()}})

def run_ablation_study(config: Dict[str, Any], 
                       train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                       ablation_name: str, param_name: str, param_values: list):
    results = []
    ablation_dir = Path(f"ablation_studies/{ablation_name}")
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    for value in param_values:
        ablation_config = copy.deepcopy(config)
        
        if param_name == 'loss.components':
            ablation_config = adjust_loss_config(ablation_config, value)
        else:
            keys = param_name.split('.')
            current = ablation_config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value
            
        if isinstance(value, list):
            value_str = '_'.join(map(str, value))
        else:
            value_str = str(value)
        
        experiment_name = f"{ablation_name}_{value_str}"
        experiment_dir = ablation_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create frames directory in experiment_dir
        frames_dir = experiment_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        
        run_results = []
        for i in range(3):  # Run 3 times for mean and std
            print(f"Running experiment: {experiment_name}, run {i+1}/3")
            run_dir = experiment_dir / f"run{i+1}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            run_result = run_simulation(ablation_config, experiment_dir, run_dir, train_loader, val_loader, test_loader)
            run_results.append(run_result)
        
        # Aggregate results
        aggregate_results(experiment_dir)
        
        # Plot aggregated losses
        plot_aggregated_losses(experiment_dir)
        
        # Clean up temporary frame files
        for png_file in frames_dir.glob('*.png'):
            png_file.unlink()
        
    print(f"Ablation study for {ablation_name} completed.")

def main():
    config = load_config("config.yaml")
    
    # Dataset setup
    pkl_file = Path(config['data']['paths'][0])
    pkl_file = Path(__file__).parent.absolute() / pkl_file
    dataset = SLOPER4D_Dataset(pkl_file=str(pkl_file),
                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                               return_torch=True,
                               fix_pts_num=config['lidar']['sample'],
                               print_info=True,
                               return_smpl=True)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(dataset, config['training']['batch_size'], test_frames=range(3000, 3100))
    
    # Run ablation studies
    all_loss_components = config['loss']['components']
    ablation_studies = [
        ("learning_rate", "training.learning_rate", [1e-2, 1e-3, 1e-4]),
        ("spatial_kernel_size", "model_architecture.spatial_kernel_size", [[0.4, 10], [0.6, 10], [0.8, 10]]),
        ("_type", "model_architecture._type", ["3D", "6D"]),
        ("use_geodesic", "loss.use_geodesic", [True, False]),
        ("batch_size", "training.batch_size", [5, 8, 10]),
        ("loss_components", "loss.components", [
            all_loss_components,
            [c for c in all_loss_components if c != "geo_loss"],
            [c for c in all_loss_components if c != "joint_loss"],
            [c for c in all_loss_components if c != "vert_loss"],
            [c for c in all_loss_components if c != "theta_smooth"],
        ])
    ]
    
    for ablation_name, param_name, param_values in ablation_studies:
        print(f"Running ablation study for {ablation_name}")
        run_ablation_study(config, train_loader, val_loader, test_loader, ablation_name, param_name, param_values)
    
    print("All ablation studies completed.")

if __name__ == "__main__":
    main()