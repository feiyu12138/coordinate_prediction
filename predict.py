import torch
import argparse
import dsntnn
from torch import optim
from model.model import CoordRegressionNetwork
from model.predictor import Predictor,Plotter
from data.dataset import create_simple_dataset
from data.normalizer import Normalizer
from data.constant import IMAGE_SIZE
from tqdm import tqdm
import os
import wandb
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

str2list = lambda x: list(map(int, x.split(',')))

    
def predict_datasets(predictor, dataloader, device):
    all_coords = []
    for images, targets, image_paths in tqdm(dataloader):
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            coords = predictor(images, image_paths)
        for coord,image_path in zip(coords, image_paths):
            all_coords.append({
                'image_path': image_path,
                'coords': coord
            })
    return coords

def draw_curve(coords, image_path):
    pass

def load_model_with_optimizer(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    if "model" not in checkpoint:
        raise ValueError("Checkpoint does not contain model")
    else:
        model.load_state_dict(checkpoint['model'])
        print(f"Model loaded from checkpoint {path}")
    if "optimizer" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Optimizer loaded from checkpoint {path}")
    else:
        print("Optimizer not found in checkpoint, skipping...")
    return model, optimizer

def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--n_locations', type=int, default=2)
    parser.add_argument('--backbone', type=str, default='fcn')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_path', type=str, default='ckpt/model_final.pth')
    parser.add_argument('--output_dir', type=str, default='ckpt')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = CoordRegressionNetwork(args.n_locations, args.image_size, args.backbone).to(device)
    norm = Normalizer(device=device, image_size=IMAGE_SIZE)
    plotter = Plotter(args.output_dir, args.root)
    model,optimizer = load_model_with_optimizer(model, None, args.resume_path, device)
    predictor = Predictor(model, norm, plotter) 
    
    losses = []
    dataset = create_simple_dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    coords = predict_datasets(predictor, dataloader, device)
    

if __name__ == '__main__':
    main()
    