import torch
import argparse
import dsntnn
from torch import optim
from model.model import CoordRegressionNetwork
from data.dataset import create_dataset
from data.normalizer import Normalizer
from data.constant import IMAGE_SIZE
from tqdm import tqdm
import os
import wandb
from torch.optim.lr_scheduler import StepLR

str2list = lambda x: list(map(int, x.split(',')))

def train_epoch(model, optimizer, train_loader, norm, device):
    total_loss = 0
    model.train()
    for images, targets in tqdm(train_loader):
        images, targets = images.to(device), targets.to(device)
        targets = norm.normalize(targets)
        optimizer.zero_grad()
        coords, heatmaps = model(images)
        euc_losses = dsntnn.euclidean_losses(coords, targets)
        reg_losses = dsntnn.js_reg_losses(heatmaps, targets, sigma_t=1.0)
        loss = dsntnn.average_loss(euc_losses + reg_losses)
        tqdm.write(f'loss: {loss.item()}')
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader)

def eval_epoch(model, val_loader, norm, device):
    model.eval()
    total_loss = 0
    for images, targets in tqdm(val_loader):
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            targets = norm.normalize(targets)
            coords, heatmaps = model(images)
            euc_losses = dsntnn.euclidean_losses(coords, targets)
            reg_losses = dsntnn.js_reg_losses(heatmaps, targets, sigma_t=1.0)
            loss = dsntnn.average_loss(euc_losses + reg_losses)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def plot_losses(losses):
    import matplotlib.pyplot as plt
    train_losses = [loss['train_loss'] for loss in losses]
    eval_losses = [loss['eval_loss'] for loss in losses]
    plt.plot(train_losses, label='train loss')
    plt.plot(eval_losses, label='eval loss')
    plt.legend()
    plt.savefig('losses.png')
    
def save_model_with_optimizer(model, optimizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)

def load_model_with_optimizer(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    if "model" not in checkpoint:
        raise ValueError("Checkpoint does not contain model")
    else:
        model.load_state_dict(checkpoint['model'])
        print(f"Model loaded from checkpoint {path}")
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Optimizer loaded from checkpoint {path}")
    else:
        print("Optimizer not found in checkpoint, skipping...")
    return model, optimizer

def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--n_locations', type=int, default=2)
    parser.add_argument('--backbone', type=str, default='fcn')
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_path', type=str, default='ckpt/model_final.pth')
    parser.add_argument('--save_dir', type=str, default='ckpt')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--project', type=str, default='my_project')
    parser.add_argument('--entity', type=str, default='my_entity')
    args = parser.parse_args()
    
    wandb.init(project=args.project, entity=args.entity, config=args)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_set, val_set = create_dataset(args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
    
    model = CoordRegressionNetwork(args.n_locations, args.image_size, args.backbone).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    if args.resume:
        model,optimizer = load_model_with_optimizer(model, optimizer, args.resume_path,device)
    
    norm = Normalizer(device=device, image_size=IMAGE_SIZE)
    losses = []
    for epoch in range(args.n_epochs):
        if args.do_train:
            train_loss = train_epoch(model, optimizer, train_loader, norm, device)
            print(f'Epoch {epoch} done, train loss: {train_loss}')
            wandb.log({"train_loss": train_loss, "epoch": epoch, "lr": scheduler.get_last_lr()[0]})
        if args.save_dir and epoch % args.save_interval == 0:
            save_model_with_optimizer(model, optimizer, f'{args.save_dir}/model_{epoch}.pth')
        if args.do_eval and epoch % args.eval_interval == 0:
            eval_loss = eval_epoch(model, val_loader, norm, device)
            print(f'eval loss: {eval_loss}')
            wandb.log({"eval_loss": eval_loss, "epoch": epoch})
        losses.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss
        }) 
        scheduler.step()
    plot_losses(losses)
    torch.save(model.state_dict(), f'{args.save_dir}/model_final.pth')

if __name__ == '__main__':
    main()
    