import os
import yaml
import torch
import logging
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import SRDataset, MixedLoss, calculate_psnr
from models import SRCNN, ESPCN, VDSR, SE_ESPCN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    parser.add_argument('--model', type=str, default='se_espcn', choices=['srcnn', 'espcn', 'vdsr', 'se_espcn'], help='模型名称')
    parser.add_argument('--resume', type=str, default=None, help='断点续训权重路径')
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    os.makedirs(f'logs/{args.model}', exist_ok=True)
    os.makedirs(f'weights/{args.model}', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/{args.model}/train.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    writer = SummaryWriter(f'logs/{args.model}/tensorboard')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    train_dataset = SRDataset('data/train_lr.npy', 'data/train_hr.npy', augment=True)
    val_dataset = SRDataset('data/val_lr.npy', 'data/val_hr.npy', augment=False)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    model_dict = {
        'srcnn': SRCNN,
        'espcn': ESPCN,
        'vdsr': VDSR,
        'se_espcn': SE_ESPCN
    }
    model = model_dict[args.model](upscale_factor=config['scale']).to(device)
    
    start_epoch = 0
    best_psnr = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best_psnr = checkpoint['best_psnr']
        logger.info(f'从epoch {start_epoch} 继续训练')
    
    criterion = MixedLoss(device, mse_weight=config['mse_weight'], perceptual_weight=config['perceptual_weight'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    for epoch in range(start_epoch, config['epochs']):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
        for lr_imgs, hr_imgs in pbar:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * lr_imgs.size(0)
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                outputs = model(lr_imgs)
                outputs = torch.clamp(outputs, 0.0, 1.0)
                
                hr_np = hr_imgs.permute(0, 2, 3, 1).cpu().numpy()
                output_np = outputs.permute(0, 2, 3, 1).cpu().numpy()
                for i in range(hr_np.shape[0]):
                    val_psnr += calculate_psnr(hr_np[i], output_np[i])
        
        val_psnr /= len(val_loader.dataset)
        
        logger.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val PSNR: {val_psnr:.2f} dB')
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('PSNR/val', val_psnr, epoch+1)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch+1)
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch+1,
                'model': model.state_dict(),
                'best_psnr': best_psnr,
                'optimizer': optimizer.state_dict(),
            }, f'weights/{args.model}/best.pth')
            logger.info(f'保存最优模型，PSNR: {best_psnr:.2f} dB')
        
        torch.save({
            'epoch': epoch+1,
            'model': model.state_dict(),
            'best_psnr': best_psnr,
            'optimizer': optimizer.state_dict(),
        }, f'weights/{args.model}/latest.pth')
        
        scheduler.step()
    
    logger.info(f'训练完成！最优验证PSNR: {best_psnr:.2f} dB')
    writer.close()

if __name__ == '__main__':
    main()