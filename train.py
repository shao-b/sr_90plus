import os
import yaml
import paddle
import logging
import argparse
from paddle.io import DataLoader
from visualdl import LogWriter
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
    writer = LogWriter(f'logs/{args.model}/visualdl')
    
    place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
    paddle.set_device(place)
    logger.info(f'使用设备: {place}')
    
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
    model = model_dict[args.model](upscale_factor=config['scale'])
    
    start_epoch = 0
    best_psnr = 0.0
    if args.resume:
        checkpoint = paddle.load(args.resume)
        model.set_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best_psnr = checkpoint['best_psnr']
        logger.info(f'从epoch {start_epoch} 继续训练')
    
    criterion = MixedLoss(mse_weight=config['mse_weight'], perceptual_weight=config['perceptual_weight'])
    optimizer = paddle.optimizer.Adam(learning_rate=config['lr'], parameters=model.parameters())
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(config['lr'], T_max=config['epochs'])
    optimizer.set_lr_scheduler(scheduler)
    
    for epoch in range(start_epoch, config['epochs']):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
        for lr_imgs, hr_imgs in pbar:
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            paddle.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.clear_grad()
            
            train_loss += loss.item() * lr_imgs.shape[0]
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_psnr = 0.0
        with paddle.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                outputs = model(lr_imgs)
                outputs = paddle.clip(outputs, 0.0, 1.0)
                
                hr_np = hr_imgs.transpose([0, 2, 3, 1]).numpy()
                output_np = outputs.transpose([0, 2, 3, 1]).numpy()
                for i in range(hr_np.shape[0]):
                    val_psnr += calculate_psnr(hr_np[i], output_np[i])
        
        val_psnr /= len(val_loader.dataset)
        
        logger.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val PSNR: {val_psnr:.2f} dB')
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('PSNR/val', val_psnr, epoch+1)
        writer.add_scalar('LR', optimizer.get_lr(), epoch+1)
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            paddle.save({
                'epoch': epoch+1,
                'model': model.state_dict(),
                'best_psnr': best_psnr,
                'optimizer': optimizer.state_dict(),
            }, f'weights/{args.model}/best.pdparams')
            logger.info(f'保存最优模型，PSNR: {best_psnr:.2f} dB')
        
        paddle.save({
            'epoch': epoch+1,
            'model': model.state_dict(),
            'best_psnr': best_psnr,
            'optimizer': optimizer.state_dict(),
        }, f'weights/{args.model}/latest.pdparams')
        
        scheduler.step()
    
    logger.info(f'训练完成！最优验证PSNR: {best_psnr:.2f} dB')
    writer.close()

if __name__ == '__main__':
    main()