import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import SRDataset, calculate_psnr, calculate_ssim
from models import SRCNN, ESPCN, VDSR, SE_ESPCN

def evaluate(model, test_loader, device):
    model.eval()
    psnr_total = 0.0
    ssim_total = 0.0
    count = 0
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(test_loader):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            outputs = model(lr_imgs)
            outputs = torch.clamp(outputs, 0.0, 1.0)
            
            hr_np = hr_imgs.permute(0, 2, 3, 1).cpu().numpy()
            output_np = outputs.permute(0, 2, 3, 1).cpu().numpy()
            
            for i in range(hr_np.shape[0]):
                psnr_total += calculate_psnr(hr_np[i], output_np[i])
                ssim_total += calculate_ssim(hr_np[i], output_np[i])
                count += 1
    
    avg_psnr = psnr_total / count
    avg_ssim = ssim_total / count
    return avg_psnr, avg_ssim

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = ['bicubic', 'srcnn', 'espcn', 'vdsr', 'se_espcn']
    test_datasets = ['Set5', 'Set14', 'BSD100', 'Urban100']
    
    os.makedirs('results', exist_ok=True)
    results = {}
    for model_name in models:
        results[model_name] = {}
        for dataset in test_datasets:
            results[model_name][dataset] = {'psnr': 0.0, 'ssim': 0.0}
    
    print('评估双三次插值...')
    for dataset in test_datasets:
        test_dataset = SRDataset(f'data/test_{dataset}_lr.npy', f'data/test_{dataset}_hr.npy', augment=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        psnr_total = 0.0
        ssim_total = 0.0
        count = 0
        
        for lr_imgs, hr_imgs in test_loader:
            bicubic = torch.nn.functional.interpolate(lr_imgs, scale_factor=4, mode='bicubic', align_corners=False)
            bicubic = torch.clamp(bicubic, 0.0, 1.0)
            
            hr_np = hr_imgs.permute(0, 2, 3, 1).numpy()
            bicubic_np = bicubic.permute(0, 2, 3, 1).numpy()
            
            for i in range(hr_np.shape[0]):
                psnr_total += calculate_psnr(hr_np[i], bicubic_np[i])
                ssim_total += calculate_ssim(hr_np[i], bicubic_np[i])
                count += 1
        
        avg_psnr = psnr_total / count
        avg_ssim = ssim_total / count
        results['bicubic'][dataset]['psnr'] = avg_psnr
        results['bicubic'][dataset]['ssim'] = avg_ssim
        print(f'{dataset} - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}')
    
    for model_name in models[1:]:
        print(f'\n评估 {model_name}...')
        model_dict = {
            'srcnn': SRCNN,
            'espcn': ESPCN,
            'vdsr': VDSR,
            'se_espcn': SE_ESPCN
        }
        model = model_dict[model_name](upscale_factor=4).to(device)
        checkpoint = torch.load(f'weights/{model_name}/best.pth')
        model.load_state_dict(checkpoint['model'])
        
        for dataset in test_datasets:
            test_dataset = SRDataset(f'data/test_{dataset}_lr.npy', f'data/test_{dataset}_hr.npy', augment=False)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            avg_psnr, avg_ssim = evaluate(model, test_loader, device)
            results[model_name][dataset]['psnr'] = avg_psnr
            results[model_name][dataset]['ssim'] = avg_ssim
            print(f'{dataset} - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}')
    
    with open('results/quantitative_results.txt', 'w') as f:
        for model_name in models:
            f.write(f'{model_name}:\n')
            for dataset in test_datasets:
                f.write(f'  {dataset}: PSNR={results[model_name][dataset]["psnr"]:.2f} dB, SSIM={results[model_name][dataset]["ssim"]:.4f}\n')
            f.write('\n')
    
    print('评估完成！结果已保存到 results/quantitative_results.txt')

if __name__ == '__main__':
    main()