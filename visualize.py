import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models import SRCNN, ESPCN, VDSR, SE_ESPCN

def visualize_single_image(img_path, models, device, save_path):
    hr = Image.open(img_path).convert('RGB')
    w, h = hr.size
    w = (w // 4) * 4
    h = (h // 4) * 4
    hr = hr.crop((0, 0, w, h))
    lr = hr.resize((w//4, h//4), Image.BICUBIC)
    
    lr_tensor = torch.from_numpy(np.array(lr)).permute(2, 0, 1).float() / 255.0
    lr_tensor = lr_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        bicubic = torch.nn.functional.interpolate(lr_tensor, scale_factor=4, mode='bicubic', align_corners=False)
        srcnn_out = models['srcnn'](lr_tensor)
        espcn_out = models['espcn'](lr_tensor)
        vdsr_out = models['vdsr'](lr_tensor)
        se_espcn_out = models['se_espcn'](lr_tensor)
    
    def tensor2img(tensor):
        img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0.0, 1.0)
        return (img * 255).astype(np.uint8)
    
    bicubic_img = tensor2img(bicubic)
    srcnn_img = tensor2img(srcnn_out)
    espcn_img = tensor2img(espcn_out)
    vdsr_img = tensor2img(vdsr_out)
    se_espcn_img = tensor2img(se_espcn_out)
    hr_img = np.array(hr)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0,0].imshow(lr.resize((w, h), Image.NEAREST))
    axes[0,0].set_title('Low Resolution (x4 Upsampled)', fontsize=14)
    axes[0,1].imshow(bicubic_img)
    axes[0,1].set_title('Bicubic', fontsize=14)
    axes[0,2].imshow(srcnn_img)
    axes[0,2].set_title('SRCNN', fontsize=14)
    axes[1,0].imshow(espcn_img)
    axes[1,0].set_title('ESPCN', fontsize=14)
    axes[1,1].imshow(vdsr_img)
    axes[1,1].set_title('VDSR', fontsize=14)
    axes[1,2].imshow(se_espcn_img)
    axes[1,2].set_title('Ours (SE-ESPCN)', fontsize=14)
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    crop_x, crop_y, crop_size = min(100, w//4), min(100, h//4), min(100, w//4)
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    imgs = [lr.resize((w, h), Image.NEAREST), bicubic_img, srcnn_img, espcn_img, vdsr_img, se_espcn_img]
    titles = ['LR', 'Bicubic', 'SRCNN', 'ESPCN', 'VDSR', 'Ours']
    
    for i, (img, title) in enumerate(zip(imgs, titles)):
        crop = img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
        axes[i].imshow(crop)
        axes[i].set_title(title, fontsize=14)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_crop.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {
        'srcnn': SRCNN(upscale_factor=4).to(device),
        'espcn': ESPCN(upscale_factor=4).to(device),
        'vdsr': VDSR(upscale_factor=4).to(device),
        'se_espcn': SE_ESPCN(upscale_factor=4).to(device)
    }
    
    for name in models:
        checkpoint = torch.load(f'weights/{name}/best.pth')
        models[name].load_state_dict(checkpoint['model'])
        models[name].eval()
    
    test_images = [
        'data/test/Set5/butterfly.png',
        'data/test/Set14/barbara.png',
        'data/test/BSD100/3096.jpg',
        'data/test/Urban100/img_001.png',
        'data/test/Urban100/img_072.png'
    ]
    
    os.makedirs('results/visualization', exist_ok=True)
    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        img_name = os.path.basename(img_path)
        save_path = f'results/visualization/{img_name}'
        visualize_single_image(img_path, models, device, save_path)
        print(f'生成对比图: {save_path}')

if __name__ == '__main__':
    main()