import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

# 加载配置
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

HR_SIZE = config['hr_size']
SCALE = config['scale']
LR_SIZE = HR_SIZE // SCALE

def process_image(hr_path, save_lr, save_hr, is_train=True):
    """
    处理单张高清图像，生成对应的低分辨率patch
    is_train=True: 裁剪为无重叠的HR_SIZE×HR_SIZE patch
    is_train=False: 调整为4的整数倍尺寸，用于测试集全图评估
    """
    hr = Image.open(hr_path).convert('RGB')
    w, h = hr.size
    
    if is_train:
        # 训练/验证集：裁剪为固定大小的patch
        for i in range(0, h - HR_SIZE + 1, HR_SIZE):
            for j in range(0, w - HR_SIZE + 1, HR_SIZE):
                hr_patch = hr.crop((j, i, j+HR_SIZE, i+HR_SIZE))
                lr_patch = hr_patch.resize((LR_SIZE, LR_SIZE), Image.BICUBIC)
                
                hr_np = np.array(hr_patch)
                lr_np = np.array(lr_patch)
                
                save_hr.append(hr_np)
                save_lr.append(lr_np)
    else:
        # 测试集：调整为4的整数倍，保留全图
        w = (w // SCALE) * SCALE
        h = (h // SCALE) * SCALE
        hr = hr.crop((0, 0, w, h))
        lr = hr.resize((w//SCALE, h//SCALE), Image.BICUBIC)
        
        save_hr.append(np.array(hr))
        save_lr.append(np.array(lr))

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    
    # ====================== 1. 处理训练集（DIV2K+Flickr2K 全部作为训练） ======================
    if not os.path.exists('data/train_lr.npy'):
        print('正在处理训练集...')
        train_hr = []
        train_lr = []
        train_dirs = ['data/train/DIV2K', 'data/train/Flickr2K']
        
        for dir_path in train_dirs:
            if not os.path.exists(dir_path):
                print(f'警告：目录 {dir_path} 不存在，跳过')
                continue
            for img_name in tqdm(os.listdir(dir_path), desc=f'处理 {dir_path}'):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    process_image(os.path.join(dir_path, img_name), train_lr, train_hr, is_train=True)
        
        # 保存全部训练集（不再划分验证集）
        np.save('data/train_lr.npy', np.array(train_lr))
        np.save('data/train_hr.npy', np.array(train_hr))
        print(f'训练集处理完成！共生成 {len(train_hr)} 个训练patch')
    
    # ====================== 2. 处理官方验证集（DIV2K_valid_HR） ======================
    if not os.path.exists('data/val_lr.npy'):
        print('\n正在处理官方验证集...')
        val_hr = []
        val_lr = []
        val_dir = 'data/val/DIV2K'
        
        if os.path.exists(val_dir):
            for img_name in tqdm(os.listdir(val_dir), desc='处理验证集'):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    process_image(os.path.join(val_dir, img_name), val_lr, val_hr, is_train=True)
            
            np.save('data/val_lr.npy', np.array(val_lr))
            np.save('data/val_hr.npy', np.array(val_hr))
            print(f'验证集处理完成！共生成 {len(val_hr)} 个验证patch')
        else:
            print('警告：官方验证集目录不存在，将使用训练集的10%作为验证集')
    
    # ====================== 3. 处理测试集（Set5/Set14/BSD100/Urban100） ======================
    test_dirs = ['data/test/Set5', 'data/test/Set14', 'data/test/BSD100', 'data/test/Urban100']
    for dir_path in test_dirs:
        if not os.path.exists(dir_path):
            continue
        dataset_name = os.path.basename(dir_path)
        if os.path.exists(f'data/test_{dataset_name}_lr.npy'):
            continue
        
        print(f'\n正在处理测试集 {dataset_name}...')
        test_hr = []
        test_lr = []
        for img_name in tqdm(os.listdir(dir_path), desc=f'处理 {dir_path}'):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                process_image(os.path.join(dir_path, img_name), test_lr, test_hr, is_train=False)
        
        np.save(f'data/test_{dataset_name}_lr.npy', np.array(test_lr))
        np.save(f'data/test_{dataset_name}_hr.npy', np.array(test_hr))
    
    print('\n✅ 所有数据预处理完成！')