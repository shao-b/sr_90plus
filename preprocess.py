import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

HR_SIZE = config['hr_size']
SCALE = config['scale']
LR_SIZE = HR_SIZE // SCALE
TRAIN_RATIO = config['train_ratio']

def process_image(hr_path, save_lr, save_hr, is_train=True):
    hr = Image.open(hr_path).convert('RGB')
    w, h = hr.size
    
    if is_train:
        for i in range(0, h - HR_SIZE + 1, HR_SIZE):
            for j in range(0, w - HR_SIZE + 1, HR_SIZE):
                hr_patch = hr.crop((j, i, j+HR_SIZE, i+HR_SIZE))
                lr_patch = hr_patch.resize((LR_SIZE, LR_SIZE), Image.BICUBIC)
                
                hr_np = np.array(hr_patch)
                lr_np = np.array(lr_patch)
                
                save_hr.append(hr_np)
                save_lr.append(lr_np)
    else:
        w = (w // SCALE) * SCALE
        h = (h // SCALE) * SCALE
        hr = hr.crop((0, 0, w, h))
        lr = hr.resize((w//SCALE, h//SCALE), Image.BICUBIC)
        
        save_hr.append(np.array(hr))
        save_lr.append(np.array(lr))

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    
    # 处理训练集
    if not os.path.exists('data/train_lr.npy'):
        print('处理训练集...')
        train_hr = []
        train_lr = []
        train_dirs = ['data/train/DIV2K', 'data/train/Flickr2K']
        
        for dir_path in train_dirs:
            if not os.path.exists(dir_path):
                continue
            for img_name in tqdm(os.listdir(dir_path), desc=f'Processing {dir_path}'):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    process_image(os.path.join(dir_path, img_name), train_lr, train_hr, is_train=True)
        
        split_idx = int(len(train_hr) * TRAIN_RATIO)
        np.save('data/train_lr.npy', np.array(train_lr[:split_idx]))
        np.save('data/train_hr.npy', np.array(train_hr[:split_idx]))
        np.save('data/val_lr.npy', np.array(train_lr[split_idx:]))
        np.save('data/val_hr.npy', np.array(train_hr[split_idx:]))
        print(f'训练集处理完成！训练集：{split_idx}个patch，验证集：{len(train_hr)-split_idx}个patch')
    
    # 处理测试集
    test_dirs = ['data/test/Set5', 'data/test/Set14', 'data/test/BSD100', 'data/test/Urban100']
    for dir_path in test_dirs:
        if not os.path.exists(dir_path):
            continue
        dataset_name = os.path.basename(dir_path)
        if os.path.exists(f'data/test_{dataset_name}_lr.npy'):
            continue
        
        print(f'处理测试集 {dataset_name}...')
        test_hr = []
        test_lr = []
        for img_name in tqdm(os.listdir(dir_path), desc=f'Processing {dir_path}'):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                process_image(os.path.join(dir_path, img_name), test_lr, test_hr, is_train=False)
        
        np.save(f'data/test_{dataset_name}_lr.npy', np.array(test_lr))
        np.save(f'data/test_{dataset_name}_hr.npy', np.array(test_hr))
    
    print('所有数据预处理完成！')