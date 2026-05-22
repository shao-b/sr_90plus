import os
import paddle
import numpy as np
import matplotlib.pyplot as plt
from models import SE_ESPCN

# 核心函数：直接从npy数组画图，不读取任何图片文件
def main():
    # 1. 加载模型
    model = SE_ESPCN(upscale_factor=4)
    checkpoint = paddle.load('weights/se_espcn/best.pdparams')
    model.set_state_dict(checkpoint['model'])
    model.eval()

    # 2. 直接加载你的验证集npy（唯一路径，绝对存在！）
    lr_imgs = np.load("data/val_lr.npy")
    hr_imgs = np.load("data/val_hr.npy")
    
    # 取第一张图
    lr = lr_imgs[0:1]  # 模型输入
    hr = hr_imgs[0]

    # 3. 模型推理
    with paddle.no_grad():
        sr = model(paddle.to_tensor(lr))
    
    # 4. 转换为图片格式
    lr_img = lr[0].transpose(1,2,0)
    sr_img = paddle.clip(sr, 0, 1).numpy()[0].transpose(1,2,0)

    # 5. 画图保存
    plt.figure(figsize=(15,5))
    plt.subplot(131), plt.imshow(lr_img), plt.title('低分辨率'), plt.axis('off')
    plt.subplot(132), plt.imshow(sr_img), plt.title('SE-ESPCN超分结果'), plt.axis('off')
    plt.subplot(133), plt.imshow(hr), plt.title('高清原图'), plt.axis('off')
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/result.png', dpi=300)
    plt.close()
    print("✅ 效果图生成成功！路径：results/result1.png")

if __name__ == '__main__':
    main()