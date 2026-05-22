import numpy as np
import paddle
import math
from models import SE_ESPCN

# ===================== 核心修复 =====================
# 加载数据 + 正确加载模型权重
lr_imgs = np.load("data/val_lr.npy")
hr_imgs = np.load("data/val_hr.npy")

model = SE_ESPCN(upscale_factor=4)
checkpoint = paddle.load("weights/se_espcn/best.pdparams")
model.set_state_dict(checkpoint['model'])  # ✅ 修正函数名
model.eval()

# 推理第一张图
i = 0
lr = lr_imgs[i]
hr = hr_imgs[i]

# 数据预处理
lr_tensor = paddle.to_tensor(lr).transpose([2,0,1]).unsqueeze(0).astype('float32') / 255.0

with paddle.no_grad():
    sr = model(lr_tensor)

# 转换为图片格式
sr_img = paddle.clip(sr, 0, 1).squeeze().transpose([1,2,0]).numpy()
sr_img = (sr_img * 255).astype(np.uint8)

# ===================== 正确计算指标 =====================
# 标准PSNR/SSIM计算（像素最大值 255）
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100
    return 10 * math.log10((255.0 ** 2) / mse)

def calculate_ssim(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    ssim_val = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_val

# ===================== 输出结果 =====================
psnr_val = calculate_psnr(hr, sr_img)
ssim_val = calculate_ssim(hr, sr_img)

print("="*50)
print("📊 图像超分辨率评估结果")
print(f"PSNR: {psnr_val:.2f} dB  ")
print(f"SSIM: {ssim_val:.4f}   ")
print("="*50)