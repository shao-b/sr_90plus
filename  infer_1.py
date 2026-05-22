import sys
sys.path.append('.')

import os
import paddle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models import SE_ESPCN

# ===================== 配置参数（只需修改这里） =====================
INPUT_IMAGE_PATH = "data/val/0801.png"  # 输入低清图片路径
OUTPUT_DIR = "results/infer"            # 输出文件夹
UPSCALE_FACTOR = 4                      # 超分倍数（和模型一致）
USE_GPU = True                          # 是否使用GPU加速
# ===================================================================

def main():
    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.basename(INPUT_IMAGE_PATH).split('.')[0]
    
    # 自动选择设备
    if USE_GPU and paddle.is_compiled_with_cuda():
        device = paddle.set_device('gpu:0')
        print(f"✅ 使用GPU加速：{device}")
    else:
        device = paddle.set_device('cpu')
        print("⚠️ 使用CPU运行")
    
    # 加载模型
    print("\n正在加载模型...")
    model = SE_ESPCN(upscale_factor=UPSCALE_FACTOR)
    checkpoint = paddle.load("weights/se_espcn/best.pdparams")
    model.set_state_dict(checkpoint['model'])
    model.eval()
    
    # 1. 读取输入低清图
    print(f"正在读取图片：{INPUT_IMAGE_PATH}")
    lr_pil = Image.open(INPUT_IMAGE_PATH).convert('RGB')
    lr_np = np.array(lr_pil)
    
    # 2. 生成双三次插值结果（传统方法）
    bicubic_pil = lr_pil.resize(
        (lr_pil.width * UPSCALE_FACTOR, lr_pil.height * UPSCALE_FACTOR),
        Image.BICUBIC
    )
    bicubic_np = np.array(bicubic_pil)
    
    # 3. 模型超分推理
    print("正在进行超分推理...")
    lr_tensor = paddle.to_tensor(lr_np).transpose([2, 0, 1]).unsqueeze(0).astype('float32') / 255.0
    
    with paddle.no_grad():
        sr_tensor = model(lr_tensor)
    
    sr_np = (paddle.clip(sr_tensor, 0, 1).squeeze().transpose([1, 2, 0]).numpy() * 255).astype(np.uint8)
    sr_pil = Image.fromarray(sr_np)
    
    # 4. 生成三列对比图
    print("正在生成对比图...")
    plt.close('all')
    plt.figure(figsize=(18, 6))
    
    # 第1列：低分辨率原图（放大显示）
    plt.subplot(1, 3, 1)
    plt.imshow(lr_np)
    plt.title(f"Low Resolution\n({lr_pil.width}×{lr_pil.height})", fontsize=14)
    plt.axis('off')
    
    # 第2列：双三次插值
    plt.subplot(1, 3, 2)
    plt.imshow(bicubic_np)
    plt.title(f"Bicubic Interpolation\n({bicubic_pil.width}×{bicubic_pil.height})", fontsize=14)
    plt.axis('off')
    
    # 第3列：SE-ESPCN超分结果
    plt.subplot(1, 3, 3)
    plt.imshow(sr_np)
    plt.title(f"SE-ESPCN Super-Resolution\n({sr_pil.width}×{sr_pil.height})", fontsize=14)
    plt.axis('off')
    
    # 保存对比图
    compare_path = os.path.join(OUTPUT_DIR, f"{filename}_compare.png")
    plt.tight_layout()
    plt.savefig(compare_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存单独的超分结果
    sr_path = os.path.join(OUTPUT_DIR, f"{filename}_sr.png")
    sr_pil.save(sr_path)
    
    # 输出结果
    print("\n" + "="*60)
    print("✅ 单张图超分完成！")
    print(f"📄 对比图路径：{compare_path}")
    print(f"🖼️  单独超分图路径：{sr_path}")
    print("="*60)

if __name__ == '__main__':
    main()