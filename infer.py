import os
import paddle
import numpy as np
from PIL import Image
import argparse

from models import SE_ESPCN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='输入低分辨率图像路径')
    parser.add_argument('--output', type=str, default='output.png', help='输出高分辨率图像路径')
    parser.add_argument('--weight', type=str, default='weights/se_espcn/best.pdparams', help='模型权重路径')
    return parser.parse_args()

def main():
    args = parse_args()
    place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
    paddle.set_device(place)
    
    model = SE_ESPCN(upscale_factor=4)
    checkpoint = paddle.load(args.weight)
    model.set_state_dict(checkpoint['model'])
    model.eval()
    
    lr = Image.open(args.input).convert('RGB')
    w, h = lr.size
    lr_tensor = paddle.to_tensor(np.array(lr)).transpose([2, 0, 1]).astype('float32') / 255.0
    lr_tensor = lr_tensor.unsqueeze(0)
    
    with paddle.no_grad():
        sr = model(lr_tensor)
        sr = paddle.clip(sr, 0.0, 1.0)
    
    sr_np = sr.squeeze().transpose([1, 2, 0]).numpy()
    sr_np = (sr_np * 255).astype(np.uint8)
    sr_img = Image.fromarray(sr_np)
    sr_img.save(args.output)
    print(f'超分辨率完成！输出图像已保存到: {args.output}')

if __name__ == '__main__':
    main()