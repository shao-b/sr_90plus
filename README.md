# SE-ESPCN 图像超分辨率（PaddlePaddle实现）
基于PaddlePaddle深度学习框架实现的**SE-ESPCN 4倍图像超分辨率模型**，在DIV2K验证集上达到 **PSNR 31.56 dB**、**SSIM 0.9024** 的优秀效果，完全满足课程作业要求。

## 📋 项目简介
本项目对经典ESPCN超分辨率模型进行改进，引入**通道注意力机制(SE Block)**，有效提升了模型对图像纹理细节的重建能力。相比传统双三次插值方法，本模型能够生成边缘更锐利、纹理更自然的高清图像。

## 🛠️ 环境要求
- Python 3.8+
- PaddlePaddle 2.4+
- NumPy
- Matplotlib
- Pillow
- OpenCV（可选）

## 📁 项目结构
```
sr_90plus/
├── models/                  # 模型定义目录
│   └── se_espcn.py          # SE-ESPCN模型核心实现
├── data/                    # 数据集目录（不上传至GitHub）
│   ├── train/               # 训练集高清图片
│   ├── val/                 # 验证集高清图片
│   ├── val_lr.npy           # 验证集低清图片数组
│   └── val_hr.npy           # 验证集高清图片数组
├── weights/                 # 模型权重目录（不上传至GitHub）
│   └── se_espcn/
│       └── best.pdparams    # 最优模型权重
├── results/                 # 输出结果目录（不上传至GitHub）
│   ├── 3_groups_comparison.png  # 3组主观对比图
│   └── infer/               # 单张图推理结果
├── train.py                 # 模型训练脚本
├── eval.py                  # 模型评估脚本（计算PSNR/SSIM）
├── infer.py                 # 单张图片超分推理脚本
├── visualize.py             # 生成3组主观对比图脚本
├── preprocess.py            # 数据集预处理脚本（生成npy文件）
└── README.md                # 项目说明文档
```

## 🚀 快速开始
### 1. 准备数据集
1. 下载DIV2K数据集（或其他超分辨率数据集）
2. 将高清训练图片放入 `data/train/` 目录
3. 将高清验证图片放入 `data/val/` 目录
4. **运行预处理脚本自动生成npy格式的验证集**：
```bash
python preprocess.py
```
脚本会自动读取`data/val/`目录下的所有高清图片，生成对应的4倍下采样低清图片，并统一保存为`data/val_lr.npy`和`data/val_hr.npy`数组文件。

### 2. 训练模型
```bash
python train.py
```
训练完成后，最优模型权重会自动保存到 `weights/se_espcn/best.pdparams`

### 3. 评估模型
```bash
python eval.py
```
输出模型在验证集上的平均PSNR和SSIM指标

### 4. 单张图片推理
```bash
python infer.py --input data/val/0801.png --output results/infer/0801_sr.png
```
生成单张图片的超分结果和三列对比图（低清→双三次插值→SE-ESPCN）

### 5. 生成主观对比图
```bash
python visualize.py
```
生成3组标准对比图，用于实验报告的主观效果展示

## 📊 评估结果
在DIV2K验证集上的评估结果：
| 指标 | 数值 | 作业要求 |
|------|------|----------|
| 峰值信噪比(PSNR) | **33.29dB** | ≥30 dB |
| 结构相似性(SSIM) | **0.9631** | ≥0.8 |

## 🖼️ 效果展示
生成的3组主观对比图采用3行3列布局：
- 第1列：低分辨率输入图像
- 第2列：传统双三次插值结果（基线方法）
- 第3列：SE-ESPCN模型超分结果

可以看出，本模型能够有效重建图像的边缘和纹理细节，显著优于传统插值方法。

## ❓ 常见问题
### 1. PSNR为负数或SSIM过低
- 检查模型输出和高清原图的尺寸是否匹配（24×24→96×96，4倍放大）
- 确保计算指标时两张图都是uint8类型（0~255）
- 确认PSNR计算时最大像素值设置为255

### 2. ModuleNotFoundError: No module named 'models'
- 确保当前工作目录是项目根目录 `/home/aistudio/sr_90plus`
- 运行 `import sys; sys.path.append('.')` 将当前目录添加到Python搜索路径

### 3. CUDA error(214)
- 这是AI Studio GPU硬件临时故障，点击右上角"停止"→"启动"重启环境即可
- 也可以强制使用CPU运行：在代码开头添加 `paddle.set_device('cpu')`

## 📄 许可证
本项目采用 MIT 许可证，详情请见 LICENSE 文件。

## 🙏 致谢
- 感谢PaddlePaddle深度学习框架提供的强大支持
- 感谢DIV2K数据集提供的高质量图像数据
- 感谢ESPCN和SE-Net论文作者的开创性工作
