import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(hr, sr, data_range=1.0):
    hr_y = 0.299*hr[...,0] + 0.587*hr[...,1] + 0.114*hr[...,2]
    sr_y = 0.299*sr[...,0] + 0.587*sr[...,1] + 0.114*sr[...,2]
    return peak_signal_noise_ratio(hr_y, sr_y, data_range=data_range)

def calculate_ssim(hr, sr, data_range=1.0):
    hr_y = 0.299*hr[...,0] + 0.587*hr[...,1] + 0.114*hr[...,2]
    sr_y = 0.299*sr[...,0] + 0.587*sr[...,1] + 0.114*sr[...,2]
    return structural_similarity(hr_y, sr_y, data_range=data_range)