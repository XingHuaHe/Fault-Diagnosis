# %%
#  Systerm packages.
import os
import sys
import argparse
# Externed packages.
import tqdm
import numpy as np
import scipy.io as scio
from PIL import Image

# %%
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./datas/Crack Datas/signals.mat", help="signals source directory")
    parser.add_argument("--target", type=str, default="./datas/Crack Datas Images/", help="images target directory")
    opt = parser.parse_args()

    # directory.
    os.makedirs(opt.target, exist_ok=True)
    signals = scio.loadmat(opt.source)
    signals = signals['signals']
    count, dim = signals.shape
    for i in tqdm.tqdm(range(count)):
        img = np.zeros((3, dim, dim))
        signal = signals[i] + abs(min(signals[i]))
        for j in range(dim):
            img[0, j, :] = (signal / (max(signal) - min(signal))) * 255
        img[1, :, :] = img[0, :, :]
        img[2, :, :] = img[0, :, :]
        img = np.transpose(img, (1,2,0))
        img = Image.fromarray(np.uint8(img))
        img.save(f'{opt.target}{i}.jpg')

# %%
