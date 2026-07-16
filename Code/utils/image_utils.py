import re
from io import BytesIO
import requests
import os
import numpy as np
from PIL import Image
from Code.settings.config import PATCH, N_ROWS, N_COLS

#
# def list_images(url):
#     html = requests.get(url).text
#     names = sorted(set(re.findall(r'href="([^"?]+\.jpg)"', html)))
#     return [url + n for n in names]
#
#
# def image_to_patches(url):
#     data = requests.get(url).content
#     img = Image.open(BytesIO(data)).convert("RGB")
#     arr = np.asarray(img)
#     patches = []
#     for r in range(N_ROWS):
#         for c in range(N_COLS):
#             patch = arr[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
#             patches.append(patch)
#     return patches

def list_images(root):
    """遞迴找出 root 資料夾底下所有 .jpg,回傳本機路徑清單。"""
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith('.jpg'):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def image_to_patches(path):
    """讀本機圖檔,切成 N_ROWS x N_COLS 個 PATCH x PATCH 的格子。"""
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img)
    patches = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            patch = arr[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
            patches.append(patch)
    return patches


def load_label(image_path):
    """讀跟圖同名、同資料夾的 .csv label,攤平成 1D 整數陣列。"""
    csv_path = os.path.splitext(image_path)[0] + '.csv'
    grid = np.loadtxt(csv_path, delimiter=',')  # (N_ROWS, N_COLS)
    return grid.reshape(-1).astype(int)
