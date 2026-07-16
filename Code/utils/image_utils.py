import re
from io import BytesIO
import requests
import numpy as np
from PIL import Image
from Code.settings.config import PATCH, N_ROWS, N_COLS


def list_images(url):
    html = requests.get(url).text
    names = sorted(set(re.findall(r'href="([^"?]+\.jpg)"', html)))
    return [url + n for n in names]


def image_to_patches(url):
    data = requests.get(url).content
    img = Image.open(BytesIO(data)).convert("RGB")
    arr = np.asarray(img)
    patches = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            patch = arr[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
            patches.append(patch)
    return patches