import os
import time
from PIL import Image
from glob import glob

import torch
import torch.nn as nn
from torchvision import transforms
from CWNet import CWNet

# =========================
# 1. 你只需要改这里：模型
# =========================
def load_model(device):
    """
    TODO: 替换成你的模型
    """
    model = CWNet()  # 占位模型（直接返回输入）

    model = model.to(device)
    model.eval()
    return model


# =========================
# 2. 图像预处理
# =========================
def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # [0,1]
    ])


# =========================
# 3. 读取图像
# =========================
def load_images(image_dir):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    paths = []
    for e in exts:
        paths.extend(glob(os.path.join(image_dir, e)))

    paths = sorted(paths)
    if len(paths) == 0:
        raise ValueError(f"未找到图像: {image_dir}")

    return paths


def load_image(path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)


# =========================
# 4. 主测速函数
# =========================
def benchmark(model, image_paths, device="cuda", warmup=10):
    transform = build_transform()

    total_time = 0.0
    total_images = 0

    model.eval()

    # warmup
    print(f"Warmup {warmup} iters...")
    with torch.no_grad():
        for i in range(min(warmup, len(image_paths))):
            img = load_image(image_paths[i], transform).unsqueeze(0).to(device)
            _ = model(img)

    print("开始正式测速...")

    with torch.no_grad():
        for i, path in enumerate(image_paths):

            img = load_image(path, transform).unsqueeze(0).to(device)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.time()

            _ = model(img)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.time()

            total_time += (end - start)
            total_images += 1

            if (i + 1) % 50 == 0:
                print(f"已处理 {i+1}/{len(image_paths)}")

    avg_time = total_time / total_images
    fps = total_images / total_time

    print("\n========== 结果 ==========")
    print(f"总图片数: {total_images}")
    print(f"平均延迟: {avg_time * 1000:.4f} ms / image")
    print(f"吞吐量: {fps:.2f} FPS")


# =========================
# 5. main
# =========================
if __name__ == "__main__":

    # ====== 修改这里 ======
    image_dir = "/data2/zhengkaihang/ttt/dataset/LOLv1/Test/input"  # 你的退化图像目录

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = load_model(device)

    print("Loading images...")
    image_paths = load_images(image_dir)

    print(f"Found {len(image_paths)} images")

    benchmark(model, image_paths, device=device)