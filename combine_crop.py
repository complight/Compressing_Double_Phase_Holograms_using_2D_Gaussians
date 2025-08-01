import re
import math
import torch
import imageio
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
import torchvision.transforms as T
from pytorch_msssim import ssim

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

def load_patch_images(folder, suffix, patch_size, name, model_name):
    # extract all files with the same suffix
    patch_files = sorted(
    folder.glob(f"{name}_{model_name}_{patch_size}_*_{suffix}.png"),
    key=lambda p: int(re.search(rf"{patch_size}_(\d+)_", p.stem).group(1)))
    images = [np.array(Image.open(p)) for p in patch_files]
    return images, patch_files

def eval(combined, original, output_dir, image_name, patch_size, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = T.ToTensor()
    
    log_dir = output_dir.parent / f"{name}_eval_{patch_size}.txt"

    # convert to tensors
    merged_tensor = to_tensor(Image.fromarray(combined)).unsqueeze(0).to(device)  # shape: [1,3,H,W]
    ori_tensor = to_tensor(original).unsqueeze(0).to(device)

    # 1. MSE
    mse_loss = F.mse_loss(merged_tensor, ori_tensor)

    # 2. PSNR
    psnr = 10 * math.log10(1.0 / mse_loss.item())

    # 3. MS-SSIM
    ssim_val = ssim(merged_tensor, ori_tensor, data_range=1, size_average=True).item()

    # log results
    W, H = original.size  
    line = "{}: {}x{}, PSNR:{:.4f}, SSIM:{:.4f}".format(
        f"{patch_size}_{image_name}", H, W, psnr, ssim_val)
    print(line)

    with open(log_dir, "a") as f:
        f.write(line + "\n")
    return

def combine_images_numpy(images, w_total, sample_path, output_dir, suffix, patch_size, name):
    rows = []
    current_row = []
    current_width = 0
    max_height = 0

    for img in images:
        h, w = img.shape[:2]
        # the last crop in the current row
        if current_width + w > w_total:
            rows.append((current_row, max_height))
            current_row = []
            current_width = 0
            max_height = 0
        # list crops in the current row
        current_row.append(img)
        current_width += w
        max_height = max(max_height, h)
    
    # in case the last row is not fully filled
    if current_row:
        rows.append((current_row, max_height))

    total_height = sum(height for _, height in rows)
    canvas = np.zeros((total_height, w_total, 3), dtype=np.uint8)

    # combine crops at pixel level
    y = 0
    for row_imgs, row_h in rows:
        x = 0
        for img in row_imgs:
            h, w = img.shape[:2]
            canvas[y:y+h, x:x+w, :3] = img[:, :, :3]
            x += w
        y += row_h

    cropped = Image.open(sample_path/ f"{name}_crop_{patch_size}.png")
    
    # evaluate the combined patches
    eval(canvas, cropped, output_dir, suffix, patch_size, name)
    
    return canvas

def save_combined_images(folder, output_dir, w_total, sample_path, patch_size, name, model_name):
    output_dir.mkdir(parents=True, exist_ok=True)

    for suffix in ["original_fitting", "horizontal_fitting", "vertical_fitting"]:
        images, _ = load_patch_images(folder, suffix, patch_size, name, model_name)
        if not images:
            print(f"No images found for {suffix}")
            continue

        combined = combine_images_numpy(images, w_total, sample_path, output_dir, suffix, patch_size, name)
        out_path = output_dir / f"{name}_{model_name}_{patch_size}_combined_{suffix}.png"
        imageio.imwrite(out_path, combined, compress_level=0)
        print(f"Saved combined image to {out_path}")

def combine(data_name, model_name, iterations, num_points_list, patch_size, name):
    target_width = patch_size*2
    checkpoints_root = Path(f"./checkpoints/{data_name}")
    sample_path = Path(f"./dataset/{data_name}")

    for num_points in num_points_list:
        folder = checkpoints_root / f"{model_name}_{iterations}_{num_points}"
        output_dir = folder / "results"
        save_combined_images(folder, output_dir, target_width, sample_path, patch_size, name, model_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--num_points_list", nargs='+', required=True, type=int)
    parser.add_argument("--patchsize", required=True, type=int)
    parser.add_argument("--name", required=True, type=str)
    args = parser.parse_args()

    combine(args.data_name, args.model_name, args.iterations, args.num_points_list, args.patchsize, args.name)

if __name__ == "__main__":
    main()
