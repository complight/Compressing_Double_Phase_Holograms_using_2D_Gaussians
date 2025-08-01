import os
import re
import math
import torch
import argparse
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from pytorch_msssim import ssim

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def eval_merged(data_name, model_name, iterations, num_points_list, patch_size, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = T.ToTensor()
    
    for num_points in num_points_list:
        merged_dir = f"./checkpoints/{data_name}/{model_name}_{iterations}_{num_points}"
        
        eval_path = os.path.join(merged_dir, f"{name}_eval_{patch_size}.txt")
        with open(eval_path, "w") as f:
            f.write("filename: HxW, PSNR, SSIM\n")

        for filename in os.listdir(merged_dir):
            if not filename.endswith(".png"):
                continue
            # iterate all compressed imgs
            file_path = os.path.join(merged_dir, filename)
            merged_img = Image.open(file_path)

            match = re.match(rf"{name}_{model_name}_{patch_size}_(\d+)", filename)
            if match:
                patch_id = match.group(1)
                ori_img = Image.open(f"./dataset/{data_name}/{name}_patch_{patch_size}_{patch_id}.png")
            else:
                continue

            # convert to tensors
            merged_tensor = to_tensor(merged_img).unsqueeze(0).to(device)  # shape: [1,3,H,W]
            ori_tensor = to_tensor(ori_img).unsqueeze(0).to(device)

            # ensure shape match
            if merged_tensor.shape != ori_tensor.shape:
                print(f"Shape mismatch in {filename}, skipping.")
                continue

            # 1. MSE
            mse_loss = F.mse_loss(merged_tensor, ori_tensor)

            # 2. PSNR
            psnr = 10 * math.log10(1.0 / mse_loss.item())

            # 3. MS-SSIM
            ssim_val = ssim(merged_tensor, ori_tensor, data_range=1, size_average=True).item()

            # log results
            image_name = filename.replace(".png", "")
            H, W = ori_img.size[1], ori_img.size[0]  # PIL: (W, H)
            line = "{}: {}x{}, PSNR:{:.4f}, SSIM:{:.4f}".format(
                image_name, H, W, psnr, ssim_val)
            print(line)

            with open(eval_path, "a") as f:
                f.write(line + "\n")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--num_points_list", nargs='+', required=True, type=int)
    parser.add_argument("--patchsize", required=True, type=int)
    parser.add_argument("--name", required=True, type=str)
    args = parser.parse_args()

    eval_merged(args.data_name, args.model_name, args.iterations, args.num_points_list, args.patchsize, args.name)

if __name__ == "__main__":
    main()
