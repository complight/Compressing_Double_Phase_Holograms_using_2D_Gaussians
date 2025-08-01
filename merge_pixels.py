import os
import imageio
import shutil
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

def merge_pixel(data_name, model_name, iterations, num_points_list, patch_size, name):
    checkpoints_root = Path(f"./checkpoints/{data_name}")

    for num_points in num_points_list:
        folder = checkpoints_root / f"{model_name}_{iterations}_{num_points}"

        # extract the image length
        patch_files = [
            p for p in folder.glob(f"{name}_patch_{patch_size}_*")
            if p.stem.split("_")[3].isdigit()
        ]
        max_patch_id = max([int(p.stem.split("_")[3]) for p in patch_files])

        for base in range(1, max_patch_id + 1, 5):
            
            # initialize image id
            original_id = base
            ver1_id = base + 1
            hor1_id = base + 2
            ver2_id = base + 3
            hor2_id = base + 4

            # save the original patch
            original_path = folder / f"{name}_patch_{patch_size}_{original_id:03d}" / f"{name}_patch_{patch_size}_{original_id:03d}_fitting.png"
            if original_path.exists():
                shutil.copyfile(original_path, folder / f"{name}_{model_name}_{patch_size}_{original_id:03d}_original_fitting.png")

            # Horizontal Merge
            img_h1_path = folder / f"{name}_patch_{patch_size}_{hor1_id:03d}" / f"{name}_patch_{patch_size}_{hor1_id:03d}_fitting.png"
            img_h2_path = folder / f"{name}_patch_{patch_size}_{hor2_id:03d}" / f"{name}_patch_{patch_size}_{hor2_id:03d}_fitting.png"
            if img_h1_path.exists() and img_h2_path.exists():
                img_h1 = np.array(Image.open(img_h1_path))
                img_h2 = np.array(Image.open(img_h2_path))
                H, W, C = img_h1.shape
                merged_h = np.zeros((H, W * 2, C), dtype=img_h1.dtype)

                dark1 = img_h1[0::2, :, :] 
                dark2 = img_h1[1::2, :, :]
                bright1 = img_h2[0::2, :, :] 
                bright2 = img_h2[1::2, :, :]

                merged_h[0::2, 0::2, :] = bright1  # even rows, even cols
                merged_h[1::2, 1::2, :] = bright2  # odd rows, odd cols
                merged_h[0::2, 1::2, :] = dark1  # even rows, odd cols
                merged_h[1::2, 0::2, :] = dark2  # odd rows, even cols

                filename = f"{name}_{model_name}_{patch_size}_{original_id:03d}_horizontal_fitting.png"
                merged_h = (merged_h).astype(np.uint8)
                imageio.imwrite(os.path.join(folder, filename), merged_h)
            
            # Vertical Merge
            img_v1_path = folder / f"{name}_patch_{patch_size}_{ver1_id:03d}" / f"{name}_patch_{patch_size}_{ver1_id:03d}_fitting.png"
            img_v2_path = folder / f"{name}_patch_{patch_size}_{ver2_id:03d}" / f"{name}_patch_{patch_size}_{ver2_id:03d}_fitting.png"
            if img_v1_path.exists() and img_v2_path.exists():
                img_v1 = np.array(Image.open(img_v1_path))
                img_v2 = np.array(Image.open(img_v2_path))
                H, W, C = img_v1.shape
                merged_v = np.zeros((H * 2, W, C), dtype=img_v1.dtype)

                dark1 = img_v1[:, 0::2, :] 
                dark2 = img_v1[:, 1::2, :]
                bright1 = img_v2[:, 0::2, :] 
                bright2 = img_v2[:, 1::2, :]

                merged_v[0::2, 0::2, :] = bright1  # even rows, even cols
                merged_v[1::2, 1::2, :] = bright2  # odd rows, odd cols
                merged_v[0::2, 1::2, :] = dark1  # even rows, odd cols
                merged_v[1::2, 0::2, :] = dark2  # odd rows, even cols

                filename = f"{name}_{model_name}_{patch_size}_{original_id:03d}_vertical_fitting.png"
                merged_v = (merged_v).astype(np.uint8)
                imageio.imwrite(os.path.join(folder, filename), merged_v)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--num_points_list", nargs='+', required=True, type=int)
    parser.add_argument("--patchsize", required=True, type=int)
    parser.add_argument("--name", required=True, type=str)
    args = parser.parse_args()

    merge_pixel(args.data_name, args.model_name, args.iterations, args.num_points_list, args.patchsize, args.name)

if __name__ == "__main__":
    main()
