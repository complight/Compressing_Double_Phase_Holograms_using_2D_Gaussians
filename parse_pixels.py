import argparse
import numpy as np
import os
import imageio
from PIL import Image

def parse_pixel(data_name, patch_id, patch_size, name):

    patch_id = int(patch_id)
    hologram_path = f"dataset/{data_name}/{name}_patch_{patch_size}_{patch_id:03d}.png"
    phase_only = np.array(Image.open(hologram_path))

    H, W, C = phase_only.shape
    assert H % 2 == 0 and W % 2 == 0, "Image dimensions must be even"

    # parse Checkerboard
    bright1 = phase_only[0::2, 0::2]  # even rows, even cols
    bright2 = phase_only[1::2, 1::2]  # odd rows, odd cols
    dark1 = phase_only[0::2, 1::2]  # even rows, odd cols
    dark2 = phase_only[1::2, 0::2]  # odd rows, even cols

    # --- Merge methods ---
    h_half, w_half, C = dark1.shape

    # 1st: vertical splicing
    h = h_half
    w = w_half*2
    dark_merge_v = np.zeros((h, w, C), dtype=phase_only.dtype)
    bright_merge_v = np.zeros((h, w, C), dtype=phase_only.dtype)

    dark_merge_v[:, 0::2, :] = dark1
    dark_merge_v[:, 1::2, :] = dark2
    bright_merge_v[:, 0::2, :] = bright1
    bright_merge_v[:, 1::2, :] = bright2

    # 2nd: horizontal splicing
    h = h_half*2
    w = w_half
    dark_merge_h = np.zeros((h, w, C), dtype=phase_only.dtype)
    bright_merge_h = np.zeros((h, w, C), dtype=phase_only.dtype)

    dark_merge_h[0::2, :, :] = dark1
    dark_merge_h[1::2, :, :] = dark2
    bright_merge_h[0::2, :, :] = bright1
    bright_merge_h[1::2, :, :] = bright2

    # save images with updated patch id
    output_dir = f"dataset/{data_name}"

    def save_image(arr, id_offset, name):
        out_path = os.path.join(output_dir, f"{name}_patch_{patch_size}_{id_offset:03d}.png")
        # odak.tools.save_image(out_path, arr, cmin = 0, cmax = 255)
        imageio.imwrite(out_path, arr)
        return id_offset + 1

    id_offset = patch_id + 1
    id_offset = save_image(dark_merge_v, id_offset, name)
    id_offset = save_image(dark_merge_h, id_offset, name)
    id_offset = save_image(bright_merge_v, id_offset, name)
    id_offset = save_image(bright_merge_h, id_offset, name)

    return id_offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--patch_id", required=True, type=str)
    parser.add_argument("--patch_size", required=True, type=str)
    parser.add_argument("--name", required=True, type=str)
    args = parser.parse_args()

    next_patch_id = parse_pixel(args.data_name, args.patch_id, args.patch_size, args.name)
    print(next_patch_id)

if __name__ == "__main__":
    main()
    
