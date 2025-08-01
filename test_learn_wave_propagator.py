import sys
import os
import odak
import numpy as np
import torch
import argparse
import math
import torch.nn.functional as F
import torchvision.transforms as T
from pytorch_msssim import ssim
import lpips
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

def test(hologram_path, output_dir, device, wavelengths, pixel_pitch, number_of_frames,
                         number_of_depth_layers, volume_depth, image_location_offset,
                         propagation_type, propagator_type, laser_channel_power,
                         aperture_size, aperture, method, name, patch_size, dataname, model_name):

    odak.tools.check_directory(output_dir)

    # Simulate hologram
    hologram_phases = odak.learn.tools.load_image(
                                                hologram_path,
                                                normalizeby = 255.,
                                                torch_style = True
                                                ).to(device) * odak.pi * 2

    resolution = [2400, 4094]
    hologram_phases = odak.learn.tools.zero_pad(hologram_phases, size = resolution)
    propagator = odak.learn.wave.propagator(
                                            resolution = resolution,
                                            wavelengths = wavelengths,
                                            pixel_pitch = pixel_pitch,
                                            number_of_frames = number_of_frames,
                                            number_of_depth_layers = number_of_depth_layers,
                                            volume_depth = volume_depth,
                                            image_location_offset = image_location_offset,
                                            propagation_type = propagation_type,
                                            propagator_type = propagator_type,
                                            laser_channel_power = laser_channel_power,
                                            aperture_size = aperture_size,
                                            aperture = aperture,
                                            method = method,
                                            device = device
                                        )
    # Reconstruct Focal Stack
    reconstruction_intensities = propagator.reconstruct(hologram_phases, amplitude = None)
    reconstruction_intensities = torch.sum(reconstruction_intensities, axis = 0)

    for depth_id, reconstruction_intensity in enumerate(reconstruction_intensities):
        odak.learn.tools.save_image(
                                    '{}/{}_{}_{}_{}_reconstruction_image_{:03d}.png'.format(output_dir, dataname, model_name, patch_size, name, depth_id),
                                    reconstruction_intensity,
                                    cmin = 0.,
                                    cmax = 1.
                                )

def eval(log_dir, original_image_path, device, output_dir, label, data_name, patch_size, name, model_name):
    to_tensor = T.ToTensor()

    # Extract size
    ori_img = Image.open(original_image_path)
    ori_tensor = to_tensor(ori_img).unsqueeze(0).to(device) #[1, 3, H, W]
    _, _, H_ori, W_ori = ori_tensor.shape

    # For evaluation
    # Load LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    to_tensor = T.ToTensor()

    if not os.path.exists(log_dir):
        with open(log_dir, "w") as f:
            f.write("filename: HxW, PSNR, SSIM, LPIPS\n")
    
    for depth_id in range(3):
        
        rec_img = Image.open(f'{output_dir}/{name}_{model_name}_{patch_size}_compressed_reconstruction_image_{depth_id:03d}.png')
        rec_tensor = to_tensor(rec_img).unsqueeze(0).to(device)  # [1, 3, H, W]

        or_img = Image.open(f'./dataset/{data_name}/{name}_{model_name}_{patch_size}_sample_reconstruction_image_{depth_id:03d}.png')
        or_tensor = to_tensor(or_img).unsqueeze(0).to(device)  # [1, 3, H, W]

        # Resize rec_tensor to match ori_tensor
        if rec_tensor.shape != ori_tensor.shape:
            _, _, H_rec, W_rec = rec_tensor.shape

            top = (H_rec - H_ori) // 2
            left = (W_rec - W_ori) // 2

            rec_tensor = rec_tensor[:, :, top:top + H_ori, left:left + W_ori]
            or_tensor = or_tensor[:, :, top:top + H_ori, left:left + W_ori]

        # Calculate metrics
        mse = F.mse_loss(rec_tensor, or_tensor)
        psnr = 10 * math.log10(1.0 / mse.item())
        
        ssim_val = ssim(rec_tensor, or_tensor, data_range=1, size_average=True).item()

        lpips_val = lpips_model((rec_tensor * 2 - 1), (or_tensor * 2 - 1)).mean().item()

        line = f"{label}_reconstruction_image_{depth_id:03d}: {rec_tensor.shape[2]}x{rec_tensor.shape[3]}, PSNR:{psnr:.4f}, SSIM:{ssim_val:.4f}, LPIPS:{lpips_val:.4f}"
        print(line)

        with open(log_dir, "a") as f:
            f.write(line + "\n")

def initialize(data_name, model_name, iterations, num_points_list, patch_size, name):

    wavelengths = [639e-9, 515e-9, 473e-9]
    pixel_pitch = 3.74e-6
    number_of_frames = 3
    number_of_depth_layers = 3
    volume_depth = 5e-3
    image_location_offset = 0.
    propagation_type = 'Bandlimited Angular Spectrum'
    propagator_type = 'forward'
    laser_channel_power = None
    aperture = None
    aperture_size = 1800
    method = 'conventional'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for num_points in num_points_list:
        print(num_points)
        # Path and file names can be changed to the final compressed images
        base_path = f"./checkpoints/{data_name}/{model_name}_{iterations}_{num_points}/results"
        
        image_info = { 
            "hor": os.path.join(base_path, f"{name}_{model_name}_{patch_size}_combined_horizontal_fitting.png"),
            "ver": os.path.join(base_path, f"{name}_{model_name}_{patch_size}_combined_vertical_fitting.png"),
            "ori": os.path.join(base_path, f"{name}_{model_name}_{patch_size}_combined_original_fitting.png")
            }
        
        # Simulate the original clipped
        ori_path = f"./dataset/{data_name}/{name}_crop_{patch_size}.png"
        output_dir = f"./dataset/{data_name}"
        
        test(
            hologram_path=ori_path,
            output_dir=output_dir,
            device=device,
            wavelengths=wavelengths,
            pixel_pitch=pixel_pitch,
            number_of_frames=number_of_frames,
            number_of_depth_layers=number_of_depth_layers,
            volume_depth=volume_depth,
            image_location_offset=image_location_offset,
            propagation_type=propagation_type,
            propagator_type=propagator_type,
            laser_channel_power=laser_channel_power,
            aperture_size=aperture_size,
            aperture=aperture,
            method=method,
            name='sample',
            patch_size=patch_size,
            dataname=name,
            model_name=model_name
        )
        
        # Simulate the compressed clipped + evaluate the compression quality
        for label, image_path in image_info.items():
            output_dir = os.path.join(base_path, label)
            log_dir = os.path.join(os.path.dirname(base_path), f"{name}_eval_{patch_size}.txt")
            test(
                hologram_path=image_path,
                output_dir=output_dir,
                device=device,
                wavelengths=wavelengths,
                pixel_pitch=pixel_pitch,
                number_of_frames=number_of_frames,
                number_of_depth_layers=number_of_depth_layers,
                volume_depth=volume_depth,
                image_location_offset=image_location_offset,
                propagation_type=propagation_type,
                propagator_type=propagator_type,
                laser_channel_power=laser_channel_power,
                aperture_size=aperture_size,
                aperture=aperture,
                method=method,
                name='compressed',
                patch_size=patch_size,
                dataname=name,
                model_name=model_name
            )
            eval(
                log_dir=log_dir,
                original_image_path=ori_path,
                device=device,
                output_dir=output_dir,
                label=label,
                data_name=data_name,
                patch_size=patch_size,
                name=name,
                model_name=model_name
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--num_points_list", nargs='+', required=True, type=int)
    parser.add_argument("--patchsize", required=True, type=int)
    parser.add_argument("--name", required=True, type=str)
    args = parser.parse_args()

    initialize(args.data_name, args.model_name, args.iterations, args.num_points_list, args.patchsize, args.name)

if __name__ == '__main__':
    main() 