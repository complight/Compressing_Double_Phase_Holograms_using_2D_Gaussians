import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image_path: Path,
        num_points: int = 2000,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        model_path = None,
        args = None,
        image_id: int = 1
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_path_to_tensor(image_path).to(self.device)

        self.model_name = model_name
        self.num_points = num_points
        image_path = Path(image_path)
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_{args.iterations}_{num_points}/{self.image_name}")
        
        self.image_id = image_id

        if model_name == "GaussianImage_Cholesky":
            if (image_id-1) % 5 == 0:
                from gaussianimage_cholesky_baseline import GaussianImage_Cholesky
            else:
                from gaussianimage_cholesky import GaussianImage_Cholesky

            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device)
            self.params_record = {
                "iter": [],
                "xyz": [], 
                "cholesky_elements": [],
                "opacity": [],
                "features": [], # Colors
            }

        elif model_name == "GaussianImage_RS":
            if (image_id-1) % 5 == 0:
                from gaussianimage_rs_baseline import GaussianImage_RS
            else:
                from gaussianimage_rs import GaussianImage_RS

            self.gaussian_model = GaussianImage_RS(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device) 
            self.params_record = {
                "iter": [],
                "xyz": [],
                "scaling": [],
                "rotation": [],
                "opacity": [],
                "features": [], # Colors
            }

        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def train(self):     
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_image)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)

                    self.params_record["iter"].append(iter)
                    self.params_record["xyz"].append(self.gaussian_model.get_xyz.cpu().numpy())
                    self.params_record["opacity"].append(self.gaussian_model.get_opacity.cpu().numpy())
                    self.params_record["features"].append(self.gaussian_model.get_features.cpu().numpy())

                    if self.model_name == "GaussianImage_Cholesky":
                        self.params_record["cholesky_elements"].append(self.gaussian_model.get_cholesky_elements.cpu().numpy())
                    elif self.model_name == "GaussianImage_RS":
                        self.params_record["scaling"].append(self.gaussian_model.get_scaling.cpu().numpy())
                        self.params_record["rotation"].append(self.gaussian_model.get_rotation.cpu().numpy())

        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ssim_value = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        np.save(self.log_dir / "params_record.npy", self.params_record)

        return psnr_value, ssim_value, end_time, test_end_time, 1/test_end_time

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()

        # PSNR & SSIM
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ssim_value = ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()

        self.logwriter.write("Test PSNR:{:.4f}, SSIM:{:.6f}".format(psnr, ssim_value))
        
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            name = self.image_name + "_fitting.png" 
            img.save(str(self.log_dir / name))

        return psnr, ssim_value

def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0) #[1, C, H, W]
    img_tensor = img_tensor[:, :, :, :] 
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument("--patchsize", required=True, type=int)
    parser.add_argument("--name", required=True, type=str)
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    print(args.num_points)
    logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    psnrs, ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0
    if args.data_name == "samplehologram":
        start = 0
        # Set image_length based on number of .png files
        dataset_path = Path(f"./dataset/{args.data_name}")
        name = args.name
        patchsize = args.patchsize

        png_files = list(dataset_path.glob(f"{name}_patch_{patchsize}_*.png"))
        image_length = len(png_files) - start
        
    # Iteration of images set
    for i in range(start, start+image_length):

        if args.data_name == "samplehologram":
            image_path = Path(args.dataset) /  f'{name}_patch_{patchsize}_{i+1:03}.png'
            print(image_path)

        trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, 
            iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path, image_id=i+1)
        psnr, ssim, training_time, eval_time, eval_fps = trainer.train()
        psnrs.append(psnr)
        ssims.append(ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write("{}: {}x{}, PSNR:{:.4f}, SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, psnr, ssim, training_time, eval_time, eval_fps))

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ssim = torch.tensor(ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length

    logwriter.write("Average: {}x{}, PSNR:{:.4f}, SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ssim, avg_training_time, avg_eval_time, avg_eval_fps))    

if __name__ == "__main__":
    main(sys.argv[1:])
