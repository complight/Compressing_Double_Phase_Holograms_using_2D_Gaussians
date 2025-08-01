import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def parse_train_log(eval_log_path):

    if not eval_log_path.exists():
        print(f"train.txt not found at {eval_log_path}")
        return None

    with open(eval_log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # extract last three lines
    last_lines = lines[-(3):]

    pattern = re.compile(
        r"(?P<name>\w+):\s+(?P<height>\d+)x(?P<width>\d+),\s+PSNR:(?P<psnr>[0-9.]+),\s+SSIM:(?P<ssim>[0-9.]+)"
    )

    metrics = {}
    image_names = []

    for line in last_lines:
        m = pattern.search(line)
        if m:
            name = m.group("name")
            w = int(m.group("width"))
            h = int(m.group("height"))  
            psnr = float(m.group("psnr"))
            ssim = float(m.group("ssim"))

            metrics[name] = {
                "width": w,
                "height": h,
                "PSNR": psnr,
                "SSIM": ssim,
            }
            image_names.append(name)
        else:
            print(f"Warning: line parse failed: {line.strip()}")

    return {"image_names": image_names, "metrics": metrics}

def plot_metrics(data_name, model_name, iterations, num_points_list, patch_size, name):

    all_data = {}

    for num_points in num_points_list:
        eval_log_path = Path(f"./checkpoints/{data_name}/{model_name}_{iterations}_{num_points}/{name}_eval_{patch_size}.txt")
        parsed = parse_train_log(eval_log_path)
        if parsed is None:
            continue
        for image_name in parsed["image_names"]:
            met = parsed["metrics"][image_name]
            w, h = met["width"], met["height"]
            compression_ratio = (num_points*8*2) / (3* patch_size * patch_size)
            if image_name not in all_data:
                all_data[image_name] = {
                    "PSNR": {},
                    "SSIM": {},
                    "width": w,
                    "height": h,
                }
            all_data[image_name]["PSNR"][compression_ratio] = met["PSNR"]
            all_data[image_name]["SSIM"][compression_ratio] = met["SSIM"]

    # Plot
    for image_name, metrics_dict in all_data.items():
        width = metrics_dict["width"]
        height = metrics_dict["height"]
        for metric in ["PSNR", "SSIM"]:
            x = sorted(metrics_dict[metric].keys())
            y = [metrics_dict[metric][xx] for xx in x]

            if len(x) == 0:
                print(f"Skipping plot {metric} for {image_name} due to no data")
                continue

            plt.figure()
            plt.plot(x, y, marker='o')
            plt.xlabel("Compression Ratio")
            plt.ylabel(metric)
            plt.title(f"{metric} vs Compression Ratio\nImage: {image_name} ({width}x{height})")
            plt.grid(True)

            save_dir = Path(f"./checkpoints/{data_name}/quality_plots")
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{name}_{model_name}_patch_{image_name}_{metric}.png"
            plt.savefig(save_dir / filename)
            plt.close()
            print(f"Saved plot {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--num_points_list", nargs='+', required=True, type=int)
    parser.add_argument("--patchsize", required=True, type=int)
    parser.add_argument("--name", required=True, type=str)
    args = parser.parse_args()

    plot_metrics(args.data_name, args.model_name, args.iterations, args.num_points_list, args.patchsize, args.name)

if __name__ == "__main__":
    main()
