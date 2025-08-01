from pathlib import Path
import argparse
import re
import matplotlib.pyplot as plt

def parse_train_log(train_log_path, fullname, patch_size):
    """
    read train.txt, return dictionary structure
    
    """
    if not train_log_path.exists():
        print(f"train.txt not found at {train_log_path}")
        return None

    with open(train_log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pattern = re.compile(
        rf"(?P<name>{fullname}_patch_{patch_size}_)(?P<id>\d+):\s+(?P<w>\d+)x(?P<h>\d+),\s+PSNR:(?P<psnr>[0-9.]+),\s+SSIM:(?P<ssim>[0-9.]+)?"
    )

    metrics = {}
    image_names = []

    for line in lines:
        m = pattern.search(line)
        if m:
            name = m.group("name")
            id = m.group("id")
            w = int(m.group("w"))
            h = int(m.group("h"))
            psnr = float(m.group("psnr"))
            ssim = float(m.group("ssim"))

            metrics[f"{name}{id}"] = {
                "id":id,
                "width": w,
                "height": h,
                "PSNR": psnr,
                "SSIM": ssim,
            }
            image_names.append(f"{name}{id}")

    return {"image_names": image_names, "metrics": metrics}

def get_numeric_id(image_name):
    return int(image_name.split("_")[-1]) 

def plot_metrics(data_name, model_name, iterations, num_points_list, name, patch_size):

    all_data = {}

    for num_points in num_points_list:
        train_log_path = Path(f"./checkpoints/{data_name}/{model_name}_{iterations}_{num_points}/train.txt")
        parsed = parse_train_log(train_log_path, name, patch_size)
        if parsed is None:
            continue
        for image_name in parsed["image_names"]:
            met = parsed["metrics"][image_name]
            w, h = met["width"], met["height"]
            id = met["id"]
            compression_ratio = (num_points*8) / (3* w * h)
            if image_name not in all_data:
                all_data[image_name] = {
                    "PSNR": {},
                    "SSIM": {},
                    "width": w,
                    "height": h,
                }
            all_data[image_name]["PSNR"][compression_ratio] = met["PSNR"]
            all_data[image_name]["SSIM"][compression_ratio] = met["SSIM"]
    
    # Sort and group every 5 by ID
    sorted_image_names = sorted(all_data.keys(), key=get_numeric_id)
    grouped = [sorted_image_names[i:i+5] for i in range(0, len(sorted_image_names), 5)]

    for group in grouped:
        if len(group) < 1:
            continue

        # 1. Plot baseline image (e.g. 001) individually
        baseline_image = group[0]
        metrics_dict = all_data[baseline_image]
        width = metrics_dict["width"]
        height = metrics_dict["height"]
        
        for metric in ["PSNR", "SSIM"]:
            x = sorted(metrics_dict[metric].keys())
            y = [metrics_dict[metric][xx] for xx in x]
            if len(x) == 0:
                print(f"Skipping plot {metric} for {baseline_image} due to no data")
                continue

            plt.figure()
            plt.plot(x, y, marker='o', label="original")
            plt.xlabel("Compression Ratio")
            plt.ylabel(metric)
            plt.title(f"{metric} vs Compression Ratio\nBaseline Image: {baseline_image} ({width}x{height})")
            plt.grid(True)

            save_dir = Path(f"./checkpoints/{data_name}/quality_plots")
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{baseline_image}_{metric}_{model_name}.png"
            plt.savefig(save_dir / filename)
            plt.close()
            print(f"Saved plot {filename}")

        # 2. Plot high-value images
        v_image = group[1]
        h_image = group[2]
        for metric in ["PSNR", "SSIM"]:
            plt.figure()
            for image_name, color, label in zip([v_image, h_image], ['blue', 'red'], ['vertical', 'horizontal']):
                metrics_dict = all_data[image_name]
                x = sorted(metrics_dict[metric].keys())
                y = [metrics_dict[metric][xx] for xx in x]
                if len(x) == 0:
                    print(f"Skipping {metric} for {image_name}")
                    continue
                plt.plot(x, y, marker='o', label=label, color=color)

            plt.xlabel("Compression Ratio")
            plt.ylabel(metric)
            plt.title(f"{metric} vs Compression Ratio\nHigh-value Images: {v_image} & {h_image}")
            plt.grid(True)
            plt.legend()

            filename = f"{baseline_image}_highvalue_{metric}_{model_name}.png"
            plt.savefig(save_dir / filename)
            plt.close()
            print(f"Saved plot {filename}")

        # 3. Plot low-value images
        v_image = group[3]
        h_image = group[4]
        for metric in ["PSNR", "SSIM"]:
            plt.figure()
            for image_name, color, label in zip([v_image, h_image], ['blue', 'red'], ['vertical', 'horizontal']):
                metrics_dict = all_data[image_name]
                x = sorted(metrics_dict[metric].keys())
                y = [metrics_dict[metric][xx] for xx in x]
                if len(x) == 0:
                    print(f"Skipping {metric} for {image_name}")
                    continue
                plt.plot(x, y, marker='o', label=label, color=color)

            plt.xlabel("Compression Ratio")
            plt.ylabel(metric)
            plt.title(f"{metric} vs Compression Ratio\nLow-value Images: {v_image} & {h_image}")
            plt.grid(True)
            plt.legend()

            filename = f"{baseline_image}_lowvalue_{metric}_{model_name}.png"
            plt.savefig(save_dir / filename)
            plt.close()
            print(f"Saved plot {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--num_points_list", nargs='+', required=True, type=int)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--patchsize", required=True, type=int)
    args = parser.parse_args()

    plot_metrics(args.data_name, args.model_name, args.iterations, args.num_points_list, args.name, args.patchsize)

if __name__ == "__main__":
    main()
