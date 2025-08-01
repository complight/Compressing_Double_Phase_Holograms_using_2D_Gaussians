import os
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def plot_params_change(data_name, model_name, iterations, patchsize, num_points_list, name, change_threshold=1e-3):
    checkpoints_root = Path(f"./checkpoints/{data_name}")
    static_points_log = []

    for num_points in num_points_list:
        folder = checkpoints_root / f"{model_name}_{iterations}_{num_points}"
        save_folder = folder / "stat_plots"
        os.makedirs(save_folder, exist_ok=True)

        # extract the image length
        patch_files = [
            p for p in folder.glob(f"{name}_patch_{patchsize}_*")
            if p.stem.split("_")[3].isdigit()
        ]
        max_patch_id = max([int(p.stem.split("_")[3]) for p in patch_files])

        for patch in range(1, max_patch_id + 1):
            
            log_file = folder / f"{name}_patch_{patchsize}_{patch:03d}"/ f"params_record.npy"
            if not log_file.exists():
                print(f"File not found: {log_file}")
                continue
            
            # extract parameters change
            params_record = np.load(log_file, allow_pickle=True).item()
            iterations_list = params_record["iter"]
            xyz = params_record["xyz"]  # [time][num_points, 2]
            features = params_record["features"]  # [time][num_points, 3]

            if model_name == "GaussianImage_Cholesky":
                cholesky_elements = params_record["cholesky_elements"]  # [time][num_points, 3]
            elif model_name == "GaussianImage_RS":
                scaling = params_record["scaling"]  # [time][num_points, 2]
                rotation = params_record["rotation"]  # [time][num_points, 1]
            else:
                print(f"Unknown model: {model_name}")
                continue

            def plot_param(param_list, param_name, param_dim, dim_labels=None):
                if dim_labels is None:
                    dim_labels = [f'' for i in range(param_dim)]

                for dim in range(param_dim):
                    plt.figure(figsize=(12, 8))
                    for point_idx in range(param_list[0].shape[0]):
                        dim_values = [param[point_idx, dim] for param in param_list]
                        plt.plot(iterations_list, dim_values, label=f'Point {point_idx}')
                    plt.xlabel('Iteration')
                    plt.ylabel(f'{param_name}_{dim_labels[dim]}')
                    plt.title(f'Patch {patch:03d} - {param_name}_{dim_labels[dim]} Change')
                    plt.grid()
                    plt.savefig(save_folder / f'{name}_patch_{patchsize}_{patch:03d}_{param_name}_{dim_labels[dim]}.png')
                    plt.close()
            
            def find_static_points(param_list, param_name):
                param_array = np.array(param_list)  # [num_iterations, num_points, dim]
                num_points = param_array.shape[1]
                num_static = 0
                static_indices = []

                for point_idx in range(num_points):
                    point_values = param_array[:, point_idx, :]  # [num_iterations, dim]
                    max_change = np.abs(point_values - point_values[0]).max()
                    if max_change < change_threshold:
                        num_static += 1
                        static_indices.append(point_idx)

                if static_indices:
                    static_points_log.append({
                        "patch_id": patch,
                        "total_gaussians": num_points,
                        "param_name": param_name,
                        "static_indices": static_indices
                    })

            # x, y
            plot_param(xyz, "xyz", 2, dim_labels=['x', 'y'])
            find_static_points(xyz, "xyz")
            # R, G, B
            plot_param(features, "features", 3, dim_labels=['R', 'G', 'B'])
            find_static_points(features, "features")

            if model_name == "GaussianImage_Cholesky":
                plot_param(cholesky_elements, "cholesky_elements", 3, dim_labels=['L00', 'L10', 'L11'])
                find_static_points(cholesky_elements, "cholesky_elements")
            elif model_name == "GaussianImage_RS":
                plot_param(scaling, "scaling", 2, dim_labels=['Scale X', 'Scale Y'])
                find_static_points(scaling, "scaling")

                plot_param(rotation, "rotation", 1)
                find_static_points(rotation, "rotation")

    if static_points_log:
        log_path = checkpoints_root / f"static_points_log_{model_name}_{iterations}_{patchsize}.txt"
        with open(log_path, 'w') as f:
            for record in static_points_log:
                f.write(f"Patch {record['patch_id']:03d} | Total Gaussians: {record['total_gaussians']} | Param: {record['param_name']} | Static Indices: {record['static_indices']}\n")
        print(f"Static points log saved to {log_path}")
    else:
        print("No static points found.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--num_points_list", nargs='+', required=True, type=int)
    parser.add_argument("--patchsize", required=True, type=int)
    parser.add_argument("--name", required=True, type=str)
    args = parser.parse_args()

    plot_params_change(args.data_name, args.model_name, args.iterations, args.patchsize, args.num_points_list, args.name)

if __name__ == "__main__":
    main()
