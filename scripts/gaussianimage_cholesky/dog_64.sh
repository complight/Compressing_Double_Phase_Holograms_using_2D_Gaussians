#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

# crop the sample hologram into patches + parse high-value and low-value pixels in each crop into two images
python crop_patch.py --data_name samplehologram --height 64 --width 64 --name dog

num_points_list64=(100 200 400 600 800)

for num_points in "${num_points_list64[@]}"
do
CUDA_VISIBLE_DEVICES=0 python train_hologram.py -d $data_path \
--data_name samplehologram --model_name GaussianImage_Cholesky --num_points $num_points --iterations 70000 --patchsize 64 --name dog --save_imgs
done

# plot parameters change for each patch
python plot_param.py --data_name samplehologram --model_name GaussianImage_Cholesky --iterations 70000 --num_points_list "${num_points_list64[@]}" --patchsize 64 --name dog

# plot quality metrics vs compression ratio for each patch
python plot_quality.py --data_name samplehologram --model_name GaussianImage_Cholesky --iterations 70000 --num_points_list "${num_points_list64[@]}" --patchsize 64 --name dog

# reorganize decomposed components back to double-phase modulation
python merge_pixels.py --data_name samplehologram --model_name GaussianImage_Cholesky --iterations 70000 --num_points_list "${num_points_list64[@]}" --patchsize 64 --name dog

# evaluate decomposition method
python eval_merge.py --data_name samplehologram --model_name GaussianImage_Cholesky --iterations 70000 --num_points_list "${num_points_list64[@]}" --patchsize 64 --name dog

# combine compressed patches
python combine_crop.py --data_name samplehologram --model_name GaussianImage_Cholesky --iterations 70000 --num_points_list "${num_points_list64[@]}" --patchsize 64 --name dog

# plot quality metrics vs compression ratio (for combination)
python plot_quality_combine.py --data_name samplehologram --model_name GaussianImage_Cholesky --iterations 70000 --num_points_list "${num_points_list64[@]}" --patchsize 64 --name dog

# simulate the compressed holograms
python test_learn_wave_propagator.py --data_name samplehologram --model_name GaussianImage_Cholesky --iterations 70000 --num_points_list "${num_points_list64[@]}" --patchsize 64 --name dog
