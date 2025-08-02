### Getting Started
This project is based on the gsplat submodule, which has been licensed under the MIT License. You can get started by:

<pre>
git submodule update --init --recursive
cd gsplat
pip install .[dev]
cd ../
pip install -r requirements.txt
</pre>

### Compressed Representation
To run the GaussianImage-based patch-based decomposition framework, each bash file in
<pre>./scripts/gaussianimage_cholesky/  # Using Cholesky factorization
or 
./scripts/gaussianimage_rs/  # Using RS factorization
</pre> 
corresponds to each sample hologram. All of them have been licensed for commercial and non-commercial purposes. You can run the corresponding command in terminal for the specific sample hologram with different patch size. For example:

<pre># Sample Hologram - Police Dog - with a 64 patch size
bash scripts/gaussianimage_cholesky/dog_64.sh dataset/samplehologram # Using Cholesky factorization
bash scripts/gaussianimage_rs/dog_64.sh dataset/samplehologram # Using RS factorization
</pre>

The default setting crops four neighboring patches from the sample hologram stored in 
<pre>./dataset/samplehologram/  
</pre>
directory with numbered file names. After cropping, each patch and its decomposed components are sequentially numbered as follows:

<pre>
|--samplehologram
| |--patch_001 # cropped patch from sample hologram
| |--patch_002 # high-value component from vertical decomposition of this patch
| |--patch_003 # high-value component from horizontal decomposition of this patch
| |--patch_004 # low-value component from vertical decomposition of this patch
| |--patch_005 # low-value component from horizontal decomposition of this patch
| |--patch_006 # another neighbouring patch
...
</pre>

After training, you can find the combination of compressed patches and the corresponding simulated reconstructions in the file:
<pre>./checkpoints/samplehologram/GaussianImage_Cholesky_70000_200/results # Using Choleksy factorization with 200 Gaussians and 70000 epochs for training
./checkpoints/samplehologram/GaussianImage_RS_70000_200/results # Using RS factorization with 200 Gaussians and 70000 epochs for training
</pre>

For a more detailed analysis, you can find the statistical plots showing parameter changes throughout the training process in the `/stat_plots` folder. Additionally, plots of quality metrics (PSNR, SSIM) versus compression ratio can be found in the `/quality_plots` folder. As an example, for the sample hologram Police Dog with a patch size of 64 using Cholesky factorization, the corresponding directories are as follows:

<pre>
|--checkpoints
|-|--samplehologram
|---|--GaussianImage_Cholesky_70000_100
|-----|--dog_patch_64_001
|-----|--dog_patch_64_002
|-----|--dog_patch_64_003
|-----|--...
|-----|--dog_patch_64_020
|-----|--**results**
|-----|--**stat_plots**
|---|--GaussianImage_Cholesky_70000_200
|---|--...
|---|--**quality_plots**
</pre>