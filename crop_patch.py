from pathlib import Path
from PIL import Image
import argparse
import subprocess

def clip(data_name, height, width, name):
    dataset_dir = Path(f"dataset/{data_name}")
    hologram_path = dataset_dir / f"{name}.png"

    img = Image.open(hologram_path)
    img_width, img_height = img.size
        
    if name == "cans" or name == "plastic":
        target_width = width*2
        target_height = width*2
        
        left = (img_width - target_width) // 4
        right = left + target_width
        top = (img_height - target_height) // 4
        bottom = top + target_height
        img = img.crop((left, top, right, bottom))  
        img_width = target_width 
        img_height = target_height
    
    if name == "husky":
        target_width = width*2
        target_height = width*2
        
        left = (img_width - target_width) // 3
        right = left + target_width
        top = (img_height - target_height) // 3
        bottom = top + target_height
        img = img.crop((left, top, right, bottom))  
        img_width = target_width 
        img_height = target_height
    
    if name == "elephants":
        target_width = width*2
        target_height = width*2
        
        left = (img_width - target_width) // 2
        right = left + target_width
        bottom = -((img_height - target_height) // 4) + img_height
        top = bottom - target_height
        img = img.crop((left, top, right, bottom))  
        img_width = target_width 
        img_height = target_height
    
    if name == "dog":
        target_width = width*2
        target_height = width*2
        
        left = (img_width - target_width) // 2
        right = left + target_width
        bottom = -((img_height - target_height) // 2) + img_height
        top = bottom - target_height
        img = img.crop((left, top, right, bottom))  
        img_width = target_width 
        img_height = target_height

    img.save(dataset_dir/ f"{name}_crop_{width}.png", format="PNG", compress_level=0)

    patch_id = 1

    out_dir = dataset_dir

    for top in range(0, img_height, height):
        bottom = min(top + height, img_height)

        for left in range(0, img_width, width):
            # compute the boundary
            right = min(left + width, img_width)

            # crop into patch
            patch = img.crop((left, top, right, bottom))

            # save the crop
            patch_path = out_dir / f"{name}_patch_{width}_{patch_id:03d}.png"
            patch.save(patch_path, format="PNG", compress_level=0)

            # call the decomposition function
            result = subprocess.run(
                ["python", "parse_pixels.py", "--data_name", data_name, "--patch_id", str(patch_id), "--patch_size", str(width), "--name", str(name)],
                capture_output=True,
                text=True
            )

            # update id
            next_id = int(result.stdout.strip())

            patch_id = next_id

    print(f"{patch_id - 1} images saved")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--name", required=True, type=str)
    args = parser.parse_args()

    clip(args.data_name, args.height, args.width, args.name)

if __name__ == "__main__":
    main()
