import os
from PIL import Image

def compress_image_under_size(input_path, output_path, target_kb=500, min_quality=10, step=5):
    quality = 95  # Start from best quality
    img = Image.open(input_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    while quality >= min_quality:
        img.save(output_path, "JPEG", quality=quality, optimize=True)
        size_kb = os.path.getsize(output_path) // 1024
        if size_kb <= target_kb:
            print(f"Compressed image saved to: {output_path} ({size_kb}KB, quality={quality})")
            return
        quality -= step
    print(f"Could not compress below {target_kb}KB, lowest file: {os.path.getsize(output_path)//1024}KB at quality={quality+step}")

# Example usage:
compress_image_under_size(
    "./source/KTM.jpg",
    "./result/KTM_compressed.jpg",
    target_kb=500
)
