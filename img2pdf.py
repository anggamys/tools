from PIL import Image
import os
import glob

def image_to_pdf(image_path, pdf_path):
    try:
        image = Image.open(image_path)
        # Convert image to RGB to avoid errors (some formats like PNG may be RGBA)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(pdf_path, "PDF", resolution=100.0)
        print(f"PDF saved to: {pdf_path}")
    except Exception as e:
        print(f"Error converting {image_path}: {e}")

# Ensure the result directory exists
os.makedirs("documents/result", exist_ok=True)

source_dir = "documents/source"
result_dir = "documents/result"

# Look for common image formats
image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif')
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(source_dir, ext)))
    image_files.extend(glob.glob(os.path.join(source_dir, ext.upper())))

if not image_files:
    print(f"No images found in {source_dir}")
else:
    for image_path in image_files:
        # Create PDF filename based on original image name
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        pdf_path = os.path.join(result_dir, f"{base_name}.pdf")

        image_to_pdf(image_path, pdf_path)
