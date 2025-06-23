from PIL import Image

def image_to_pdf(image_path, pdf_path):
    image = Image.open(image_path)
    # Convert image to RGB to avoid errors (some formats like PNG may be RGBA)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(pdf_path, "PDF", resolution=100.0)
    print(f"PDF saved to: {pdf_path}")

# Example usage:
image_to_pdf("./result/KTM_compressed.jpg", "./result/KTM_compressed.pdf")
