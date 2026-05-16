from PIL import Image
import os

input_dir = "result/frames"
output_dir = "result/rotated"

# Buat folder output jika belum ada
os.makedirs(output_dir, exist_ok=True)

# Loop semua file di folder input
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Buka dan rotasi gambar
        image = Image.open(input_path)
        rotated = image.rotate(90, expand=True)
        rotated.save(output_path)

        print(f"Rotated: {filename}")

print("Semua gambar selesai diputar dan disimpan di", output_dir)
