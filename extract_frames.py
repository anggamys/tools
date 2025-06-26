import cv2
import os

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    count = 0

    while success:
        filename = os.path.join(output_folder, f"frame_{count:06d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        success, frame = vidcap.read()
        count += 1

    vidcap.release()
    print(f"Done! {count} frames extracted.")

# Example usage:
extract_frames("./source/", "./result/frames/")
