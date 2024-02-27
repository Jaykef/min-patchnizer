import os
import cv2
import matplotlib.pyplot as plt

def extract_video_patches(video_path, patch_width, patch_height, output_dir):
  
    os.makedirs(output_dir, exist_ok=True)
  
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to patch size 
        frame = cv2.resize(frame, (patch_width, patch_height))

        # Save patch
        patch_path = os.path.join(output_dir, f"patch_{frame_count}.jpg")
        cv2.imwrite(patch_path, frame)

        frame_count += 1

    cap.release()


if __name__ == "__main__":
    # Define parameters
    video_path = "dogs_in_snow.mp4"
    patch_width = 500  
    patch_height = 250 
    output_dir = "video_patches"

    extract_video_patches(video_path, patch_width, patch_height, output_dir)

