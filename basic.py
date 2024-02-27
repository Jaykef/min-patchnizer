import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import time

def extract_frames(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["ffmpeg", "-i", video_path, "-vf", f"fps={frame_rate}", f"{output_dir}/frame_%04d.jpg"])

def animate_frames(frames_dir, frame_rate=2):
    frame_files = sorted(os.listdir(frames_dir))
    frames = [Image.open(os.path.join(frames_dir, frame_file)) for frame_file in frame_files]

    fig, ax = plt.subplots(figsize=(frames[0].width / 100, frames[0].height / 100))
    ax.axis('off')
    ax.imshow(frames[0], aspect='auto')
    ax.set_aspect('equal')

    def update(frame):
        ax.clear()
        ax.imshow(frames[frame], aspect='auto')
        ax.axis('off')

    ani = FuncAnimation(fig, update, frames=len(frames), interval=1000 / frame_rate)
    plt.tight_layout()
    plt.show()


def load_image(image_path):
    return np.array(Image.open(image_path))


def extract_patches(image, patch_size, step_size):
    patches = []
    height, width, _ = image.shape
    for y in range(0, height - patch_size + 1, step_size):
        for x in range(0, width - patch_size + 1, step_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches


def compute_linear_embeddings(patch):
    return np.mean(patch, axis=(0, 1))


def process_video(video_path, output_dir, embeddings_dir, patch_size, step_size, frame_rate):
    
    extract_frames(video_path, output_dir, frame_rate)

   
    os.makedirs(embeddings_dir, exist_ok=True)

    
    for frame_file in sorted(os.listdir(output_dir)):
        frame_path = os.path.join(output_dir, frame_file)
        image = load_image(frame_path)  # Load image using PIL
        patches = extract_patches(image, patch_size, step_size)
        embeddings = [compute_linear_embeddings(patch) for patch in patches]

        # Save embeddings
        frame_name = os.path.splitext(frame_file)[0]
        embeddings_path = os.path.join(embeddings_dir, f"{frame_name}_embeddings.npy")
        np.save(embeddings_path, embeddings)


video_path = "dogs.mp4"
output_dir = "video_patches"
embeddings_dir = "video_embeddings"
patch_size = 64
step_size = 32
frame_rate = 2  
process_video(video_path, output_dir, embeddings_dir, patch_size, step_size, frame_rate)
animate_frames(output_dir, frame_rate)
