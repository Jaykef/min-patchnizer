import os
import cv2
import numpy as np

def extract_patches(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % 10 == 0:
            frame_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()

def split_into_patches(image, patch_size):
    height, width, _ = image.shape
    patches = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches

def linearly_embed(patch):
    return patch.reshape(-1)

def add_position_embedding(embedding, position):
    return np.concatenate((embedding, [position]))

def save_embeddings(embeddings, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, embedding in enumerate(embeddings):
        np.save(os.path.join(output_dir, f'embedding_{i}.npy'), embedding)

# Example usage:
video_path = "dogs.mp4"
video_patches_dir = "video_patches"
patch_embeddings_dir = "patch_embeddings"

# Parameters
patch_size = 64  # Size of each patch

# Step 1: Extract patches from the video
extract_patches(video_path, video_patches_dir)

# Step 2: Split the image patches into fixed-size patches
embeddings = []
for file_name in os.listdir(video_patches_dir):
    if file_name.endswith('.jpg'):
        image_path = os.path.join(video_patches_dir, file_name)
        image = cv2.imread(image_path)
        patches = split_into_patches(image, patch_size)
        for patch in patches:
            embedding = linearly_embed(patch)
            embeddings.append(embedding)

# Step 3: Add position embeddings
embeddings_with_positions = [add_position_embedding(embedding, i) for i, embedding in enumerate(embeddings)]

# Step 4: Save the resulting sequence of vectors in the patch_embeddings folder
save_embeddings(embeddings_with_positions, patch_embeddings_dir)
