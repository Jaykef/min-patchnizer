import os
import cv2
import numpy as np
import time

class Patchnizer:
    """
    Simple implemenatation of the three stages involved (extract_image_frames from video, reduce image_frames_to_patches of fixed sizes 16x16 pixels, then linearly_embed_patches into a 1D vector sequence with additional position embeddings
    """
    def __init__(self, video_path, output_dir_video_frames, output_dir_image_patches, output_dir_patch_embeddings, patch_size):
        self.video_path = video_path
        self.output_dir_video_frames = output_dir_video_frames
        self.output_dir_image_patches = output_dir_image_patches
        self.output_dir_patch_embeddings = output_dir_patch_embeddings
        self.patch_size = patch_size

    def extract_image_frames(self):
        os.makedirs(self.output_dir_video_frames, exist_ok=True)
        cap = cv2.VideoCapture(self.video_path)
        count = 0
        print("Extracting image frames...")
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % 10 == 0:
                frame_path = os.path.join(self.output_dir_video_frames, f"frame_{count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                print(f"Frame {count} saved.")
            count += 1
        cap.release()
        print("Image frames extraction completed.")
        end_time = time.time()
        return end_time - start_time

    def image_frames_to_patches(self):
        os.makedirs(self.output_dir_image_patches, exist_ok=True)
        print("Converting image frames to patches...")
        start_time = time.time()
        for file_name in os.listdir(self.output_dir_video_frames):
            if file_name.endswith('.jpg'):
                image_path = os.path.join(self.output_dir_video_frames, file_name)
                image = cv2.imread(image_path)
                frame_folder = os.path.join(self.output_dir_image_patches, f"{os.path.splitext(file_name)[0]}")
                os.makedirs(frame_folder, exist_ok=True)
                print(f"Processing image {file_name}...")
                for i in range(0, image.shape[0], self.patch_size):
                    for j in range(0, image.shape[1], self.patch_size):
                        patch = image[i:i+self.patch_size, j:j+self.patch_size]
                        patch_path = os.path.join(frame_folder, f"patch_{j:04d}.jpg")
                        cv2.imwrite(patch_path, patch)
                        print(f"Patch {patch_path} saved.")
        print("Image frames to patches conversion completed.")
        end_time = time.time()
        return end_time - start_time

    def linearly_embed_patches(self):
        os.makedirs(self.output_dir_patch_embeddings, exist_ok=True)
        print("Linearly embedding patches...")
        start_time = time.time()
        for folder_name in os.listdir(self.output_dir_image_patches):
            folder_path = os.path.join(self.output_dir_image_patches, folder_name)
            if os.path.isdir(folder_path):
                embeddings_folder = os.path.join(self.output_dir_patch_embeddings, folder_name)
                os.makedirs(embeddings_folder, exist_ok=True)
                print(f"Processing patches in folder {folder_name}...")
                for image_name in os.listdir(folder_path):
                    if image_name.endswith('.jpg'):
                        image_path = os.path.join(folder_path, image_name)
                        image = cv2.imread(image_path)
                        # Reshape image patch into a linear vector
                        linear_embedding = image.reshape(-1)
                        # Add position embeddings
                        position_embedding = np.arange(len(linear_embedding))
                        # Concatenate linear embedding with position embedding
                        patch_embedding = np.concatenate([linear_embedding, position_embedding])
                        patch_embedding_text = ' '.join(map(str, patch_embedding))
                        # Save patch embedding as .txt file
                        embedding_path = os.path.join(embeddings_folder, f"patch_embedding_{os.path.splitext(image_name)[0]}.txt")
                        with open(embedding_path, 'w') as f:
                            f.write(patch_embedding_text)
                        print(f"Embedding for patch {image_name} saved.")
        print("Patch embedding completed.")
        end_time = time.time()
        return end_time - start_time
