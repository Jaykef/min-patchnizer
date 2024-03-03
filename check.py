import os
import numpy as np
import cv2

def recover_patches(embeddings_dir, output_dir):
    "Checks to see if the patch embeddings match the original image patches by recovering back the original image patches from patch_embeddings"
    
    os.makedirs(output_dir, exist_ok=True)
    for folder_name in os.listdir(embeddings_dir):
        folder_path = os.path.join(embeddings_dir, folder_name)
        if os.path.isdir(folder_path):
            patches_folder = os.path.join(output_dir, folder_name)
            os.makedirs(patches_folder, exist_ok=True)
            for embedding_name in os.listdir(folder_path):
                if embedding_name.endswith('.txt'):
                    embedding_path = os.path.join(folder_path, embedding_name)
                    with open(embedding_path, 'r') as f:
                        embedding = np.fromstring(f.read(), sep=' ')
                    # Extract linear embedding
                    linear_embedding = embedding[:-(len(embedding) // 2)]
                    # Calculate patch size
                    patch_size = int(np.sqrt(len(linear_embedding) // 3))
                    patch_channels = 3  # RGB channels
                    # Check if the embedding size matches the expected patch size
                    if len(linear_embedding) == patch_size * patch_size * patch_channels:
                        # Reshape linear embedding into the original patch shape
                        patch = linear_embedding.reshape(patch_size, patch_size, patch_channels)
                        # Save the patch as original .jpg files
                        patch_path = os.path.join(patches_folder, f"{os.path.splitext(embedding_name)[0]}.jpg")
                        cv2.imwrite(patch_path, patch)
                        print(f"Recovered patch saved: {patch_path}")
                    else:
                        print(f"Error: Invalid embedding size for {embedding_name}")

embeddings_dir = "patch_embeddings"
recovered_patches_dir = "check"

recover_patches(embeddings_dir, recovered_patches_dir)
