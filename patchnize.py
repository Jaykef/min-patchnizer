from patchnizer import Patchnizer
import time

patch_size = 16
video_path = "tests/dogs_in_snow.mp4"
output_dir_video_frames = "video_frames"
output_dir_image_patches = "image_patches"
output_dir_patch_embeddings = "patch_embeddings"

start_time = time.time()

patchnizer = Patchnizer(video_path, output_dir_video_frames, output_dir_image_patches, output_dir_patch_embeddings, patch_size)
time_extract_image_frames = patchnizer.extract_image_frames()
time_image_frames_to_patches = patchnizer.image_frames_to_patches()
time_linearly_embed_patches = patchnizer.linearly_embed_patches()

end_time = time.time()
total_time = end_time - start_time

print(f"Time taken to extract_image_frames: {time_extract_image_frames:.2f} seconds")
print(f"Time taken for image_frames_to_patches: {time_image_frames_to_patches:.2f} seconds")
print(f"Time taken to linearly_embed_patches: {time_linearly_embed_patches:.2f} seconds")
print(f"Total time taken for whole patchization: {total_time:.2f} seconds")
