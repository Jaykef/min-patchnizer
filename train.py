from base import Patchnizer

patch_size = 16
video_path = "dogs.mp4"
output_dir_video_frames = "video_frames"
output_dir_image_patches = "image_patches"
output_dir_patch_embeddings = "patch_embeddings"

patchnizer = Patchnizer(video_path, output_dir_video_frames, output_dir_image_patches, output_dir_patch_embeddings, patch_size)
patchnizer.extract_image_frames()
patchnizer.image_frames_to_patches()
patchnizer.linearly_embed_patches()
