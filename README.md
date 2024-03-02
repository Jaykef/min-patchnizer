# min-patchnizer

Minimal, clean code for video/image "patchnization". The code here, first extracts still images (frames) from a video, splits the image frames into smaller fixed-size patches, linearly embeds each of them, adds position embeddings, saves the resulting sequence of vectors for use in a Vision Transformer encoder. I tried training the resulting sequence vectors with Karpathy's minbpe and it took ~30s per frame to tokenize.

<img width="825" alt="Screenshot 2024-02-29 at 10 33 33" src="https://github.com/Jaykef/sora-patchnizer/assets/11355002/1aa23e7a-56ed-4e31-af4f-79e969734b0d">
<img width="849" alt="Screenshot 2024-03-02 at 16 02 57" src="https://github.com/Jaykef/min-patchnizer/assets/11355002/446e283b-950d-4c46-babc-8d8c459f15fb"><br>

The files in this repo work as follows:

<ul>
  <li>1. <a href="https://github.com/Jaykef/min-patchnizer/blob/main/patchnizer.py">patchnizer.py</a> Holds code for simple implemenatation of the three stages involved (extract_image_frames from video, reduce image_frames_to_patches of fixed sizes 16x16 pixels, then linearly_embed_patches into a 1D vector sequence with additional position embeddings.</li>
  
  <li>2. <a href="https://github.com/Jaykef/min-patchnizer/blob/main/patchnize.py">patchnize.py</a> performs the whole process with custom configs (patch_size, created dirs, video - I am using the "dogs playing in snow" video by sora).</li>

  <li>3. <a href="https://github.com/Jaykef/min-patchnizer/blob/main/patchnize.py">train.py</a> Trains the resulting one-dimensional vector sequence (linear_patch_embeddings + position_embeddings) on Karpathy's minbpe (a minimal implementation of the byte-pair encoding algorithm).</li>
</ul>


The whole process builds on the approach introduced in the Vision Transformer paper: <a href="https://arxiv.org/abs/2010.11929">An image is worth 16x16 words: Transformers for image recognition at scale."</a>

<br>

