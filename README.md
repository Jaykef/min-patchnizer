# min-patchnizer

Minimal, clean code for video/image "patchnization". The code here, first extracts still images (frames) from a video (.mp4), splits the image frames into smaller fixed-size patches, linearly embeds each of them, adds position embeddings, saves the resulting sequence of vectors for use in a Vision Transformer encoder. I tried training the resulting sequence vectors with Karpathy's minbpe and it took ~30s per frame to tokenize.

<img width="825" alt="Screenshot 2024-02-29 at 10 33 33" src="https://github.com/Jaykef/sora-patchnizer/assets/11355002/1aa23e7a-56ed-4e31-af4f-79e969734b0d"><br>

The whole process builds on the approach introduced in the Vision Transformer paper: <a href="https://arxiv.org/abs/2010.11929">An image is worth 16x16 words: Transformers for image recognition at scale."</a>

<br><img width="849" alt="Screenshot 2024-03-02 at 16 02 57" src="https://github.com/Jaykef/min-patchnizer/assets/11355002/446e283b-950d-4c46-babc-8d8c459f15fb">

