# min-patchnizer

Minimal, clean code for video/image tokenization. extracting patches from video frames, as proposed in Sora technical report: "At a high level, we turn videos into patches by first compressing videos into a lower-dimensional latent space,19 and subsequently decomposing the representation into spacetime patches"


<img width="825" alt="Screenshot 2024-02-29 at 10 33 33" src="https://github.com/Jaykef/sora-patchnizer/assets/11355002/1aa23e7a-56ed-4e31-af4f-79e969734b0d"><br>


The code here, first extracts image frames from a video (.mp4), split the image frames into smaller fixed-size patches, linearly embed each of them, add position embeddings, save the resulting sequence of vectors for use in a Transformer.

The whole process builds on the approach introduced in the Vision Transformer paper: "An image is worth 16x16 words: Transformers for image recognition at scale."

<br><img width="982" alt="Screenshot 2024-02-29 at 10 38 47" src="https://github.com/Jaykef/sora-patchnizer/assets/11355002/61f5a5a3-9cee-45c3-8a7f-6fc3598e9623">
