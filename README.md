# min-patchnizer

Minimal, clean code for video/image "patchnization" - a process commonly used in tokenizing visual data for use in a Transformer encoder. The code here, first extracts still images (frames) from a video, splits the image frames into smaller fixed-size patches, linearly embeds each of them, adds position embeddings and then saves the resulting sequence of vectors for use in a Vision Transformer encoder. I tried training the resulting sequence vectors with Karpathy's minbpe and it took 2173.45 seconds per frame to tokenize. The whole "patchnization" took ~77.40 for a 20s video on my M2 Air.

![IMG_5672](https://github.com/Jaykef/min-patchnizer/assets/11355002/de2eb521-58d5-4308-b061-19a32217cbb2)
<br><br>

The files in this repo work as follows:

<ul>
  <li><a href="https://github.com/Jaykef/min-patchnizer/blob/main/patchnizer.py">patchnizer.py</a>: Holds code for simple implemenatation of the three stages involved (extract_image_frames from video, reduce image_frames_to_patches of fixed sizes 16x16 pixels, then linearly_embed_patches into a 1D vector sequence with additional position embeddings.</li>
  
  <li><a href="https://github.com/Jaykef/min-patchnizer/blob/main/patchnize.py">patchnize.py</a>: performs the whole process with custom configs (patch_size, created dirs, video - I am using the "dogs playing in snow" video by sora).</li>

  <li><a href="https://github.com/Jaykef/min-patchnizer/blob/main/patchnize.py">train.py</a>: Trains the resulting one-dimensional vector sequence (linear_patch_embeddings + position_embeddings) on Karpathy's minbpe (a minimal implementation of the byte-pair encoding algorithm).</li>

  <li><a href="https://github.com/Jaykef/min-patchnizer/blob/main/patchnize.py">check.py</a>: Checks to see if the patch embeddings match the original image patches and then reconstructs the original image frames - this basically just do the reverse of linear embedding.</li>
</ul>


The whole process builds on the approach introduced in the Vision Transformer paper: <a href="https://arxiv.org/abs/2010.11929">"An image is worth 16x16 words: Transformers for image recognition at scale."</a>

Youtube Video: <a href="https://youtu.be/eT1mJE4J38o?si=9uTeLo6eFoNmbJLt">Watch Demo</a>

## Usage

  1. First patchnize: ``` python patchnize.py```
  
  2. Next check: ``` python check.py``` 
  
  2. Then train: ``` python train.py```

## References
<ul>
  <li><a href="https://openai.com/research/video-generation-models-as-world-simulators">SORA Technical Report</a></li>
  
  <li><a href="https://arxiv.org/abs/2010.11929">"An image is worth 16x16 words: Transformers for image recognition at scale", Alexey Dosovitskiy et al.</a></li>

  <li><a href="https://github.com/karpathy/minbpe#:~:text=/-,minbpe,-Type">minbpe by karpathy</a></li>
</ul>

## License
MIT

