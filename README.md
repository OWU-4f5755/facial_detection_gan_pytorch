# Advanced Facial Image Morphing with Latent Space Interpolation using StyleGAN2 and Dlib ResNET

## Summary
Developed a comprehensive end-to-end pipeline that processes input images, utilizes a pretrained StyleGAN2 model to project them into the latent space, and generates a smooth morphing video between facial images through latent space interpolation.

## Key Features
1. Utilized the state-of-the-art StyleGAN2 generative model to produce realistic and high-quality face morphing videos.
2. Automated facial landmark detection and cropping using a well-established, pretrained neural network library, enhancing input image compatibility with the StyleGAN2 model.
3. Implemented a seamless pipeline that can be executed on Google Colab, offering an accessible platform with GPU support for widespread use.

## Possible Future Directions
1. Upgrade to dlib's 68-point facial landmark model for better alignment of the input images and more visually aesthetic morphing videos.
2. Local implementation of this pipeline with MPS (Apple Metal) acceleration vs. using Google Colab's cloud platform.
