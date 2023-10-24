# %%
import sys
import platform
import torch
import pandas as pd
import sklearn as sk

has_gpu = torch.cuda.is_available()
has_mps = getattr(torch, 'has_mps', False)
device = "mps" if has_mps else "gpu" if has_gpu else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}\n")
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print(f"GPU is {'available' if has_gpu else 'NOT AVAILABLE'}")
print(f"MPS (Apple Metal) is {'AVAILABLE' if has_mps else 'NOT AVAILABLE'}")
print(f"Target device is {device}")

# %%bash
# # No need to activate conda environment here
# # conda activate facial_detection_gan_pytorch

# # Download and extract shape_predictor_5_face_landmarks.dat
# wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
# bzip2 -d shape_predictor_5_face_landmarks.dat.bz2

# # Clone StyleGAN2-ada-pytorch repository and install requirements
# git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
# pip install ninja

# %%
# %%bash

# echo "Current working directory:"
# pwd

# echo "Listing files in the current working directory:"
# ls

# echo "Listing files in the StyleGAN2-ada-pytorch directory:"
# ls stylegan2-ada-pytorch

# echo "Finding the location of the ninja package:"
# pip show ninja

# %%
# import os
# import sys
# import cv2
# import numpy as np
# from PIL import Image
# import dlib
# from matplotlib import pyplot as plt
# import torch
# import dnnlib
# import legacy
# import PIL.Image
# import numpy as np
# import imageio
# from tqdm import tqdm

import os
import sys

# Set working directory
# project_root = "/Users/oscarwu_admin_1.0/repos/facial_detection_gan_pytorch"
project_root = "./"
os.chdir(project_root)

# Update sys.path
sys.path.insert(0, os.path.join(project_root, "stylegan2-ada-pytorch"))


# %%

# Constants
NETWORK = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
STEPS = 150
FPS = 30
FREEZE_STEPS = 30

img_list = [
    file
    for file in os.listdir(project_root)
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp"))
]
if len(img_list) < 2:
    print("Upload at least 2 images for morphing.")

# %%
if img_list:
    print("The images have been read.")

# %%
# !pwd

# %%
# # stylegan2 yielded better results than stylegan3 for feature vectors of selfies, so I'll use v2
# # 150 imgs for imgs b/w 2 imgs uploaded
# # 30 images @ beginning and end since otherwise it's just jumping sequences

# NETWORK = "https://nvlabs-fi-cdn.nvidia.com/"\
#   "stylegan2-ada-pytorch/pretrained/ffhq.pkl"
# STEPS = 150
# FPS = 30
# FREEZE_STEPS = 30

# # HIDE OUTPUT
# import os
# from google.colab import files

# uploaded_files = files.upload()

# img_list = []
# for k, v in uploaded_files.items():
#     _, ext = os.path.splitext(k)
#     os.remove(k)
#     image_name = f"{k}{ext}"
#     open(image_name, 'wb').write(v)
#     img_list.append(image_name)

# if len(img_list) < 2:
#   print("Upload at least 2 images for morphing.")

# %%
import sys

print("Current sys.path:")
print(sys.path)

# %%
# HIDE OUTPUT
# 5 facial landmark predictor - base of mouth and nose
# !wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
# !bzip2 -d shape_predictor_5_face_landmarks.dat.bz2

# # HIDE OUTPUT
# import sys
# !git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
# !pip install ninja
# sys.path.insert(0, "/content/stylegan2-ada-pytorch")

stylegan2_ada_pytorch_dir = os.path.join("stylegan2-ada-pytorch")
sys.path.insert(0, stylegan2_ada_pytorch_dir)

print("Updated sys.path:")
print(sys.path)


# %%
sys.path.insert(0, "/Users/oscarwu_admin_1.0/repos/facial_detection_gan_pytorch/notebooks/stylegan2-ada-pytorch")

print("Updated sys.path:")
print(sys.path)

# %%
print(sys.path)

# %%
import sys
import os

project_directory = "/Users/oscarwu_admin_1.0/repos/facial_detection_gan_pytorch/notebooks"

# Insert the stylegan2-ada-pytorch directory
sys.path.insert(0, os.path.join(project_directory, "stylegan2-ada-pytorch"))

# Insert the project_directory
sys.path.insert(1, project_directory)

print(sys.path)

# %%
# import cv2
# import numpy as np
# from PIL import Image
# import dlib
# from matplotlib import pyplot as plt
import cv2 # !!! Why is this not importing?
import numpy as np
from PIL import Image
import dlib
from matplotlib import pyplot as plt
import torch
import dnnlib
import legacy
import PIL.Image
import numpy as np
import imageio
from tqdm import tqdm

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

# %%

def crop_stylegan(img):
    dets = detector(img, 1)
    if len(dets) == 0:
        raise ValueError("No face detected")

    d = dets[0]
    shape = predictor(img, d)

    x1, y1 = shape.part(0).x, shape.part(0).y
    x2, y2 = shape.part(2).x, shape.part(2).y
    x3, y3 = shape.part(4).x, shape.part(4).y

    center = dlib.point((x1 + x2) // 2, (y1 + y2) // 2)
    width = np.linalg.norm(np.array([x1, y1]) - np.array([x3, y3]))

    size = int(width * 2.2)
    half_size = size // 2
    left, top = center.x - half_size, center.y - half_size
    right, bottom = left + size, top + size

    cropped_img = img[top:bottom, left:right]
    return cv2.resize(cropped_img, (1024, 1024))

def process_images(img_list):
    cropped_images = []
    for img_name in img_list:
        img = cv2.imread(img_name)
        if img is None:
            raise ValueError(f"{img_name} not found")

        cropped_img = crop_stylegan(img)
        cropped_images.append(cropped_img)

        img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f'cropped {img_name}')
        plt.show()
    return cropped_images

cropped_images = process_images(img_list)

# Generate GAN images and latent vectors for each input image
gan_images = []
latent_vectors = []

for i, cropped_img in enumerate(cropped_images):
    cv2.imwrite(f"cropped_{i}.png", cropped_img)

    # HIDE OUTPUT
    cmd = f"python /content/stylegan2-ada-pytorch/projector.py "\
      f"--save-video 0 --num-steps 1000 --outdir=out_{i} "\
      f"--target=cropped_{i}.png --network={NETWORK}"
    !{cmd}

    img_gan = cv2.imread(f'/content/out_{i}/proj.png')
    img_rgb = cv2.cvtColor(img_gan, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f'gan-image-{i}')
    plt.show()

    latent_vector = np.load(f'/content/out_{i}/projected_w.npz')['w']
    gan_images.append(img_gan)
    latent_vectors.append(latent_vector)

# Create morph video with all images
import torch
import dnnlib
import legacy
import PIL.Image
import numpy as np
import imageio
from tqdm.notebook import tqdm

network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2"\
  "-ada-pytorch/pretrained/ffhq.pkl"
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as fp:
    G = legacy.load_network_pkl(fp)['G_ema']\
      .requires_grad_(False).to(device)

video = imageio.get_writer('/content/movie.mp4', mode='I', fps=FPS, codec='libx264', bitrate='16M')

for idx in range(len(latent_vectors) - 1):
    lvec1 = latent_vectors[idx]
    lvec2 = latent_vectors[idx + 1]

    diff = lvec2 - lvec1
    step = diff / STEPS
    current = lvec1.copy()

    for j in tqdm(range(STEPS)):
        z = torch.from_numpy(current).to(device)
        synth_image = G.synthesis(z, noise_mode='const')
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255)\
          .to(torch.uint8)[0].cpu().numpy()

        repeat = FREEZE_STEPS if j == 0 or j == (STEPS - 1) else 1
        for i in range(repeat):
            video.append_data(synth_image)
        current = current + step

video.close()

# HIDE OUTPUT
from google.colab import files
files.download("movie.mp4")
