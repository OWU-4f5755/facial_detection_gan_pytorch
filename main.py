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

# # Set working directory
# project_root = "/Users/oscarwu_admin_1.0/repos/facial_detection_gan_pytorch"
# os.chdir(project_root)

# # Update sys.path
# sys.path.insert(0, os.path.join(project_root, "stylegan2-ada-pytorch"))

# # Constants
# NETWORK = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
# STEPS = 150
# FPS = 30
# FREEZE_STEPS = 30

# # Read image files
# img_list = []
# for file in os.listdir(project_root):
#     if file.endswith((".jpg", ".jpeg", ".png", ".bmp")):
#         img_list.append(file)

# if len(img_list) < 2:
#     print("Upload at least 2 images for morphing.")

# import cv2
# import numpy as np
# from PIL import Image
# import dlib
# from matplotlib import pyplot as plt

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

# def crop_stylegan(img):
#     dets = detector(img, 1)
#     if len(dets) == 0:
#         raise ValueError("No face detected")

#     d = dets[0]
#     shape = predictor(img, d)

#     x1, y1 = shape.part(0).x, shape.part(0).y
#     x2, y2 = shape.part(2).x, shape.part(2).y
#     x3, y3 = shape.part(4).x, shape.part(4).y

#     center = dlib.point((x1 + x2) // 2, (y1 + y2) // 2)
#     width = np.linalg.norm(np.array([x1, y1]) - np.array([x3, y3]))

#     size = int(width * 2.2)
#     half_size = size // 2
#     left, top = center.x - half_size, center.y - half_size
#     right, bottom = left + size, top + size

#     cropped_img = img[top:bottom, left:right]
#     cropped_img = cv2.resize(cropped_img, (1024, 1024))
#     return cropped_img

# def process_images(img_list):
#     cropped_images = []
#     for img_name in img_list:
#         img = cv2.imread(img_name)
#         if img is None:
#             raise ValueError(f"{img_name} not found")

#         cropped_img = crop_stylegan(img)
#         cropped_images.append(cropped_img)

#         img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
#         plt.imshow(img_rgb)
#         plt.title(f'cropped {img_name}')
#         plt.show()
#     return cropped_images

# cropped_images = process_images(img_list)

# # Generate GAN images and latent vectors for each input image
# gan_images = []
# latent_vectors = []

# # to execute shell commands
# import subprocess
# subprocess.run(cmd, shell=True, check=True)

# # for i, cropped_img in enumerate(cropped_images):
# #     cv2.imwrite(f"cropped_{i}.png", cropped_img)

# #     # HIDE OUTPUT
# #     cmd = f"python /content/stylegan2-ada-pytorch/projector.py "\
# #       f"--save-video 0 --num-steps 1000 --outdir=out_{i} "\
# #       f"--target=cropped_{i}.png --network={NETWORK}"
# #     !{cmd} # cannot use in python script

# #     img_gan = cv2.imread(f'/content/out_{i}/proj.png')
# #     img_rgb = cv2.cvtColor(img_gan, cv2.COLOR_BGR2RGB)
# #     plt.imshow(img_rgb)
# #     plt.title(f'gan-image-{i}')
# #     plt.show()

# #     latent_vector = np.load(f'/content/out_{i}/projected_w.npz')['w']
# #     gan_images.append(img_gan)
# #     latent_vectors.append(latent_vector)

# for i, cropped_img in enumerate(cropped_images):
#     cv2.imwrite(f"cropped_{i}.png", cropped_img)

#     cmd = f"python {project_root}/stylegan2-ada-pytorch/projector.py "\
#           f"--save-video 0 --num-steps 1000 --outdir=out_{i} "\
#           f"--target=cropped_{i}.png --network={NETWORK}"
#     subprocess.run(cmd, shell=True, check=True)

#     img_gan = cv2.imread(f'{project_root}/out_{i}/proj.png')
#     img_rgb = cv2.cvtColor(img_gan, cv2.COLOR_BGR2RGB)
#     plt.imshow(img_rgb)
#     plt.title(f'gan-image-{i}')
#     plt.show()

#     latent_vector = np.load(f'{project_root}/out_{i}/projected_w.npz')['w']
#     gan_images.append(img_gan)
#     latent_vectors.append(latent_vector)

# # Create morph video with all images
# import torch
# import dnnlib
# import legacy
# import PIL.Image
# import numpy as np
# import imageio
# from tqdm.notebook import tqdm

# network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2"\
#   "-ada-pytorch/pretrained/ffhq.pkl"
# device = torch.device('cuda')
# with dnnlib.util.open_url(network_pkl) as fp:
#     G = legacy.load_network_pkl(fp)['G_ema']\
#       .requires_grad_(False).to(device)

# video = imageio.get_writer('/content/movie.mp4', mode='I', fps=FPS, codec='libx264', bitrate='16M')

# for idx in range(len(latent_vectors) - 1):
#     lvec1 = latent_vectors[idx]
#     lvec2 = latent_vectors[idx + 1]

#     diff = lvec2 - lvec1
#     step = diff / STEPS
#     current = lvec1.copy()

#     for j in tqdm(range(STEPS)):
#         z = torch.from_numpy(current).to(device)
#         synth_image = G.synthesis(z, noise_mode='const')
#         synth_image = (synth_image + 1) * (255/2)
#         synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255)\
#           .to(torch.uint8)[0].cpu().numpy()

#         repeat = FREEZE_STEPS if j == 0 or j == (STEPS - 1) else 1
#         for i in range(repeat):
#             video.append_data(synth_image)
#         current = current + step

# video.close()

# # With the following lines:
# output_file = "morph_movie.mp4"
# print(f"Generated output file: {output_file}")


# from google.colab import drive
# drive.mount('/content/drive')

# stylegan2 yielded better results than stylegan3 for feature vectors of selfies, so I'll use v2
# 150 imgs for imgs b/w 2 imgs uploaded
# 30 images @ beginning and end since otherwise it's just jumping sequences

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

# # HIDE OUTPUT
# # 5 facial landmark predictor - base of mouth and nose
# !wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
# !bzip2 -d shape_predictor_5_face_landmarks.dat.bz2

# # HIDE OUTPUT
# import sys
# !git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
# !pip install ninja
# sys.path.insert(0, "/content/stylegan2-ada-pytorch")

# import cv2
# import numpy as np
# from PIL import Image
# import dlib
# from matplotlib import pyplot as plt

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

# def crop_stylegan(img):
#     dets = detector(img, 1)
#     if len(dets) == 0:
#         raise ValueError("No face detected")

#     d = dets[0]
#     shape = predictor(img, d)

#     x1, y1 = shape.part(0).x, shape.part(0).y
#     x2, y2 = shape.part(2).x, shape.part(2).y
#     x3, y3 = shape.part(4).x, shape.part(4).y

#     center = dlib.point((x1 + x2) // 2, (y1 + y2) // 2)
#     width = np.linalg.norm(np.array([x1, y1]) - np.array([x3, y3]))

#     size = int(width * 2.2)
#     half_size = size // 2
#     left, top = center.x - half_size, center.y - half_size
#     right, bottom = left + size, top + size

#     cropped_img = img[top:bottom, left:right]
#     cropped_img = cv2.resize(cropped_img, (1024, 1024))
#     return cropped_img

# def process_images(img_list):
#     cropped_images = []
#     for img_name in img_list:
#         img = cv2.imread(img_name)
#         if img is None:
#             raise ValueError(f"{img_name} not found")

#         cropped_img = crop_stylegan(img)
#         cropped_images.append(cropped_img)

#         img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
#         plt.imshow(img_rgb)
#         plt.title(f'cropped {img_name}')
#         plt.show()
#     return cropped_images

# cropped_images = process_images(img_list)

# # Generate GAN images and latent vectors for each input image
# gan_images = []
# latent_vectors = []

# for i, cropped_img in enumerate(cropped_images):
#     cv2.imwrite(f"cropped_{i}.png", cropped_img)

#     # HIDE OUTPUT
#     cmd = f"python /content/stylegan2-ada-pytorch/projector.py "\
#       f"--save-video 0 --num-steps 1000 --outdir=out_{i} "\
#       f"--target=cropped_{i}.png --network={NETWORK}"
#     !{cmd}

#     img_gan = cv2.imread(f'/content/out_{i}/proj.png')
#     img_rgb = cv2.cvtColor(img_gan, cv2.COLOR_BGR2RGB)
#     plt.imshow(img_rgb)
#     plt.title(f'gan-image-{i}')
#     plt.show()

#     latent_vector = np.load(f'/content/out_{i}/projected_w.npz')['w']
#     gan_images.append(img_gan)
#     latent_vectors.append(latent_vector)

# # Create morph video with all images
# import torch
# import dnnlib
# import legacy
# import PIL.Image
# import numpy as np
# import imageio
# from tqdm.notebook import tqdm

# network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2"\
#   "-ada-pytorch/pretrained/ffhq.pkl"
# device = torch.device('cuda')
# with dnnlib.util.open_url(network_pkl) as fp:
#     G = legacy.load_network_pkl(fp)['G_ema']\
#       .requires_grad_(False).to(device)

# video = imageio.get_writer('/content/movie.mp4', mode='I', fps=FPS, codec='libx264', bitrate='16M')

# for idx in range(len(latent_vectors) - 1):
#     lvec1 = latent_vectors[idx]
#     lvec2 = latent_vectors[idx + 1]

#     diff = lvec2 - lvec1
#     step = diff / STEPS
#     current = lvec1.copy()

#     for j in tqdm(range(STEPS)):
#         z = torch.from_numpy(current).to(device)
#         synth_image = G.synthesis(z, noise_mode='const')
#         synth_image = (synth_image + 1) * (255/2)
#         synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255)\
#           .to(torch.uint8)[0].cpu().numpy()

#         repeat = FREEZE_STEPS if j == 0 or j == (STEPS - 1) else 1
#         for i in range(repeat):
#             video.append_data(synth_image)
#         current = current + step

# video.close()

# # HIDE OUTPUT
# from google.colab import files
# files.download("morph_movie.mp4")
