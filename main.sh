#!/bin/zsh

# Activate conda environment
conda activate facial_detection_gan_pytorch

# Download and extract shape_predictor_5_face_landmarks.dat
wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
bzip2 -d shape_predictor_5_face_landmarks.dat.bz2

# Clone StyleGAN2-ada-pytorch repository and install requirements
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
pip install ninja

# Run main.py
python main.py
