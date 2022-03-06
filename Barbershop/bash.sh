#! /usr/bin/env bash
pip install Flask==2.0.3
mkdir /root/.cache/torch
mkdir /root/.cache/torch/hub
mkdir /root/.cache/torch/hub/checkpoints
#! 拷贝预训练模型
cp pretrained_models/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
cp pretrained_models/resnet18-5c106cde.pth /root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
#! 图像预处理
python align_face.py
#! 运行主文件,进行图像处理
python barbershop.py --im_path1 test.png --im_path2 25.png --im_path3 117.png --sign fidelity --smooth 5
sleep 1d


