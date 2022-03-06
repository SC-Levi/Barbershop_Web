
## 项目介绍
### 一键换发 (Barbershop) 
基于 StyleGAN2 的发型编辑项目，用户上传自拍照，实时推理更换发型。


### 文件结构
```
├── LICENSE
├── README.md
├── align_face.py # 提取人脸信息 && 图像预处理
├── bash.sh      # 安装必要库 && 加载预训练模型 && 测试
├── datasets     # 数据预处理
│   ├── __init__.py
│   └── image_dataset.py # 加载图像路径 && 处理文件格式
├── input
│   ├── face     # unprocessed文件预处理后存放路径
├── losses       # 损失函数
├── main.py      # 测试
├── models       # 模型
├── pretrained_models # 预训练模型
├── unprocessed  # 自拍照存放路径
│   └── 90.jpg
└── utils       
│       ├── __pycache__ 
│       ├── drive.py        # 路径加载
│       ├── data_utils.py   # 数据预处理
│       ├── model_utils.py  # 模型加载
│       ├── PCA_utils.py    # 主成分分析
│       ├── shape_predictor.py # 轮廓预测
│       ├── seg_utils.py    # 分割
│       ├── bicubic.py      # 图像插值
│       └── image_utils.py  # 发型合成
```

## 部署方法

**Step 1**: 点击项目页面中的"部署" 
<div align="center">
<img width="60%" src="https://minio.platform.oneflow.cloud/media/upload/faa4ad69bd054e95b57028b0c166924e.png" alt="Step 1">
</div>

**Step 2**: 选择模型文件，选中所有文件
<div align="center">
<img width="60%" src="https://minio.platform.oneflow.cloud/media/upload/c96ccf6abf134ea3bfbaa638f96d1c59.png" alt="Step 2">
</div>

**Step 3**: 填写基本信息
<div align="center">
<img width="60%" src="https://minio.platform.oneflow.cloud/media/upload/850ed1ef61914815a5ac7b300b2b1a3f.png" alt="Step 3">
</div>

**Step 4**: 填写配置信息
- "工作环境"选择"公开环境"中的 "oneflow-master+torch-1.9.1-cu11.1-cudnn8"
- "启动命令行"填写为 `cd /workspace && bash bash.sh && python align_face.py && python main.py --im_path1 90.png --im_path2 15.png --im_path3 117.png --sign fidelity --smooth 5`

<div align="center">
<img width="60%" src="https://minio.platform.oneflow.cloud/media/upload/287cf4ffc6bd4fb58c21760c6c482d69.png" alt="Step 4">
</div>

**Step 5**: 选择运行环境
<div align="center">
<img width="60%" src="https://minio.platform.oneflow.cloud/media/upload/f73aeb0c79af444eaae6f45e18e36475.png">
</div>

### 在服务器端进行推理
本项目同时支持在服务器端进行推理，具体用法如下所示：

### 准备
在项目根目录下执行：
```
pip install scikit-image gdown==3.10.1 ninja==1.10.2.3 dlib==19.23.0
```
安装 Python 依赖。


### 推理
对文件夹目录图片作为模型推理的输入。

**图像预处理**

在项目根目录下执行：
```
python align_face.py
```

**运行主函数**

在项目根目录下执行：
```
python main.py --im_path1 <自拍名称.png> \
	       --im_path2 <目标发型.png> \ 
	       --im_path3 <发色及风格.png> --sign fidelity --smooth 5
```

## 参考

- [Barbershop](https://github.com/ZPdesu/Barbershop)
