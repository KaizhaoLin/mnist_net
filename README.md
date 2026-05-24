# MNIST 手写数字识别

这是一个基于 PyTorch 的 MNIST 手写数字识别项目。项目使用一个简化版 LeNet 卷积神经网络完成 `0-9` 十类数字分类，包含数据加载、模型训练、训练日志、曲线可视化、模型保存和单张图片预测流程。

## 项目功能

- 自动下载并加载 MNIST 数据集
- 使用卷积神经网络训练手写数字分类模型
- 保存最佳模型和最终模型权重
- 输出训练日志、损失曲线和准确率曲线
- 支持对本地单张手写数字图片进行预测

## 项目结构

```text
mnist_net/
├── data/                 # MNIST 数据集，运行训练时自动生成
├── figures/              # 训练曲线图
│   ├── acc_curve.png
│   └── loss_curve.png
├── logs/                 # 训练日志
├── saved_models/         # 训练好的模型权重
│   ├── mnist_net_best.pth
│   └── mnist_net_last.pth
├── src/
│   ├── dataloader.py     # 数据加载与数据增强
│   └── model.py          # LeNet 模型定义
├── test_images/          # 用于预测测试的手写数字图片
├── predict.py            # 单张图片预测脚本
├── train.py              # 模型训练脚本
└── README.md
```

## 环境配置

建议使用 Python 3.9。

```bash
conda create -n mnist_net python=3.9
conda activate mnist_net
```

安装 PyTorch、torchvision 和 matplotlib。根据你的 CUDA/CPU 环境选择合适的 PyTorch 安装命令，例如 CPU 版本：

```bash
pip install torch torchvision matplotlib pillow
```

也可以到 PyTorch 官网选择与你设备匹配的安装命令。

## 训练模型

进入项目目录后运行：

```bash
python train.py
```

训练脚本会：

- 使用 `torchvision.datasets.MNIST` 自动下载数据集
- 对训练集进行轻量随机仿射增强
- 训练 30 个 epoch
- 将最佳模型保存到 `saved_models/mnist_net_best.pth`
- 将最终模型保存到 `saved_models/mnist_net_last.pth`
- 将训练日志保存到 `logs/`
- 将损失曲线和准确率曲线保存到 `figures/`

当前已有训练结果中，模型在测试集上的最佳准确率约为 **99.31%**。

## 使用模型预测

可以使用 `predict.py` 对单张图片进行预测：

```bash
python predict.py --image_path test_images/0.png
```

如果想指定模型权重：

```bash
python predict.py --image_path test_images/0.png --model_path saved_models/mnist_net_best.pth
```

脚本会输出使用的设备、图片路径、模型路径、原始 logits，以及最终预测的数字。

## 模型说明

模型定义在 `src/model.py` 中，整体结构为：

```text
输入图像 1x28x28
→ Conv2d(1, 6) + ReLU + MaxPool
→ Conv2d(6, 12) + ReLU + MaxPool
→ Flatten
→ Linear(588, 160) + ReLU
→ Linear(160, 80) + ReLU
→ Linear(80, 10)
```

输出维度为 10，对应数字类别 `0-9`。

## 示例输出

运行预测脚本后，终端会显示类似结果：

```text
Using device: cpu
Image to predict: test_images/0.png
Model file used: saved_models/mnist_net_best.pth

Predicting...
-> The predicted digit is: 0
```

## 备注

- `data/` 目录由训练脚本自动下载生成，通常不需要提交到 GitHub。
- `saved_models/` 中的权重文件较小，可以保留，方便直接运行预测。
- 如果预测自己的图片，建议使用黑白或高对比度数字图片；脚本会自动转灰度、缩放到 `28x28`，并做反色处理。
