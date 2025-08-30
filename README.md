开放词汇视频异常检测 (OVVAD)

该项目实现了一种基于开放词汇的视频异常检测系统（OVVAD），利用CLIP模型进行语义理解，并通过时序适配器（Temporal Adapter）提升视频序列中的异常检测性能。

特性

时序适配器（TA）：通过建模视频帧之间的时序依赖，增强视频特征表示，这对于视频异常检测至关重要。

语义知识注入（SKI）：将CLIP模型的文本特征注入到视频特征中，使得模型能够更好地理解正常和异常视频模式。

NAS模块：通过调整特征和相应提示生成新的（novel）异常样本，以适应新类别。

自定义数据集和数据增强：包含一个数据集类，处理视频特征加载、填充和类别分配，同时提供时序异常注释用于训练和评估。

灵活的训练和评估：支持在UCF-Crime数据集上进行模型的训练和评估，包含正常类和异常类的视频。

安装

确保您使用的是Python 3.7+版本，并通过pip或conda安装所需的依赖。

pip install -r requirements.txt


依赖项：

PyTorch

HuggingFace Transformers（用于CLIP模型）

NumPy

SciPy

OpenCV（可选，用于视频处理）

SymPy

数据准备
UCF-Crime数据集

为了运行此项目，您需要UCF-Crime数据集，这是一个广泛使用的视频异常检测数据集。您可以从UCF-Crime官网
下载。

下载后，您需要按照以下结构组织数据集：

UCF_Crimes/
  ├── Videos/
  ├── Features/
  └── Anomaly_Detection_splits/


将特征文件（通常为.npy格式）和视频文件放置在相应的目录下。特征应当已经从每个视频中提取出来。

CLIP模型

您需要使用OpenAI预训练的CLIP模型。可以通过HuggingFace的Transformers库来下载和加载该模型：

from transformers import CLIPModel, CLIPProcessor

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

时序异常注释

在训练时，确保您的数据集包含时序异常注释，通常用于帧级异常检测。

使用方法
训练模型

要在您的数据集上训练模型，请使用以下命令：

python ucf_train.py


该脚本将：

加载视频特征和注释。

使用时序适配器（TA）和语义知识注入（SKI）模块训练OVVAD模型。

使用NAS模块生成合成的异常样本。

使用Adam优化器优化模型。

测试模型

训练完成后，您可以使用以下命令进行评估：

python ucf_test_test.py


该脚本将：

加载训练好的模型。

在测试视频上进行异常检测。

计算基于AUC和AP的评估指标，评估模型在正常类和新异常类上的表现。

模型输出

模型将输出帧级的异常检测结果，并计算基于AUC和AP的整体和每个类别（基类和新异常类）指标。

文件结构
/mnt/data/
  ├── model.py           # 主模型定义（OVVAD、时序适配器、SKI、NAS模块）
  ├── ucf_train.py       # UCF-Crime数据集训练脚本
  └── ucf_test_test.py   # 测试和评估脚本
