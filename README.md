# Faster-RCNN
single object detection
## Faster-Rcnn：Two-Stage object detection based on Pytorch
---

## The environment
Anaconda
python==3.7.16
CUDA 11.6
pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6

## 以训练好的权值文件（ep190-loss0.435-val_loss0.361.pth）下载链接：
   https://drive.google.com/file/d/1c1WdJbnbysHz0cPa79W0gVq7snmpI6io/view?usp=share_link
   
## 可以在Pytorch下载机器对应的版本。网址：https://pytorch.org/get-started/locally/

## 训练步骤

### 训练自己建立的数据集
1. 数据集的准备  
训练前需要自己制作好数据集，包括图片resize, 真实框的标注（LabelImg工具）等。
训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。   
训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。
图片文件放置在压缩文件JEPGImages中，解压后将2000张图片拷贝至VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages里面。

2. 数据集的处理  
在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt，这两个txt文件会直接保存在主文件夹中。   
修改voc_annotation.py里面的参数，即仅需要修改classes_path，classes_path用于指向检测类别所对应的txt。   
训练自己的数据集时，自己建立一个cls_classes.txt，里面写自己所需要区分的类别。
本次训练只有一个类别，类别名称为：Anomaly
此时model_data/cls_classes.txt文件内容为：      
```
Anomaly

```
修改voc_annotation.py中的classes_path，使其对应cls_classes.txt，并运行voc_annotation.py。 会自动划分数据集，包括训练集，验证集和测试集。

3. 开始网络训练  
**训练的参数较多，均在train.py中，大家可以在下载库后仔细查看注释，其中最重要的部分依然是train.py里的classes_path，一定要对应cls_classes.txt。 
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会保存在logs文件夹中。

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是frcnn.py和predict.py。在frcnn.py里面修改model_path以及classes_path。  
**model_path指向训练好的权值文件，在logs文件夹里。  
classes_path指向检测类别所对应的txt。**  
完成修改后就可以运行predict.py进行检测。运行后输入图片路径即可检测。  

## 预测步骤
### 使用自己训练的权重
1. 按照训练步骤训练。  
2. 在frcnn.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"    : 'logs/ep190-loss0.435-val_loss0.361.pth', （这个权重文件是训练好的保存在logs文件夹下的。）
    "classes_path"  : 'model_data/cls_classes.txt',
    #---------------------------------------------------------------------#
    #   网络的主干特征提取网络，resnet50或者vgg
    #---------------------------------------------------------------------#
    "backbone"      : "resnet50",
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"    : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"       : 0.3,
    #---------------------------------------------------------------------#
    #   用于指定先验框的大小
    #---------------------------------------------------------------------#
    'anchors_size'  : [4, 16, 32],
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"          : True,
}
```
3. 运行predict.py，输入  
```
img/1.jpg

```
4. 在predict.py里面进行设置还可以进行fps测试和video视频检测。  

## 评估步骤 
### 评估自己的数据集
1. 本文使用VOC格式进行评估。  
2. 如果在训练前已经运行过voc_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。如果想要修改测试集的比例，可以修改voc_annotation.py文件下的trainval_percent。
trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。
3. 利用voc_annotation.py划分测试集后，前往get_map.py文件修改classes_path，classes_path用于指向检测类别所对应的txt，这个txt和训练时的txt一样。评估自己的数据集必须要修改。
4. 在frcnn.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
5. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

#### 我的笔记本配置，two GPUs (CPU Intel(R) Iris(R) Xe Graphics @2.70GHz, RAM: 16GB and GPU: Nvidia GeForce RTX 3060 Laptop × 2ea)。
#### 当batch_size=2,epoch=200 的情况下，训练一次大概需要840分钟。一个GPU的训练时间会更长。建议使用GPU训练，CPU训练速度特别的慢。

## References
https://github.com/chenyuntc/simple-faster-rcnn-pytorch  
https://github.com/eriklindernoren/PyTorch-YOLOv3  
https://github.com/BobLiu20/YOLOv3_PyTorch  
https://github.com/bubbliiiing/faster-rcnn-pytorch
