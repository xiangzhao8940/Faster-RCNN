# Faster-RCNN
single object detection
## Faster-Rcnn：Two-Stage object detection based on Pytorch
---

## The environment
Anaconda
python==3.7.16
CUDA 11.6
pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6

## The trained weights file（ep190-loss0.435-val_loss0.361.pth）download link：
   https://drive.google.com/file/d/1c1WdJbnbysHz0cPa79W0gVq7snmpI6io/view?usp=share_link
   
## The corresponding version of the machine can be downloaded at the Pytorch.URL：https://pytorch.org/get-started/locally/

## Training steps

### Train the self-built datasets
1. Preparation of the dataset  
Before training, you need to make your own data set, including picture resize, real box annotation (LabelImg tool), etc.
Place the tag file in the Annotation under the VOC2007 folder under the VOCdevkit folder before training.   
Place the picture file in the JPEGImages under the VOC2007 folder under the VOCdevkit folder before training.
The picture file is placed in the compressed file JEPGImages, and copy 2000 pictures to the JPEGImages under the VOC2007 folder under the VOCdevkit folder.

2. Processing of Datasets 
After completing the placement of the data set, we need to use voc_annotation.py to obtain the 2007_train.txt and 2007_val.txt for training, and the two txt files will be directly saved in the main folder.  
To modify the parameters in the voc_annotation.py, only the modified classes_path is required, and the classes_path is used to point to the txt corresponding to the detection category.  
When training your own data set, build an cls_classes.txt that describes the categories you need to distinguish.
There is only one category in this training session, and the category name is: Anomaly
The content of the model_data/cls_classes.txt file is listed below:     
```
Anomaly

```
Modify the classes_path in the voc_annotation.py to correspond to the cls_classes.txt, and run the voc_annotation.py. Will automatically divide the dataset, including the training set, validation set and test set.

3. Start training  
** There are many training parameters, which are all in train.py. You can carefully review the comments after downloading the library. The most important part is still the classes_path in train.py, which must correspond to cls_classes.txt.
After modifying the classes_path, you can run train.py and start the training. After training multiple epochs, the weights will be saved in the logs folder.

4. Prediction of training results 
Training result prediction requires two files, respectively, frcnn.py and predict.py. Modify the model_path and classes_path in frcnn.py.
** Model _ path points to the trained weight file, in the logs folder. classes_path points to the txt corresponding to the detection category.
** After completing the modification, the predict.py can be run for testing. After running, enter the image path to detect it.

## Prediction steps
### Use your own training weights
1. Follow the training steps to training.  
2. In the frcnn.py file, modify the mode _ path and classes_path to the trained file; * * mode _ path corresponds to the weight file under the logs folder, classes_path is the class of mode _ path**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   Using your own trained model for prediction, you must modify the model_path and classes_path！
    #   The model_path points to the weight file under the logs folder, and the classes_path points to the txt under the model_data
    #   If a shape mismatch occurs, also pay attention to the modification of the model_path and classes _ parameter parameters during training
    #--------------------------------------------------------------------------#
    "model_path"    : 'logs/ep190-loss0.435-val_loss0.361.pth', （This weight file is trained and saved under the logs folder.）
    "classes_path"  : 'model_data/cls_classes.txt',
    #---------------------------------------------------------------------#
    #   Network backbone features of the network extracted network，resnet50 or vgg
    #---------------------------------------------------------------------#
    "backbone"      : "resnet50",
    #---------------------------------------------------------------------#
    #   Only prediction boxes with scores greater than confidence will be retained
    #---------------------------------------------------------------------#
    "confidence"    : 0.5,
    #---------------------------------------------------------------------#
    #   Size of the nms _ iou used for NMS
    #---------------------------------------------------------------------#
    "nms_iou"       : 0.3,
    #---------------------------------------------------------------------#
    #   Use to specify the size of the prior box
    #---------------------------------------------------------------------#
    'anchors_size'  : [4, 16, 32],
    #-------------------------------#
    #   Whether to use CUDA
    #   No GPU can be set to False
    #-------------------------------#
    "cuda"          : True,
}
```
3. Run predict.py, and enter  
```
img/1.jpg

```
4. Settings in predict.py can also perform fps test and video video detection. 

## Evaluation steps 
### Evaluate your own dataset
1. This article uses VOC format for evaluation.  
2. If the voc_annotation.py file has been run before training, the code automatically divides the dataset into the training, validation, and test set. If you want to modify the proportion of the test set, you can modify the trainval_percent under the voc_annotation.py file.
trainval_percent is used to specify the ratio of the (training set validation set) to the test set, by default (training set validation set): test set = 9:1. The train_percent is used to specify the ratio of training set to validation set in (training set, validation set), by default, training set: validation set = 9:1.
3. After dividing the test set with voc_annotation.py, go to the get_map.py file to modify classes_path. classes_path is used to point to the txt corresponding to the detection category, which is the same as the txt at training. The data set that evaluates yourself must be modified.
4. Modify the model_path and classes_path in frcnn.py.* * Model _ path points to the trained weight file, in the logs folder. The classes_path points to the txt corresponding to the detection category.** 
5. Run get_map.py, which is saved in the map_out folder.

#### My laptop configuration，two GPUs (CPU Intel(R) Iris(R) Xe Graphics @2.70GHz, RAM: 16GB and GPU: Nvidia GeForce RTX 3060 Laptop × 2ea)。
#### When batch_size=2 and epoch=200, training takes about 840 minutes at a time. A GPU will train for longer. It is recommended to use GPU training, and the CPU training speed is particularly slow.

## References
https://github.com/chenyuntc/simple-faster-rcnn-pytorch  
https://github.com/eriklindernoren/PyTorch-YOLOv3  
https://github.com/BobLiu20/YOLOv3_PyTorch  
https://github.com/bubbliiiing/faster-rcnn-pytorch
