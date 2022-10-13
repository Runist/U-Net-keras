# Keras  - U-Net

## Part 1. Introduction

U-Net is pixel level classification, and the output is the category of each pixel, and different categories of pixels will display different colors. U-Net  is often used in biomedical images, and the image data in this task is often small. Therefore, Ciresan et al. trained a convolutional neural network to predict the class tag of each pixel using the sliding window to provide the surrounding area (patch) of the pixel as input.

### Models for this repository

I didn't find any pretrain weights. It's also different to other network like FCN, SegNet, doesn't have familiar backbone. U-Net Paper said it used [isbi](http://brainiac2.mit.edu/isbi_challenge/) dataset, but I didn't complete reproduce U-Net which has more suitable for medical image padding manipulation. You can think of it as a simple U-Net. So I just random initialize weights to train in VOC dataset.

I still implement *ISBIdataset.py*. It has **elastic_transform** function which is a data augmentation for  cell membranes described in this paper.

| Dataset                | MIoU   | Pixel accuracy |
| ---------------------- | ------ | -------------- |
| VOC train dataset      | 0.9141 | 0.9837         |
| VOC validation dataset | 0.4472 | 0.8939         |



## Part 2. Quick  Start

1. Pull this repository.

```shell
git clone https://github.com/Runist/U-Net-keras.git
```

2. You need to install some dependency package.

```shell
cd U-Net-keras
pip install -r requirements.txt
```

3. Download the *[VOC](https://www.kaggle.com/huanghanchina/pascal-voc-2012)* dataset(VOC [SegmetationClassAug](http://home.bharathh.info/pubs/codes/SBD/download.html) if you need).  
4. Getting U-Net weights.

```shell
wget https://github.com/Runist/U-Net-keras/releases/download/v0.1/unet_weights.h5
```

5. Run **predict.py**, you'll see the result of U-Net.

```shell
python predict.py
```

Input image:

![2007_000129.jpg](https://i.loli.net/2021/06/30/wetEJVlFqZ9digL.jpg)

Output image（resize to 320 x 320）:

![unet.jpg](https://i.loli.net/2021/06/30/pRtF5TldJcUf9Dn.jpg)

## Part 3. Train your own dataset
1. You should rewrite your data pipeline, *Dateset* where in *dataset.py* is the base class, such as  *VOCdataset.py*. 

```python
class VOCDataset(Dataset):
    def __init__(self, annotation_path, batch_size=4, target_size=(320, 320), num_classes=21, aug=False):
        super().__init__(target_size, num_classes)
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.annotation_path = annotation_path
        self.aug = aug
        self.read_voc_annotation()
        self.set_image_info()
```

2. Start training.

```shell
python train.py
```

3. Running *evaluate.py* to get mean iou and pixel accuracy.

```shell
python evaluate.py
--------------------------------------------------------------------------------
Total MIOU: 0.4472
Object MIOU: 0.4243
pixel acc: 0.8939
IOU:  [0.90583348 0.61631588 0.46176812 0.32607745 0.20159544 0.33146206
 0.73923587 0.61883991 0.61589186 0.22619494 0.26843258 0.28243826
 0.47725354 0.40103127 0.50151519 0.6797911  0.20051836 0.33062457
 0.28617971 0.49376667 0.42613842]
```
4. If you can't connect isbi website, I also provide **isbi** dataset.

```shell
wget https://github.com/Runist/U-Net-keras/releases/download/v0.1/membrane.zip
```

   

## Part 4. Reference and other implement

-  [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

- [zhixuhao](https://github.com/zhixuhao)/[unet](https://github.com/zhixuhao/unet)

  
