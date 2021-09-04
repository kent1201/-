# Medical Image Classification
研究所課程 醫學影像專題一  分類

## Introduction

The purpose of this paper is to use deep learning to identify and classify polypoidal choroidal vasculopathy (PCV) [1] and choroidal neovascularization (CNV) [2] in medical images. Polypoid choroidal vascular disease, one of the subspecies of age-related macular degeneration, can cause macular hemorrhage and water accumulation in the retina, and is one of the most important causes of vision loss in the elderly. Choroidal neovascularization is also one of the subspecies of age-related macular degeneration, but the mechanism of retinal hemorrhage caused by choroidal neovascularization is different. At present, the two have not yet recognized the unique characteristic genes, so that people can not distinguish between the two by analyzing the genetic correlation, which is not good for early diagnosis and treatment [3]. At present, the common discrimination method is to identify the detailed differences between the two (for example: blood vessel radius, abnormal branch blood vessel network, etc.) through experienced doctors, which is labor-consuming and time-consuming. In this experiment, we used the four architectures of VGG16, ResNet50, ResNet101, ResNet152, plus transfer learning, and used different optimizers and data augmentation experiments to learn the subtle differences between the two images. ResNet101 + SGD + Transfer learning scheme, and the trained model has a recognition rate of up to 98%. 

## Model architecture
VGG 16

![VGG 16](https://github.com/kent1201/Medical-Image-Classification/blob/main/VGG16.png)

Resnet

![Resnet](https://github.com/kent1201/Medical-Image-Classification/blob/main/Resnet.png)

資料來源: https://blog.csdn.net/zzc15806

## Project architecture

`\cases` 不同實驗比較結果。

`\gradient check` 檢查不同 epoch 下不同 layers 的 gradient 情形。

`\model` 不同 architecture。

`\test` 測試資料集。

`dataset.py` Dtaset based on Pytorch。

`move_file.py` Divide the dataset into train and test.

`test.py` Test the classification model.

`train.py` Train the classification model.

`utils.py` the hyper parameters setting.


