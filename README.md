# libtorch-yolov3-Tracker
Integrate libtorch-yolov3 with tracking algorithm making this project more engineering.  Just a framework.

Many features and details need to be added as required according to demand.Hope to give some help.

This project is inspired by the [libtorch-yolov3](https://github.com/walktree/libtorch-yolov3).  Thanks to the author.

## Requirements
1. LibTorch v1.0.0+
2. Cuda
3. OpenCV 3.2  + opencv_contrib

## Opencv_contrib
The API below  is used in this tracking  algorithm,   you should know, because there is no such API  in the official code, I added it. 
```
CV_WRAP bool updateKCF( const Mat& image, Rect2d& boundingBox ,double& maxVal);
```
Modified [opencv_contrib](https://pan.baidu.com/s/1c_ngZQ6gMB1VURW0SE13rQ) is here


## Training
Train  yolov3 in  [Darknet](https://pjreddie.com/darknet/yolo/) 

you can download this [weight](https://pan.baidu.com/s/1c_ngZQ6gMB1VURW0SE13rQ) for test.

## Run tracker
modify **labels_name** in main.cpp

```
./yolo-tracker   <your_own_cfg>  <your_own_weights>  <path_to_video>
```

