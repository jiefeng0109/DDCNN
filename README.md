#DDCNN: Divide-and-Conquer Dual-Architecture Convolutional Neural Network for Classification of Hyperspectral Images
It contains a Tensorflow implementation of Divide-and-Conquer Dual-Architecture Convolutional Neural Network (DDCNN) for hyperspectral image classification. 
If you find this code useful in your research, please consider citing the following paper:  
Jie Feng, Lin Wang, Haipeng Yu, Licheng Jiao, Xiangrong Zhang. [Divide-and-conquer dual-architecture convolutional neural network for classification of hyperspectral images](https://www.mdpi.com/2072-4292/11/5/484). Remote Sensing, 2019, 11(5): 484.

##Data process
####Data preprocessing
```
python pre2.py
```
####Dividing homogeneity and heterogeneity
```
python select.py
```
####Data expand
if you want to expand the data
```
python data_expand.py
python select_expand.py
```
##Train and Test
train by homogeneity data
```
python yunzhi.py 
``` 
train by heterogeneity data
```
python bianjie.py
```
obtain result by Integrate homogeneous and heterogeneous results 
```
python vote.py
```
Additional: Suppixels1, SuppixelsP, SuppixelsS are the results of Indian Pines, PaviaU and 
Salinas superpixels that need to be imported. The number of superpixels is (100, 1000, 100). 
If you need to change, please create it by yourself. 
