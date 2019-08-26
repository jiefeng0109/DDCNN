# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:45:24 2017

@author: Administrator
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
import time
import scipy.io as sio
import os
import pandas as pd
import math

dim_out = 16
dropout = [0.5,0.8,1]
batch_size = 100
learn_rate= 0.01
display_step = 100
num_epoch = 5000
#w = [1,3,5,19]
w = 3
dim_input = [150,100,20,10,1]
threshold = 0.001
data_name = 'Indian_pines'
path = os.getcwd()
pre_dir = path + '/data/' + data_name
pre = sio.loadmat(pre_dir + '/' + data_name + '_pre.mat')
data_norm = pre['data_norm']
labels_ori = pre['labels_ori']
x_train = pre['train_x']
y_train = pre['train_y'][0]
train_loc = pre['train_loc']
x_test = pre['test_x']
y_test = pre['test_y'][0]
test_loc = pre['test_loc']
num_list = pre['train_num_list'][0]
cumlist = pre['cumlist'][0]
sp1 = sio.loadmat(path+'/superpixels1.mat')['labels']

def gausdis(vector1,vector2):
    sigma = 0.1
    dis = np.linalg.norm(vector1-vector2)
    gaus_dis = math.exp(-dis*dis/(2*sigma*sigma))
    return gaus_dis
#vote = {}
#label_expand = np.lib.pad(sp1,((1,1),(1,1)),'constant', constant_values=0)
#for i in range(3):
#    for j in range(3):
#        ith_label = label_expand[train_loc[0][1]+i,train_loc[1][1]+j]
#        vote[ith_label] = vote.get(ith_label,0)+1
#sortedvote = sorted(vote.items(),key = lambda x:x[1], reverse = True) 
#ratio = sortedvote[0][1]/9
mingaus_list = []
meanlist = np.zeros((16,200))
for i in range(16):
    start = cumlist[i]
    num_ev = num_list[i]
    mean = np.mean(x_train[start:start+num_ev,:],axis=0)
    meanlist[i,:] = mean
    gaus = []
    for j in range(num_ev):
        gaus_dis = gausdis(x_train[start+j,:],mean)
        gaus.append(gaus_dis)
    minindex = gaus.index(min(gaus))
    print(gaus.index(min(gaus)))
    mingaus = gaus[minindex]
    mingaus_list.append(mingaus)
    
size = np.shape(data_norm)
vote = {}
data_expand = np.zeros((int(size[0]+w-1),int(size[1]+w-1),size[2]))
for j in range(size[2]):    
    data_expand[:,:,j] = np.lib.pad(data_norm[:,:,j], ((int(w / 2), int(w / 2)), (int(w / 2),int(w / 2))), 'symmetric')
count = 0
for i in range(w):
    for j in range(w):
        locx = train_loc[0][1]
        locy = train_loc[1][1]
        meanv = meanlist[0]
        vector = data_expand[locx+i,locy+j,:]
        gaus_dis = gausdis(vector,meanv)
        print(gaus_dis)
        if (gaus_dis>=mingaus_list[0]):
            count+=1
ratio = count/(w*w)
        