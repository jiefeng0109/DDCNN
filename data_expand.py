# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:26:31 2017

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
#data_name = 'KSC'
data_name = 'PaviaU'
#data_name = 'Indian_pines'
#data_name = 'washington'
#data_name = 'Salinas'
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
classes = 9
#sp = sio.loadmat(path+'/superpixels1.mat')['labels']
#data = sio.loadmat(path+'/Indian_pines_corrected.mat')['indian_pines_corrected']
#label = sio.loadmat(path+'/Indian_pines_gt.mat')['indian_pines_gt']

sp = sio.loadmat(path+'/superpixelsP.mat')['labels']
data = sio.loadmat(path+'/PaviaU.mat')['paviaU']
label = sio.loadmat(path+'/PaviaU_gt.mat')['paviaU_gt']

#sp = sio.loadmat(path+'/superpixelsX.mat')['labels']
#data = sio.loadmat(path+'/KSC.mat')['KSC']
#label = sio.loadmat(path+'/KSC_gt.mat')['KSC_gt']

#sp = sio.loadmat(path+'/superpixelsH.mat')['labels']
#data = sio.loadmat(path+'/washington_datax.mat')['washington_datax']
#label = sio.loadmat(path+'/washington_gt.mat')['washington_gt']
#
#sp = sio.loadmat(path+'/superpixelsS.mat')['labels']
#data = sio.loadmat(path+'/Salinas_corrected.mat')['salinas_corrected']
#labels = sio.loadmat(path+'/Salinas_gt.mat')['salinas_gt']

def gausdis(vector1,vector2):
    sigma = 0.1
    dis = np.linalg.norm(vector1-vector2)
    gaus_dis = math.exp(-dis*dis/(2*sigma*sigma))
    return gaus_dis

num = []
cum = 0
cuml = [0]
x_loc1 = []
x_loc2 = []
for i in range(5000):
    loc1, loc2 = np.where(sp == i)
    x_loc1.extend(loc1)
    x_loc2.extend(loc2)
    num.append(len(loc1))
    cum += len(loc1)
    cuml.append(cum)
    x_loc = np.vstack([x_loc1, x_loc2])

superset = []
num_tra_init = len(train_loc[0]) 
mytrain = {}
mytrain1 = {}   
for i in range(num_tra_init):
    loc1 = train_loc[0][i]
    loc2 = train_loc[1][i]
    sign = sp[loc1,loc2]
    mytrain[sign] = mytrain.get(sign,0)+1
    if sign not in superset:
        superset.append(sign)
        mytrain1[sign] = mytrain1.get(sign,[[loc1,loc2]])
    else:
        mytrain1[sign] = np.vstack((mytrain1[sign],[loc1,loc2]))

#def combine(outputList,sortList):  
#    CombineList = list()  
#    for index in range(len(outputList)):  
#        CombineList.append((outputList[index],sortList[index]))  
#    return CombineList
def labelcount(location):
    num = np.shape(location)[0]
    vote = {}
    for i in range(num):
        locx = location[i][0]
        locy = location[i][1]
        ith_label = labels_ori[locx,locy]
        vote[ith_label] = vote.get(ith_label,0)+1
    sortedvote = sorted(vote.items(),key = lambda x:x[1], reverse = True)
    ratio = sortedvote[0][1]/num
    return sortedvote[0][0],ratio
'''
stand_gaus = []
for sign in mytrain1.keys():
    stand_gaus_ev_sp = []
    if mytrain[sign]==1:
        stand_gaus.append(0)
    else:
        for j in range(mytrain[sign]):
            loc1 = mytrain1[sign][j][0]
            loc2 = mytrain1[sign][j][1]
            data_duibi0 = data_norm[loc1,loc2,:]
            for k in range(mytrain[sign]):
                if k>j:
                    loc1 = mytrain1[sign][k][0]
                    loc2 = mytrain1[sign][k][1]
                    data_duibi1 = data_norm[loc1,loc2,:]
                    gaus_dis = gausdis(data_duibi0,data_duibi1)
                    stand_gaus_ev_sp.append(gaus_dis)
        min_gaus = min(stand_gaus_ev_sp)
        stand_gaus.append(min_gaus)

c = []
pos1 = []
pos2 = []
pos = [] 
xb = []  
ith = 0     
for sign in mytrain1.keys():
    count = 0
    count1 = 0
    standard = stand_gaus[ith]
    ith+=1
    for j in range(num[sign]):
#        print(j)
        start = cuml[sign]
        locx = x_loc[0][start+j]
        locy = x_loc[1][start+j]
        data_exe = data_norm[locx,locy,:] 
        gaus_list = []
#        index_list = []
        for k in range(mytrain[sign]):
            loc1 = mytrain1[sign][k][0]
            loc2 = mytrain1[sign][k][1]
            data_duibi = data_norm[loc1,loc2,:]
            gaus_dis = gausdis(data_exe,data_duibi) 
            gaus_list.append(gaus_dis)
#            index_list.append(k)
        dis = sorted(gaus_list,reverse=True)
#        gaus_list.sort(reverse=True)
##        dis = combine(index_list,gaus_list)
##        dis_sorted = dis.sort(key=lambda x:x[1],reverse=True)
        if (dis[0]==1)or(mytrain[sign]==1):
#            print(j)
            count1+=1
            continue
        else:
#            mid = int(mytrain[sign]/2)
            guas_mid = dis[-1]
##        elif (mytrain[sign]>1)and(mytrain[sign]<=5):
            location = mytrain1[sign]
            label_exe,ratio = labelcount(location)            
            if guas_mid>standard:  
               count+=1
               pos1.append(locx)
               pos2.append(locy)
               xb.append(int(label_exe))
               pos = np.vstack([pos1,pos2])
'''
c = []
pos1 = []
pos2 = []
pos = [] 
xb = []       
for sign in mytrain1.keys():
    count = 0
    count1 = 0
    for j in range(num[sign]):
#        print(j)
        start = cuml[sign]
        locx = x_loc[0][start+j]
        locy = x_loc[1][start+j]
        data_exe = data_norm[locx,locy,:] 
        gaus_list = []
        index_list = []
        for k in range(mytrain[sign]):
            loc1 = mytrain1[sign][k][0]
            loc2 = mytrain1[sign][k][1]
            data_duibi = data_norm[loc1,loc2,:]
            gaus_dis = gausdis(data_exe,data_duibi) 
            gaus_list.append(gaus_dis)
            index_list.append(k)
        dis = sorted(gaus_list,reverse=True)
#        gaus_list.sort(reverse=True)
##        dis = combine(index_list,gaus_list)
##        dis_sorted = dis.sort(key=lambda x:x[1],reverse=True)
        if (dis[0]==1)or(mytrain[sign]==1):
#            print(j)
            count1+=1
            continue
        else:
            mid = int(mytrain[sign]/2)
            guas_mid = dis[mid]
##        elif (mytrain[sign]>1)and(mytrain[sign]<=5):
            location = mytrain1[sign]
            label_exe,ratio = labelcount(location)            
            if guas_mid>=0.7:  
               count+=1
               pos1.append(locx)
               pos2.append(locy)
               xb.append(int(label_exe))
               pos = np.vstack([pos1,pos2])        
#    print(sign,count,count1,mid,num[sign])

c = np.zeros((classes))
exenum = np.shape(xb)[0]
for i in range(exenum):
    a = xb[i]
    for j in range(1,classes+1):
        if a==j:
            c[j-1] = c[j-1]+1
print(c) 

#add_train = np.array(xb,dtype=int)
#y_train = np.concatenate((y_train,xb),axis=0)
#train_loc = np.hstack((train_loc,pos)) 
num_stand = 200
xb = np.array(xb)
num_exe = []
cum_exe = 0
cum_exe_list = [0]
lab = []
exe_loc0 = []
exe_loc1 = []
exe_loc =[]
for i in range(1,classes+1):
#    location0,location1 = np.where(xb==i)
    index = [idx for idx, e in enumerate(xb) if e==i]
    if index==[]:
        num_exe.append(0)
        cum_exe = cum_exe+0
        cum_exe_list.append(cum_exe)
        continue
    num_ev_exe = len(index)
    index = np.array(index)
    shuf = np.arange(num_ev_exe)
    np.random.shuffle(shuf)
    index1 = index[shuf]
    if num_ev_exe>=num_stand:
        num_exe.append(num_stand)
        cum_exe = cum_exe+num_stand
        cum_exe_list.append(cum_exe)
        every_lab = xb[index1[:num_stand]]
        lab.extend(every_lab)
        every_pos0 = pos[0,index1[:num_stand]]
        every_pos1 = pos[1,index1[:num_stand]]
        exe_loc0.extend(every_pos0)
        exe_loc1.extend(every_pos1)
        exe_loc = np.vstack([exe_loc0,exe_loc1])
    else:
#        np.random.shuffle(num_ev_exe)
#        index1 = index[num_ev_exe]
        num_exe.append(num_ev_exe)
        cum_exe = cum_exe+num_ev_exe
        cum_exe_list.append(cum_exe)
        every_lab = xb[index1]
        lab.extend(every_lab)
        every_pos0 = pos[0,index1]
        every_pos1 = pos[1,index1]
        exe_loc0.extend(every_pos0)
        exe_loc1.extend(every_pos1)
        exe_loc = np.vstack([exe_loc0,exe_loc1])
sio.savemat(path + '/expand/'+data_name+'/expand_70.mat',{'expand_loc':exe_loc,'expand_label':lab
                                                       ,'cum_exe_list':cum_exe_list,'num_exe':num_exe})


