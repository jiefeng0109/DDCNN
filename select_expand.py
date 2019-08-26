# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 18:23:48 2019

@author: fengjie-win
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 20:20:21 2017

@author: Administrator
"""


import numpy as np


import scipy.io as sio
import os

import math

dim_out = 9
band = 103
#band = 200


w = [1,9,5,29]

threshold = 0.001
#data_name = 'Indian_pines'
data_name = 'PaviaU'
#data_name = 'KSC'
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
#data = sio.loadmat(pre_dir+'/Indian_pines_corrected.mat')['indian_pines_corrected'].astype('float32')
num_list = pre['train_num_list'][0]
cumlist = pre['cumlist'][0]
sp1 = sio.loadmat(path+'/superpixelsP.mat')['labels']
size = np.shape(data_norm)

#y_train = np.concatenate((y_train,expand_label),axis=0)
#train_loc = np.hstack((train_loc,expand_loc)) 

def gausdis(vector1,vector2):
    sigma = 0.1
    dis = np.linalg.norm(vector1-vector2)
    gaus_dis = math.exp(-dis*dis/(2*sigma*sigma))
    return gaus_dis
def get_input(w,loc,dim_input):

    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim_input)
    data_PCA = pca.fit_transform(data_norm.reshape(data_norm.shape[0]*data_norm.shape[1], -1))
    data_PCA = data_PCA.reshape(data_norm.shape[0], data_norm.shape[1],dim_input)

    # 转换数据维度
    from pre2 import windowFeature
    X = windowFeature(data_PCA, loc, w)
    
    return X
    

#
#def neibor_vote(locx,locy,w):    
#    vote = {}
#    label_expand = np.lib.pad(labels_ori,((int(w / 2), int(w / 2)), (int(w / 2),int(w / 2))),'constant', constant_values=0)
#    for i in range(w):
#        for j in range(w):
#            ith_label = label_expand[locx+i,locy+j]
#            vote[ith_label] = vote.get(ith_label,0)+1
#    sortedvote = sorted(vote.items(),key = lambda x:x[1], reverse = True)
#    num_all = int(w*w)
#    ratio = sortedvote[0][1]/num_all
#    return sortedvote[0][0],ratio
mingaus_list = []
meanlist = np.zeros((dim_out ,band))
for i in range(dim_out ):
    start = cumlist[i]
    num_ev = num_list[i]
    mean = np.mean(x_train[start:start+num_ev,:],axis=0)
    meanlist[i,:] = mean
    gaus = []
    for j in range(num_ev):
        gaus_dis = gausdis(x_train[start+j,:],mean)
        gaus.append(gaus_dis)
    minindex = gaus.index(min(gaus))
#    print(gaus.index(min(gaus)))
    mingaus = gaus[minindex]
    mingaus_list.append(mingaus)
    


def neibor_ratio(locx,locy,w,mean,mingaus):
    size = np.shape(data_norm)
    data_expand = np.zeros((int(size[0]+w-1),int(size[1]+w-1),size[2]))
    for j in range(size[2]):    
        data_expand[:,:,j] = np.lib.pad(data_norm[:,:,j], ((int(w / 2), int(w / 2)), (int(w / 2),int(w / 2))), 'symmetric')
    count = 0
    for i in range(w):
        for j in range(w):
#            locx = train_loc[0][1]
#            locy = train_loc[1][1]
#            meanv = meanlist[0]
            vector = data_expand[locx+i,locy+j,:]
            gaus_dis = gausdis(vector,mean)
            if (gaus_dis>=mingaus):
                count+=1
    ratio = count/(w*w)
    return ratio
    
def neibor_vote1(locx,locy,w):    
    vote = {}
    label_expand = np.lib.pad(sp1,((int(w / 2), int(w / 2)), (int(w / 2),int(w / 2))),'constant', constant_values=0)
    for i in range(w):
        for j in range(w):
            ith_label = label_expand[locx+i,locy+j]
            vote[ith_label] = vote.get(ith_label,0)+1
    sortedvote = sorted(vote.items(),key = lambda x:x[1], reverse = True)
    num_all = int(w*w)
    ratio = sortedvote[0][1]/num_all
    return sortedvote[0][0],ratio

def get_testlabel(testdata,meanlist):
    classnum = np.shape(meanlist)[0]
    gaus_t_list = []
    for i in range(classnum):
        gaus_dis_t = gausdis(testdata,meanlist[i])
        gaus_t_list.append(gaus_dis_t)
    maxindex = gaus_t_list.index(max(gaus_t_list))
    return maxindex

expand_loc = sio.loadmat(path + '/expand/'+data_name+'/expand_70.mat')['expand_loc']
expand_label = sio.loadmat(path + '/expand/'+data_name+'/expand_70.mat')['expand_label'][0]
cum_exe_list = sio.loadmat(path + '/expand/'+data_name+'/expand_70.mat')['cum_exe_list'][0]
num_exe = sio.loadmat(path + '/expand/'+data_name+'/expand_70.mat')['num_exe'][0]
loc_exe_bianjiex = []
loc_exe_yunzhix = []
loc_exe_bianjiey = []
loc_exe_yunzhiy = []
label_exe_bianjie = []
label_exe_yunzhi = []  
for i in range(dim_out):
    start1 = cum_exe_list[i]
    num_ev_exe = num_exe[i]
    mean = meanlist[i,:]
    mingaus = mingaus_list[i]
    countbe = 0
    countye = 0
    for j in range(num_ev_exe):
        ith = start1+j
        ratio = neibor_ratio(expand_loc[0][ith],expand_loc[1][ith],w[2],mean,mingaus)
        most_label,ratio1 = neibor_vote1(expand_loc[0][ith],expand_loc[1][ith],w[2])
        if (ratio<0.9)and(ratio1<0.9):
#        if (ratio1<=0.9):
            loc_exe_bianjiex.append(expand_loc[0][ith])
            loc_exe_bianjiey.append(expand_loc[1][ith])
            label_exe_bianjie.append(expand_label[ith])
            countbe+=1
        else :
            loc_exe_yunzhix.append(expand_loc[0][ith])
            loc_exe_yunzhiy.append(expand_loc[1][ith])
            label_exe_yunzhi.append(expand_label[ith])
            countye+=1    
    print(countbe,countye)
    countbe = 0
    countye = 0
loc_exe_bianjie = np.vstack([loc_exe_bianjiex,loc_exe_bianjiey])        
loc_exe_yunzhi = np.vstack([loc_exe_yunzhix,loc_exe_yunzhiy])
print('expand train bianjie shape',loc_exe_bianjie.shape)
print('expand train yunzhi shape',loc_exe_yunzhi.shape)

loc_train_yunzhi = sio.loadmat(path + '/location/'+data_name+'/'+'yunzhi_noe.mat')['new_yunzhi_tran_loc'] 
label_train_yunzhi = sio.loadmat(path + '/location/'+data_name+'/'+'yunzhi_noe.mat')['new_yunzhi_tran_label'][0]
loc_train_bianjie = sio.loadmat(path + '/location/'+data_name+'/'+'bianjie_noe.mat')['new_bianjie_tran_loc'] 
label_train_bianjie = sio.loadmat(path + '/location/'+data_name+'/'+'bianjie_noe.mat')['new_bianjie_tran_label'][0] 

loc_test_yunzhi = sio.loadmat(path + '/location/'+data_name+'/'+'yunzhi_noe.mat')['loc_test_yunzhi'] 
label_test_yunzhi = sio.loadmat(path + '/location/'+data_name+'/'+'yunzhi_noe.mat')['label_test_yunzhi'][0]
loc_test_bianjie = sio.loadmat(path + '/location/'+data_name+'/'+'bianjie_noe.mat')['loc_test_bianjie'] 
label_test_bianjie = sio.loadmat(path + '/location/'+data_name+'/'+'bianjie_noe.mat')['label_test_bianjie'][0]

#new_bianjie_tran_loc = np.hstack((train_loc,loc_train_bianjie))
#new_yunzhi_tran_loc = np.hstack((train_loc,loc_train_yunzhi))
#new_bianjie_tran_label = np.hstack((y_train,label_train_bianjie))
#new_yunzhi_tran_label = np.hstack((y_train,label_train_yunzhi))

new_bianjie_tran_loc = np.hstack((train_loc,loc_train_bianjie,loc_exe_bianjie))
new_yunzhi_tran_loc = np.hstack((train_loc,loc_train_yunzhi,loc_exe_yunzhi))
new_bianjie_tran_label = np.hstack((y_train,label_train_bianjie,label_exe_bianjie))
new_yunzhi_tran_label = np.hstack((y_train,label_train_yunzhi,label_exe_yunzhi))

#new_bianjie_tran_loc = loc_train_bianjie
#new_yunzhi_tran_loc = loc_train_yunzhi
#new_bianjie_tran_label = label_train_bianjie
#new_yunzhi_tran_label = label_train_yunzhi

#new_bianjie_tran_loc = np.hstack((loc_train_bianjie,loc_exe_bianjie))
#new_yunzhi_tran_loc = np.hstack((loc_train_yunzhi,loc_exe_yunzhi))
#new_bianjie_tran_label = np.hstack((label_train_bianjie,label_exe_bianjie))
#new_yunzhi_tran_label = np.hstack((label_train_yunzhi,label_exe_yunzhi))

sio.savemat(path + '/location/'+data_name+'/'+'/yunzhi_we.mat',{'new_yunzhi_tran_loc':new_yunzhi_tran_loc,'new_yunzhi_tran_label':new_yunzhi_tran_label,'loc_test_yunzhi':loc_test_yunzhi,'label_test_yunzhi':label_test_yunzhi})
sio.savemat(path + '/location/'+data_name+'/'+'/bianjie_we.mat',{'new_bianjie_tran_loc':new_bianjie_tran_loc,'new_bianjie_tran_label':new_bianjie_tran_label,'loc_test_bianjie':loc_test_bianjie,'label_test_bianjie':label_test_bianjie})
#loc_test_bianjiex = []
#loc_test_yunzhix = []
#loc_test_bianjiey = []
#loc_test_yunzhiy = []
#label_test_bianjie = []
#label_test_yunzhi = []  
#for i in range(test_size[1]):
#    most_label,ratio = neibor_vote(test_loc[0][i],test_loc[1][i],w[1])
#    ith_label = labels_ori[test_loc[0][i],test_loc[1][i]]
#    if (ratio>0.8)and(most_label==ith_label):
#        loc_test_yunzhix.append(test_loc[0][i])
#        loc_test_yunzhiy.append(test_loc[1][i])
#        label_test_yunzhi.append(y_test[i])
#    else :
#        loc_test_bianjiex.append(test_loc[0][i])
#        loc_test_bianjiey.append(test_loc[1][i])
#        label_test_bianjie.append(y_test[i])
#
#loc_test_bianjie = np.vstack([loc_test_bianjiex,loc_test_bianjiey])        
#loc_test_yunzhi = np.vstack([loc_test_yunzhix,loc_test_yunzhiy])
#print('test bianjie shape',loc_test_bianjie.shape)
#print('test yunzhi shape',loc_test_yunzhi.shape)
#
#new_bianjie_tran_loc = np.hstack((train_loc,loc_train_bianjie))
#new_yunzhi_tran_loc = np.hstack((train_loc,loc_train_yunzhi))
#new_bianjie_tran_label = np.hstack((y_train,label_train_bianjie))
#new_yunzhi_tran_label = np.hstack((y_train,label_train_yunzhi))
'''
loc_all_bianjiex = []
loc_all_yunzhix = []
loc_all_bianjiey = []
loc_all_yunzhiy = []
label_tr = np.reshape(labels_ori,[size[0]*size[1]])
size = np.shape(labels_ori)
locx = []
locy = []
for i in range(size[0]):
    for j in range(size[1]):
        locx.append(i)
        locy.append(j)
loc_all = np.vstack((locx,locy))
for i in range(size[0]*size[1]):
    if (label_tr[i]==0):
#        locx = loc_all[0][i]
#        locy = loc_all[1][i]
#        testdata = data_norm[locx,locy,:]
#        ith_label = get_testlabel(testdata,meanlist)
#        mean = meanlist[ith_label,:]
#        mingaus = mingaus_list[ith_label]
#        ratio = neibor_ratio(loc_all[0][i],loc_all[1][i],w[2],mean,mingaus)
        most_label,ratio1 = neibor_vote1(loc_all[0][i],loc_all[1][i],w[2])
        if ratio1>=0.9:
#        if(ratio>0.97)or(ratio1>0.97):
            loc_all_yunzhix.append(loc_all[0][i])
            loc_all_yunzhiy.append(loc_all[1][i])
        else :
            loc_all_bianjiex.append(loc_all[0][i])
            loc_all_bianjiey.append(loc_all[1][i])
    if i%10000==0:
        print(i)
loc_all_bianjie = np.vstack([loc_all_bianjiex,loc_all_bianjiey])        
loc_all_yunzhi = np.vstack([loc_all_yunzhix,loc_all_yunzhiy])

loc_all_bianjie = np.hstack((loc_train_bianjie,loc_test_bianjie,loc_all_bianjie))
loc_all_yunzhi = np.hstack((loc_train_yunzhi,loc_test_yunzhi,loc_all_yunzhi))

#bianjie_all = get_input(w[1],loc_all_bianjie,dim_input[0])
#yunzhi_all = get_input(w[3],loc_all_yunzhi,dim_input[3])

#print(' bianjie shape',bianjie_all.shape)
#print(' yunzhi shape',yunzhi_all.shape)
path = os.getcwd()
#sio.savemat(path+'/data_all/'+data_name+'/'+data_name+'bj_yz.mat',{'bianjie_all':bianjie_all,'yunzhi_all':yunzhi_all})
sio.savemat(path+'/location/'+data_name+'/'+'loc_all.mat',{'loc_all_bianjie':loc_all_bianjie,'loc_all_yunzhi':loc_all_yunzhi})

'''
 
