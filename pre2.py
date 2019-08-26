# -*- coding: utf-8 -*-


import numpy as np
import scipy.io as sio
import os

from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import OrthogonalMatchingPursuit
import scipy.stats as st


def readData(data_name):
    ''' 读取原始数据和标准类标 '''
    path = os.getcwd()+'/data/'+data_name
    print(path)
    if data_name == 'Indian_pines':
        data = sio.loadmat(path+'/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat(path+'/Indian_pines_gt.mat')['indian_pines_gt']
    elif data_name == 'PaviaU':
        data = sio.loadmat(path+'/PaviaU.mat')['paviaU']
        labels = sio.loadmat(path+'/PaviaU_gt.mat')['paviaU_gt']
    elif data_name == 'Salinas':
        data = sio.loadmat(path+'/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat(path+'/Salinas_gt.mat')['salinas_gt']
    elif data_name == 'KSC':
        data = sio.loadmat(path+'/KSC.mat')['KSC']
        labels = sio.loadmat(path+'/KSC_gt.mat')['KSC_gt']
    elif data_name == 'washington':
        data = sio.loadmat(path+'/washington_datax.mat')['washington_datax']
        labels = sio.loadmat(path+'/washington_gt.mat')['washington_gt']
    data = np.float64(data)
    labels = np.array(labels).astype(float)
    return data, labels

def normalizeData(data):
    ''' 原始数据归一化处理（每条） '''
    data_norm = np.zeros(np.shape(data))
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            data_norm[i,j,:] = preprocessing.normalize(data[i,j,:].reshape(1,-1))[0]
    return data_norm

#def pcanormalize(data):
#    ''' 原始数据归一化处理（每条） '''
#    data_norm = np.zeros(np.shape(data))
#    for i in range(np.shape(data)[0]):
#        for j in range(np.shape(data)[1]):
#            rowmax = max(data[i,j,:])
#            rowmin = min(data[i,j,:])
#            data_norm[i,j,:] = (data[i,j,:]-rowmin)/(rowmax-rowmin)
#    return data_norm
    
def selectTrainTest(data, labels, p):
    ''' 从所有类中每类选取训练样本和测试样本 '''
    c = int(labels.max())
    x = np.array([], dtype=float).reshape(-1, data.shape[2])  # 训练样本
    xb = []
    x_loc1 = []
    x_loc2 = []
    x_loc = []
    y = np.array([], dtype=float).reshape(-1, data.shape[2])
    yb = []
    y_loc1 = []
    y_loc2 = []
    y_loc = []
    num_list = []
    cum = 0
    cumlist = [0]
    for i in range(1, c+1):
    #i = 1
        loc1, loc2 = np.where(labels == i)
        num = len(loc1)
        order = np.random.permutation(range(num))
        loc1 = loc1[order]
        loc2 = loc2[order]
        num1 = int(np.round(num*p)+1)
        print(num,num1,num-num1)
        num_list.append(num1)
        cum = cum+num1
        cumlist.append(cum)
        x = np.vstack([x, data[loc1[:num1], loc2[:num1], :]])
        y = np.vstack([y, data[loc1[num1:], loc2[num1:], :]])
        xb.extend([i]*num1)
        yb.extend([i]*(num-num1))
        x_loc1.extend(loc1[:num1])
        x_loc2.extend(loc2[:num1])
        y_loc1.extend(loc1[num1:])
        y_loc2.extend(loc2[num1:])
        x_loc = np.vstack([x_loc1, x_loc2])
        y_loc = np.vstack([y_loc1, y_loc2])
    return x, xb, x_loc, y, yb, y_loc,num_list,cumlist
'''    
def getPCA(data, n_components):

    M, N, D = data.shape
    pca = PCA(n_components=n_components)
    dataPCA = pca.fit_transform(data.reshape(M*N, D)).reshape(M, N, -1)
    return dataPCA

    
def getSVM(data, labels, train_x, train_y, train_loc, test_x, test_y, test_loc):
    c = int(labels.max())
    # SVM参数选择(实际上选好了），并记录SVM分类正确率    
#    info = {"C":0, "gamma":0, "acc":0}
#    for C in range(322,323):
#        for gamma in xrange(1458,1459):
#            gamma = gamma*0.01
#            clf = svm.SVC(C=C, gamma=gamma, decision_function_shape='ovr', probability=True)
#            clf.fit(x, xb)
#            a = clf.score(y, yb)
#            if a > info["acc"]:
#                info["acc"] = a
#                info["C"] = C
#                info["gamma"] = gamma
    best_c = 491
    best_gamma = 14.59
    clf = svm.SVC(C=best_c, gamma=best_gamma,
                  decision_function_shape='ovr', probability=True)
    clf.fit(train_x, train_y)
    acc_SVM = clf.score(test_x, test_y)  # SVM分类正确率          
    test_y1 = clf.predict(test_x)
    # 得到最终预测类标及结果图
    #%matplotlib qt
    import copy
    labels1 = copy.deepcopy(labels)
    for i in range(test_loc.shape[1]):labels1[test_loc[0, i], test_loc[1, i]] = test_y1[i] + 1  
    # 对比图
#    plt.figure()
#    plt.subplot(121);plt.imshow(labels)
#    plt.axis('off')
#    plt.subplot(122);plt.imshow(labels1)
#    plt.axis('off')
#    # 单独结果图
#    plt.figure(2)
#    plt.imshow(labels1)
#    plt.axis('off')
    
    # 得到SVM分类概率
    #svmProb = clf.predict_proba(data_norm.reshape((data.shape[0]*data.shape[1], -1)))
    #dataSVM = svmProb.reshape((data.shape[0], data.shape[1], -1))
    data_SVM = np.zeros((data.shape[0], data.shape[1], c))
    for i in range(train_loc.shape[1]):
        loc1 = train_loc[0][i]
        loc2 = train_loc[1][i]
        data_SVM[loc1, loc2, :] = clf.predict_proba(data[loc1, loc2, :].reshape(1,-1))[0]
    for j in range(test_loc.shape[1]):
        loc1 = test_loc[0][j]
        loc2 = test_loc[1][j]
        data_SVM[loc1, loc2, :] = clf.predict_proba(data[loc1, loc2, :].reshape(1,-1))[0]
    return (data_SVM, acc_SVM, labels1)
'''    

def windowFeature(data, loc, w ):

    ''' 从扩展矩阵中得到窗口特征'''
    size = np.shape(data)
    data_expand = np.zeros((int(size[0]+w-1),int(size[1]+w-1),size[2]))
    newdata = np.zeros((len(loc[0]), w, w,size[2]))
    for j in range(size[2]):    
        data_expand[:,:,j] = np.lib.pad(data[:,:,j], ((int(w / 2), int(w / 2)), (int(w / 2),int(w / 2))), 'symmetric')
        newdata[:,:,:,j] = np.zeros((len(loc[0]), w, w))
        for i in range(len(loc[0])):
            loc1 = int(loc[0][i])
            loc2 = int(loc[1][i])
            f = data_expand[loc1:int(loc1 + w), loc2:int(loc2 + w),j]
            newdata[i, :, :,j] = f
    return newdata

  
if __name__ == '__main__':

#    data_name = 'Indian_pines'
    data_name = 'PaviaU'
#    data_name = 'Salinas'
##    data_name = 'KSC'
#    data_name = 'washington'
    # 预处理
    data_ori, labels_ori = readData(data_name)
    data_norm = normalizeData(data_ori)
    if data_name == 'Indian_pines':
        p = 0.05
    elif data_name == 'PaviaU':
        p = 0.03
    elif data_name == 'Salinas':
        p = 0.01
    elif data_name == 'KSC':
        p = 0.05
    elif data_name == 'washington':
        p = 0.09
    train_x, train_y, train_loc, test_x, test_y, test_loc,train_num_list,cumlist= selectTrainTest(data_norm, labels_ori, p)
    
    path = os.getcwd()
    sio.savemat(path+'/data/'+data_name+'/'+data_name+'_pre', {'train_x':train_x,
                'train_y':train_y, 'train_loc':train_loc, 'test_x':test_x,
                'test_y':test_y, 'test_loc':test_loc, 'data_norm':data_norm,
                'labels_ori':labels_ori,'train_num_list':train_num_list,'cumlist':cumlist})
