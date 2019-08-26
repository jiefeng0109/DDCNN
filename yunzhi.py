import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
import time
import scipy.io as sio
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

dim_out = 9

dropout = [0.5,0.8,1]
batch_size = 128
learn_rate= 0.01
display_step = 2000
num_epoch = 1000
w = [1,3,5,31]


dim_input = [150,100,20,10,1]
#data_name = 'Indian_pines'
data_name = 'PaviaU'
#data_name = 'KSC'
#data_name = 'washington'
#data_name = 'Salinas'
path = os.getcwd()
pre_dir = path + '/data/' + data_name
pre = sio.loadmat(pre_dir + '/' + data_name + '_pre.mat')
data_norm = pre['data_norm']
#labels_ori = pre['labels_ori']
#x_train = pre['train_x']
#y_train = pre['train_y'][0]
#train_loc = pre['train_loc']
#x_test = pre['test_x']
#y_test = pre['test_y'][0]
#test_loc = pre['test_loc']  
def get_input(w,loc,dim_input,data_norm):
    print(data_norm.shape)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim_input)
    data_PCA = pca.fit_transform(data_norm.reshape(data_norm.shape[0]*data_norm.shape[1], -1))
    data_PCA = data_PCA.reshape(data_norm.shape[0], data_norm.shape[1],dim_input)

    # 转换数据维度
    from pre2 import windowFeature
    X = windowFeature(data_PCA, loc, w)
    
    return X
new_yunzhi_tran_loc = sio.loadmat(path + '/location/'+data_name+'/'+'yunzhi_noe.mat')['new_yunzhi_tran_loc'] 
loc_test_yunzhi = sio.loadmat(path + '/location/'+data_name+'/'+'yunzhi_noe.mat')['loc_test_yunzhi'] 
new_yunzhi_tran_label = sio.loadmat(path + '/location/'+data_name+'/'+'yunzhi_noe.mat')['new_yunzhi_tran_label'][0]
label_test_yunzhi = sio.loadmat(path + '/location/'+data_name+'/'+'yunzhi_noe.mat')['label_test_yunzhi'][0]

#X_train0 = get_input(w[0],train_loc,dim_input[0])
#X_train1 = get_input(w[1],train_loc,dim_input[1])
#X_train2 = get_input(w[2],train_loc,dim_input[2])
X_train3 = get_input(w[3],new_yunzhi_tran_loc,dim_input[3],data_norm)
#X_test0 = get_input(w[0],loc_test_bianjie,dim_input[0])
#X_test1 = get_input(w[1],loc_test_bianjie,dim_input[1])
#X_test2 = get_input(w[2],loc_test_bianjie,dim_input[2])
X_test3 = get_input(w[3],loc_test_yunzhi,dim_input[3],data_norm)
#print('X_train0 shape:', X_train0.shape)
#print('X_test0 shape:', X_test0.shape)
#print('X_train1 shape:', X_train1.shape)
#print('X_test1 shape:', X_test1.shape)
#print('X_train2 shape:', X_train2.shape)
#print('X_test2 shape:', X_test2.shape)
print('X_train3 shape:', X_train3.shape)
print('X_test3 shape:', X_test3.shape)
#print('X_train4 shape:', X_train4.shape)
#print('X_test4 shape:', X_test4.shape)
def one_hot(lable,class_number):
    one_hot_array = np.zeros([len(lable),class_number])
    for i in range(len(lable)):
        one_hot_array[i,int(lable[i]-1)] = 1
    return one_hot_array

Y_train = one_hot(new_yunzhi_tran_label,dim_out )
#Y_test = one_hot(y_train,dim_out )
#Y_test_bianjie = one_hot(label_test_bianjie,dim_out)
Y_test_yunzhi = one_hot(label_test_yunzhi,dim_out)

index_train = np.arange(X_train3.shape[0])
np.random.shuffle(index_train)
X_train3 = X_train3[index_train, :]
Y_train = Y_train[index_train, :]

index_test = np.arange(X_test3.shape[0])
np.random.shuffle(index_test)
X_test3 = X_test3[index_test, :]
Y_test_yunzhi = Y_test_yunzhi[index_test, :]

def next_batchx(image,batch_size):
    start = batch_size-128
    end = batch_size
    return image[start:end,:,:,:]
def next_batchy(lable,batch_size):
    start = batch_size-128
    end = batch_size
    return lable[start:end]

def batch_norm(inputs, is_training,is_conv_out=True,decay = 0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])   

        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,pop_mean, pop_var, beta, scale, 0.001)
def conv2dlayer(x,W,B,stride):
    x = tf.nn.conv2d(x,W,stride,padding='SAME',name='CONV')
    h = tf.nn.bias_add(x,B)
    bn = batch_norm(h, is_training=True,is_conv_out=True,decay = 0.999)
    convout = tf.nn.relu(bn)
    return convout
def contrary_one_hot(label):
    size=len(label)
    label_ori=np.empty(size)
    for i in range(size):
        label_ori[i]=np.argmax(label[i])+1
    return label_ori

#x0 = tf.placeholder(tf.float32,[None,w[0],w[0],dim_input[0]],name='x_input0')
#x1 = tf.placeholder(tf.float32,[None,w[1],w[1],dim_input[1]],name='x_input1')
#x2 = tf.placeholder(tf.float32,[None,w[2],w[2],dim_input[2]],name='x_input2')
x3 = tf.placeholder(tf.float32,[None,w[3],w[3],dim_input[3]],name='x_input3')
#x4 = tf.placeholder(tf.float32,[None,w[2],w[2],dim_input[4]],name='x_input4')
y = tf.placeholder(tf.float32,[None,dim_out],name='y_output')
dropout = tf.placeholder(tf.float32,name='dropout')
#z = tf.placeholder(tf.int32,[None],name='x_size')

weights={'W10':tf.Variable(tf.truncated_normal([1,1,dim_input[3],10],stddev=0.1)),
         
         'W11':tf.Variable(tf.truncated_normal([3,3,dim_input[3],10],stddev=0.1)),

         'W12':tf.Variable(tf.truncated_normal([5,5,dim_input[3],10],stddev=0.1)),
         
#         'W13':tf.Variable(tf.truncated_normal([5,5,dim_input[3],32],stddev=0.1)),
         
#         'W2':tf.Variable(tf.truncated_normal([1,1,30,64],stddev=0.1)),
         
         'W23':tf.Variable(tf.truncated_normal([5,5,30,32],stddev=0.1)),        

#         'W3':tf.Variable(tf.truncated_normal([1,1,64,128],stddev=0.1)),

         'W33':tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1)), 

#         'W4':tf.Variable(tf.truncated_normal([1,1,128,256],stddev=0.1)),

         'W43':tf.Variable(tf.truncated_normal([5,5,64,128],stddev=0.1)),
         
         'W6':tf.Variable(tf.truncated_normal([3,3,128,128],stddev=0.1)),
 
#         'W5':tf.Variable(tf.truncated_normal([256*3,dim_out],stddev=0.1)),
                                              
         'W53':tf.Variable(tf.truncated_normal([128*2*2,dim_out],stddev=0.1)),
        }

bias={'B1':tf.Variable(tf.constant(0.1,shape=[10])),
         
      'B2':tf.Variable(tf.constant(0.1,shape=[32])),

      'B3':tf.Variable(tf.constant(0.1,shape=[64])),

      'B4':tf.Variable(tf.constant(0.1,shape=[128])),

      'B5':tf.Variable(tf.constant(0.1,shape=[dim_out])),
      }

def branch0(x0,weights,bias,dropout,w,dim_input):
    x0 = tf.reshape(x0,shape=[-1,w,w,dim_input])
    conv1 = conv2dlayer(x0,weights['W10'],bias['B1'],[1,1,1,1])
    pool1 = tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
    conv2=conv2dlayer(pool1,weights['W2'],bias['B2'],[1,1,1,1])
    dpt1 = tf.nn.dropout(conv2,dropout)
    pool2 = tf.nn.max_pool(dpt1,[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL2')
    conv3=conv2dlayer(pool2,weights['W3'],bias['B3'],[1,1,1,1])
    dpt2 = tf.nn.dropout(conv3,dropout)
    pool3=tf.nn.max_pool(dpt2,[1,2,2,1],[1,2,2,1], padding='SAME', name='POOL3')
    conv4=conv2dlayer(pool3,weights['W4'],bias['B4'],[1,1,1,1])
    reshape = tf.reshape(conv4,[-1,weights['W4'].get_shape().as_list()[3]])
    return reshape
def branch1(x1,weights,bias,dropout,w,dim_input):
    x1 = tf.reshape(x1,shape=[-1,w,w,dim_input])    
    conv1 = conv2dlayer(x1,weights['W11'],bias['B1'],[1,1,1,1])
    pool1 = tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
    conv2=conv2dlayer(pool1,weights['W2'],bias['B2'],[1,1,1,1])
    dpt1 = tf.nn.dropout(conv2,dropout)
    pool2 = tf.nn.max_pool(dpt1,[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL2')
    conv3=conv2dlayer(pool2,weights['W3'],bias['B3'],[1,1,1,1])
    dpt2 = tf.nn.dropout(conv3,dropout)
    pool3=tf.nn.max_pool(dpt2,[1,2,2,1],[1,2,2,1], padding='SAME', name='POOL3')
    conv4=conv2dlayer(pool3,weights['W4'],bias['B4'],[1,1,1,1])
    reshape = tf.reshape(conv4,[-1,weights['W4'].get_shape().as_list()[3]])
    return reshape
def branch2(x2,weights,bias,dropout,w,dim_input):
    x2 = tf.reshape(x2,shape=[-1,w,w,dim_input])
    conv1 = conv2dlayer(x2,weights['W12'],bias['B1'],[1,1,1,1])
    pool1 = tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
    conv2=conv2dlayer(pool1,weights['W2'],bias['B2'],[1,1,1,1])
    dpt1 = tf.nn.dropout(conv2,dropout)
    pool2 = tf.nn.max_pool(dpt1,[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL2')
    conv3=conv2dlayer(pool2,weights['W3'],bias['B3'],[1,1,1,1])
    dpt2 = tf.nn.dropout(conv3,dropout)
    pool3=tf.nn.max_pool(dpt2,[1,2,2,1],[1,2,2,1], padding='SAME', name='POOL3')
    conv4=conv2dlayer(pool3,weights['W4'],bias['B4'],[1,1,1,1])
    reshape = tf.reshape(conv4,[-1,weights['W4'].get_shape().as_list()[3]])
    return reshape
def branch3(x3,weights,bias,dropout,w,dim_input):
    x3 = tf.reshape(x3,shape=[-1,w,w,dim_input])
    conv10 = conv2dlayer(x3,weights['W10'],bias['B1'],[1,1,1,1])
    conv11 = conv2dlayer(x3,weights['W11'],bias['B1'],[1,1,1,1])
    conv12 = conv2dlayer(x3,weights['W12'],bias['B1'],[1,1,1,1])
    conv1 = tf.concat([conv10,conv11,conv12],3)
    pool1 = tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
    conv2=conv2dlayer(pool1,weights['W23'],bias['B2'],[1,1,1,1])
    dpt1 = tf.nn.dropout(conv2,dropout)
    pool2 = tf.nn.max_pool(dpt1,[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
    conv3=conv2dlayer(pool2,weights['W33'],bias['B3'],[1,1,1,1])
    dpt2 = tf.nn.dropout(conv3,dropout)
    pool3=tf.nn.max_pool(dpt2,[1,2,2,1],[1,2,2,1], padding='SAME', name='POOL3')
    conv4=conv2dlayer(pool3,weights['W43'],bias['B4'],[1,1,1,1])
    pool4=tf.nn.max_pool(conv4,[1,2,2,1],[1,2,2,1], padding='SAME', name='POOL4')
#    conv5=conv2dlayer(pool4,weights['W6'],bias['B4'],[1,1,1,1])
    reshape = tf.reshape(pool4,[-1,weights['W53'].get_shape().as_list()[0]])
    return reshape

def get_oa(X_valid):
    size = np.shape(X_valid)
    num = size[0]
    index_all = 0
    step = 2000
    y_pred = []
    while index_all<num:
        if index_all + step > num:
            input = X_valid[index_all:, :, :, :]
        else:
            input = X_valid[index_all:(index_all+step), :, :, :]
        index_all += step
        temp1 = y_.eval(feed_dict={x3: input,dropout:1})
        y_pred1=contrary_one_hot(temp1).astype('int32')
        y_pred.extend(y_pred1)
    
    return y_pred

def shuffdata(data,loc):
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    data = data[index, :]
    loc = loc[:,index]
    return data, loc
    
#reshape0 = branch0(x0,weights,bias,dropout[0],w[0],dim_input[0])
#reshape1 = branch1(x1,weights,bias,dropout[0],w[1],dim_input[1])  
#reshape2 = branch2(x2,weights,bias,dropout[0],w[2],dim_input[2])
reshape3 = branch3(x3,weights,bias,dropout,w[3],dim_input[3])
#reshape = tf.concat([reshape0,reshape1,reshape2],1)
#dpt = tf.nn.dropout(reshape,dropout[0])
#gate = gate(x4,w[2],dim_input[4],threshold)
#f1 = reshape*gate
f2 = tf.add(tf.matmul(reshape3,weights['W53']),bias['B5'])
#dpt3 = tf.nn.dropout(f2,dropout)
y_=tf.nn.softmax(f2)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=f2, name=None))
#cross_entropy = tf.reduce_mean(-(y*tf.log(y_)+(1-y)*tf.log(1-y_)))
train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(cross_entropy) 
#train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    epoch = 0
    index = batch_size
    time_train_start=time.clock()
    while epoch<num_epoch:
#    batch_x0 = next_batchx(X_train0,index)
#    batch_x1 = next_batchx(X_train1,index)
#    batch_x2 = next_batchx(X_train2,index)
        batch_x3 = next_batchx(X_train3,index)
#    batch_x4 = next_batchx(X_train4,index)
        batch_y = next_batchy(Y_train,index)

        sess.run(train_step, feed_dict={x3: batch_x3,y: batch_y,dropout:0.5})
        if step%display_step == 0:
            loss,acc = sess.run([cross_entropy,accuracy], feed_dict={x3: batch_x3,y: batch_y,dropout:0.5})
            print('step %d,training accuracy %f,cross_entrop %f'%(step,acc,loss))
        #print("step" + str(step) + ", Minibatch Loss= " + \
                  #"{:.6f}".format(loss) + ", Training Accuracy= " + \
                  #"{:.5f}".format(acc))
#            print("test accuracy %g"%accuracy.eval(feed_dict={x3: X_test3,y: Y_test_yunzhi,dropout:1}))
            
            y_pred = get_oa(X_test3)
            y_tr=contrary_one_hot(Y_test_yunzhi).astype('int32')
            oa = accuracy_score(y_tr,y_pred)
            print("test accuracy %g"%oa)
        index = index+batch_size
        step += 1
        if index>X_train3.shape[0]:
            index = batch_size
            epoch=epoch+1
    time_train_end=time.clock()   
    print("Optimization Finished!")
    time_test_start=time.clock()
    y_pred = get_oa(X_test3)
    y_tr=contrary_one_hot(Y_test_yunzhi).astype('int32')
#    temp = y_.eval(feed_dict={x3: X_test3,y: Y_test_yunzhi,dropout:1})
#    y_pred=contrary_one_hot(temp).astype('int32')
#    y_tr=contrary_one_hot(Y_test_yunzhi).astype('int32')
    oa=accuracy_score(y_tr,y_pred)
    per_class_acc=recall_score(y_tr,y_pred,average=None)
    aa=np.mean(per_class_acc)
    kappa=cohen_kappa_score(y_tr,y_pred)
    time_test_end=time.clock()
    print(per_class_acc)
    print(oa,aa,kappa)
    print((time_train_end-time_train_start),(time_test_end-time_test_start))

    sio.savemat(path+'/label/'+data_name+'/'+'label.mat',{'label_yunzhi_tr':y_tr,'label_yunzhi_pre':y_pred})

##    data_all_dir = path + '/data_all/' + data_name
##    data = sio.loadmat(data_all_dir + '/' + data_name + 'bj_yz.mat')
##    data_all = data['yunzhi_all']
#    loc = sio.loadmat(path+'/label/'+data_name+'/'+'loc.mat')
#    loc_all = loc['loc_all_yunzhi'].astype(int)
#    data_all = get_input(w[3],loc_all,dim_input[3],data_norm)
#    data_all,loc_all =  shuffdata(data_all,loc_all)
##    temp1=y_.eval(feed_dict={x3: data_all,dropout:1})
##    y_pred=contrary_one_hot(temp1).astype('int32')
#    y_pred = get_oa(data_all)
#    sio.savemat(path+'/label/'+data_name+'/'+'label_all_yunzhi.mat',{'label__all_yunzhi_pre':y_pred})
#    sio.savemat(path+'/label/'+data_name+'/'+'locy.mat',{'loc_all_yunzhi':loc_all})

with tf.Session() as sess1:
    sess1.run(init)
    step = 1
    epoch = 0
    index = batch_size
    time_train_start=time.clock()
    while epoch<num_epoch:
#    batch_x0 = next_batchx(X_train0,index)
#    batch_x1 = next_batchx(X_train1,index)
#    batch_x2 = next_batchx(X_train2,index)
        batch_x3 = next_batchx(X_train3,index)
#    batch_x4 = next_batchx(X_train4,index)
        batch_y = next_batchy(Y_train,index)

        sess1.run(train_step, feed_dict={x3: batch_x3,y: batch_y,dropout:0.5})
        index = index+batch_size
        step += 1
        if index>X_train3.shape[0]:
            index = batch_size
            epoch=epoch+1
    time_train_end=time.clock()   
    print("Optimization Finished!")
    time_test_start=time.clock() 

    y_pred = get_oa(X_test3)
    y_tr=contrary_one_hot(Y_test_yunzhi).astype('int32')
#    temp = y_.eval(feed_dict={x3: X_test3,y: Y_test_yunzhi,dropout:1})
#    y_pred=contrary_one_hot(temp).astype('int32')
#    y_tr=contrary_one_hot(Y_test_yunzhi).astype('int32')
    oa=accuracy_score(y_tr,y_pred)
    per_class_acc=recall_score(y_tr,y_pred,average=None)
    aa=np.mean(per_class_acc)
    kappa=cohen_kappa_score(y_tr,y_pred)
    time_test_end=time.clock()
    print(per_class_acc)
    print(oa,aa,kappa)
    print((time_train_end-time_train_start),(time_test_end-time_test_start))

    sio.savemat(path+'/label/'+data_name+'/'+'label1.mat',{'label_yunzhi_tr':y_tr,'label_yunzhi_pre':y_pred})

with tf.Session() as sess2:
    sess2.run(init)
    step = 1
    epoch = 0
    index = batch_size
    time_train_start=time.clock()
    while epoch<num_epoch:
#    batch_x0 = next_batchx(X_train0,index)
#    batch_x1 = next_batchx(X_train1,index)
#    batch_x2 = next_batchx(X_train2,index)
        batch_x3 = next_batchx(X_train3,index)
#    batch_x4 = next_batchx(X_train4,index)
        batch_y = next_batchy(Y_train,index)
        sess2.run(train_step, feed_dict={x3: batch_x3,y: batch_y,dropout:0.5})
        index = index+batch_size
        step += 1
        if index>X_train3.shape[0]:
            index = batch_size
            epoch=epoch+1
    time_train_end=time.clock()   
    print("Optimization Finished!")

    time_test_start=time.clock()
    y_pred = get_oa(X_test3)
    y_tr=contrary_one_hot(Y_test_yunzhi).astype('int32')
#    temp = y_.eval(feed_dict={x3: X_test3,y: Y_test_yunzhi,dropout:1})
#    y_pred=contrary_one_hot(temp).astype('int32')
#    y_tr=contrary_one_hot(Y_test_yunzhi).astype('int32')
    oa=accuracy_score(y_tr,y_pred)
    per_class_acc=recall_score(y_tr,y_pred,average=None)
    aa=np.mean(per_class_acc)
    kappa=cohen_kappa_score(y_tr,y_pred)
    time_test_end=time.clock()
    print(per_class_acc)
    print(oa,aa,kappa)
    print((time_train_end-time_train_start),(time_test_end-time_test_start))

    sio.savemat(path+'/label/'+data_name+'/'+'label2.mat',{'label_yunzhi_tr':y_tr,'label_yunzhi_pre':y_pred})
    
with tf.Session() as sess3:
    sess3.run(init)
    step = 1
    epoch = 0
    index = batch_size
    time_train_start=time.clock()
    while epoch<num_epoch:
#    batch_x0 = next_batchx(X_train0,index)
#    batch_x1 = next_batchx(X_train1,index)
#    batch_x2 = next_batchx(X_train2,index)
        batch_x3 = next_batchx(X_train3,index)
#    batch_x4 = next_batchx(X_train4,index)
        batch_y = next_batchy(Y_train,index)
        sess3.run(train_step, feed_dict={x3: batch_x3,y: batch_y,dropout:0.5})
        index = index+batch_size
        step += 1
        if index>X_train3.shape[0]:
            index = batch_size
            epoch=epoch+1
    time_train_end=time.clock()   
    print("Optimization Finished!")
    time_test_start=time.clock()
    y_pred = get_oa(X_test3)
    y_tr=contrary_one_hot(Y_test_yunzhi).astype('int32')
#    temp = y_.eval(feed_dict={x3: X_test3,y: Y_test_yunzhi,dropout:1})
#    y_pred=contrary_one_hot(temp).astype('int32')
#    y_tr=contrary_one_hot(Y_test_yunzhi).astype('int32')
    oa=accuracy_score(y_tr,y_pred)
    per_class_acc=recall_score(y_tr,y_pred,average=None)
    aa=np.mean(per_class_acc)
    kappa=cohen_kappa_score(y_tr,y_pred)
    time_test_end=time.clock()
    print(per_class_acc)
    print(oa,aa,kappa)
    print((time_train_end-time_train_start),(time_test_end-time_test_start))

    sio.savemat(path+'/label/'+data_name+'/'+'label3.mat',{'label_yunzhi_tr':y_tr,'label_yunzhi_pre':y_pred})
    
with tf.Session() as sess4:
    sess4.run(init)
    step = 1
    epoch = 0
    index = batch_size
    time_train_start=time.clock()
    while epoch<num_epoch:
#    batch_x0 = next_batchx(X_train0,index)
#    batch_x1 = next_batchx(X_train1,index)
#    batch_x2 = next_batchx(X_train2,index)
        batch_x3 = next_batchx(X_train3,index)
#    batch_x4 = next_batchx(X_train4,index)
        batch_y = next_batchy(Y_train,index)
        sess4.run(train_step, feed_dict={x3: batch_x3,y: batch_y,dropout:0.5})
        index = index+batch_size
        step += 1
        if index>X_train3.shape[0]:
            index = batch_size
            epoch=epoch+1
    time_train_end=time.clock()   
    print("Optimization Finished!")
    time_test_start=time.clock()
    y_pred = get_oa(X_test3)
    y_tr=contrary_one_hot(Y_test_yunzhi).astype('int32')
#    temp = y_.eval(feed_dict={x3: X_test3,y: Y_test_yunzhi,dropout:1})
#    y_pred=contrary_one_hot(temp).astype('int32')
#    y_tr=contrary_one_hot(Y_test_yunzhi).astype('int32')
    oa=accuracy_score(y_tr,y_pred)
    per_class_acc=recall_score(y_tr,y_pred,average=None)
    aa=np.mean(per_class_acc)
    kappa=cohen_kappa_score(y_tr,y_pred)
    time_test_end=time.clock()
    print(per_class_acc)
    print(oa,aa,kappa)
    print((time_train_end-time_train_start),(time_test_end-time_test_start))

    sio.savemat(path+'/label/'+data_name+'/'+'label4.mat',{'label_yunzhi_tr':y_tr,'label_yunzhi_pre':y_pred})


#oa = accuracy_score(tf.argmax(y,1), tf.argmax(y_,1))
#print(oa)
#data_all_dir = path + '/data_all/' + data_name
#data = sio.loadmat(data_all_dir + '/' + data_name + '0.mat')
#data_all0 = data['data_all0']
#data_all1 = data['data_all1']
#data_all2 = data['data_all2']
#data_all3 = data['data_all3']
#label_all = data['label_all']
#temp1=y_.eval(feed_dict={x0: data_all0,x1: data_all1,x2: data_all2,x3:data_all3})
#y_pred=contrary_one_hot(temp1).astype('int32')
#plot_label=np.reshape(y_pred,[145,145])
#sio.savemat(path+'/plot/'+data_name+'/'+'plot_label_multi.mat',{'plot_label':plot_label})
#
#dataframe = pd.DataFrame({'variance0':variance0,'variance1':variance1,
#'variance2':variance2,'variance3':variance3,'variance4':variance4,'loc0':train_loc[0],'loc1':train_loc[1]})
#dataframe.to_csv(path+'/mean_var/'+data_name+'/'+'mean_var.csv',index=False)
#sio.savemat(path+'/mean_var/'+data_name+'/'+'mean_var.mat',{'mean_variance0':mean_variance0,'mean_variance1':mean_variance1,
#'mean_variance2':mean_variance2,'mean_variance3':mean_variance3,'loc':train_loc})
