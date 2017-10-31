import os
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sc
import numpy as np
import random
import pandas as pd
import time
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# this function is used to transfer one column label to one hot label
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(int(n_values))[np.array(y_, dtype=np.int32)]

#EEG eegmmidb person dependent raw data mixed read
feature = sc.loadmat(os.path.join(__location__, 'sample.mat'))##10 person, 5 class(0-7), person independent, 8*20W, samples, 0-51 features, 52: class label
all = feature['filtered']



##Shuffle reading
np.random.shuffle(all)
all=all[:28000,0:193] #1:64 Alpha, 65:128 Beta, 129:192: Raw 193: label
print all.shape


# use one of the subject as example
train_data=all[0:21000] #1 million samples
test_data=all[21000:28000]
np.random.shuffle(train_data)  # mix eeg_all
np.random.shuffle(test_data)


no_fea=64
n_steps=1

feature_training_task1 =train_data[:,0:no_fea]          #Sub-tasks - Alpha
feature_training_task2 =train_data[:,no_fea:no_fea*2]   #Sub-tasks - Beta
feature_training_join =train_data[:,no_fea*2:no_fea*3]  #join-tasks - Raw

feature_training_task1 =feature_training_task1.reshape([21000,n_steps,no_fea/n_steps])
feature_training_task2 =feature_training_task2.reshape([21000,n_steps,no_fea/n_steps])
feature_training_join =feature_training_join.reshape([21000,n_steps,no_fea/n_steps])

feature_testing_task1 =test_data[:,0:no_fea]
feature_testing_task2 =test_data[:,no_fea:no_fea*2]
feature_testing_join =test_data[:,no_fea*2:no_fea*3]

feature_testing_task1 =feature_testing_task1.reshape([7000,n_steps,no_fea/n_steps])
feature_testing_task2 =feature_testing_task2.reshape([7000,n_steps,no_fea/n_steps])
feature_testing_join =feature_testing_join.reshape([7000,n_steps,no_fea/n_steps])

label_training =train_data[:,no_fea*3]
label_training =one_hot(label_training)
label_testing =test_data[:,no_fea*3]
label_testing =one_hot(label_testing)
print label_training
print all.shape


a=feature_training_task1
b=feature_testing_task1

## parameters
########################################
nodes1=64
nodes2=64
nodesjoin=64
lameda=0.004
lr=0.001
iterations=3500

batch_size=7000
train_alpha=[]
n_group=3
for i in range(n_group):
    f =a[(0+batch_size*i):(batch_size+batch_size*i)]
    train_alpha.append(f)
print "feature alpha"
print (train_alpha[0].shape)

a=feature_training_task2
train_beta=[]
for i in range(n_group):
    f =a[(0+batch_size*i):(batch_size+batch_size*i)]
    train_beta.append(f)
print "feature beta"
print (train_beta[0].shape)

a=feature_training_join
train_join=[]
for i in range(n_group):
    f =a[(0+batch_size*i):(batch_size+batch_size*i)]
    train_join.append(f)
print "feature join"
print (train_join[0].shape)

train_label=[]
for i in range(n_group):
    f =label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
    train_label.append(f)
print "label"
print (train_label[0].shape)




# hyperparameters
n_inputs = no_fea/n_steps  # MNIST data input (img shape: 11*99)
# n_steps =  # time steps
n_hidden1_task1 = nodes1   # neurons in hidden layer
n_hidden2_task1 = nodes1
n_hidden1_join = nodesjoin

n_hidden1_task2 = nodes2
n_hidden2_task2 = nodes2
n_classes = 6   # MNIST classes (0-9 digits)


# tf Graph input

x1 = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
x2 = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
xjoin =tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="features")
y = tf.placeholder(tf.float32, [None, n_classes])
#y2 = tf.placeholder(tf.float32, [None, n_classes])


# Define weights and biases
########################################
weights = {

'in_join': tf.Variable(tf.random_normal([n_inputs,n_hidden1_join]), trainable=True),
'n_hidden1_join': tf.Variable(tf.random_normal([n_hidden1_join, n_hidden1_join]), trainable=True),
# (28, 128)
'in_task1': tf.Variable(tf.random_normal([n_inputs+n_hidden1_join,n_hidden1_task1]), trainable=True),
'n_hidden1_task1': tf.Variable(tf.random_normal([n_hidden1_task1+n_hidden1_task1, n_hidden1_task1]), trainable=True),
#(128,128)
'in_task2': tf.Variable(tf.random_normal([n_inputs+n_hidden1_join,n_hidden1_task2]), trainable=True),
'n_hidden1_task2': tf.Variable(tf.random_normal([n_hidden1_task2+n_hidden1_task1, n_hidden1_task2 ]), trainable=True),



# (128, 10)
'out_task1': tf.Variable(tf.random_normal([n_hidden1_task1, n_classes]), trainable=True),
'out_task2': tf.Variable(tf.random_normal([n_hidden1_task2, n_classes]), trainable=True),
'out_task_join': tf.Variable(tf.random_normal([n_hidden1_task1, n_classes]), trainable=True),
}

biases = {
# (128, )
'in_task1': tf.Variable(tf.constant(0.1, shape=[n_hidden1_task1])),
'task1': tf.Variable(tf.constant(0.1, shape=[n_hidden2_task1 ])),

'in_task2': tf.Variable(tf.constant(0.1, shape=[n_hidden1_task2])),
'task2': tf.Variable(tf.constant(0.1, shape=[n_hidden1_task2])),

'in_join': tf.Variable(tf.constant(0.1, shape=[n_hidden1_join])),
'join': tf.Variable(tf.constant(0.1, shape=[n_hidden1_join])),

# (10, )
'out_task1': tf.Variable(tf.constant(0.1, shape=[n_classes ]), trainable=True),
'out_task2': tf.Variable(tf.constant(0.1, shape=[n_classes ]), trainable=True),
'out_task_join': tf.Variable(tf.constant(0.1, shape=[n_classes]), trainable=True)
}

# hidden layer for input to cell
########################################

# transpose the inputs shape from
# X ==> (128 batch * 28 steps, 28 inputs)
X1 = tf.reshape(x1, [-1, n_inputs])
X2 = tf.reshape(x2, [-1, n_inputs])
X_join = tf.reshape(xjoin, [-1, n_inputs])
# into hidden
# X_in = (128 batch * 28 steps, 128 hidden)
X_hidd1_join = tf.sigmoid(tf.matmul(X_join, weights['in_join']) + biases['in_join'])
X_hidd1_task1 = tf.sigmoid(tf.matmul(tf.concat([X1,X_hidd1_join],axis=1), weights['in_task1']) + biases['in_task1'])
X_hidd1_task2 = tf.sigmoid(tf.matmul(tf.concat([X2,X_hidd1_join],axis=1), weights['in_task2']) + biases['in_task2'])
# X_hidd1 = tf.matmul(X, weights['in']) + biases['in']
X_hidd2_join = tf.matmul(X_hidd1_join, weights['n_hidden1_join']) + biases['join']
X_hidd2_task1 = tf.matmul(tf.concat([X_hidd1_task1,X_hidd2_join],axis=1), weights['n_hidden1_task1']) + biases['task1']
X_hidd2_task2 = tf.matmul(tf.concat([X_hidd1_task2,X_hidd2_join],axis=1), weights['n_hidden1_task2']) + biases['task2']


X_hidd3_join = tf.matmul(X_hidd2_join, weights['n_hidden1_join']) + biases['join']
X_hidd3_task1 = tf.matmul(tf.concat([X_hidd2_task1,X_hidd3_join],axis=1), weights['n_hidden1_task1']) + biases['task1']
X_hidd3_task2 = tf.matmul(tf.concat([X_hidd2_task2,X_hidd3_join],axis=1), weights['n_hidden1_task2']) + biases['task2']

X_hidd4_join = tf.matmul(X_hidd3_join, weights['n_hidden1_join']) + biases['join']
X_hidd4_task1 = tf.matmul(tf.concat([X_hidd3_task1,X_hidd4_join],axis=1), weights['n_hidden1_task1']) + biases['task1']
X_hidd4_task2 = tf.matmul(tf.concat([X_hidd3_task2,X_hidd4_join],axis=1), weights['n_hidden1_task2']) + biases['task2']


X_taskjoin = tf.reshape(X_hidd4_join, [-1, n_steps, n_hidden1_join])
X_task1 = tf.reshape(X_hidd4_task1, [-1, n_steps,n_hidden1_task1])
X_task2 = tf.reshape(X_hidd4_task2, [-1, n_steps, n_hidden1_task2])


#share lstm cell
lstm_cell_share = tf.contrib.rnn.BasicLSTMCell(n_hidden1_join, forget_bias=1, state_is_tuple=True)
init_statejoin = lstm_cell_share.zero_state(batch_size, dtype=tf.float32)
with tf.variable_scope('lstmshare'):
	outputsjoin, final_state1 = tf.nn.dynamic_rnn(lstm_cell_share, X_taskjoin, initial_state=init_statejoin, time_major=False)
outputsjoin = tf.unstack(tf.transpose(outputsjoin, [1, 0, 2]))
outputsjoin=outputsjoin[-1]

X_hidd4_task1=tf.stack((X_hidd4_task1,outputsjoin))
X_hidd4_task2=tf.stack((X_hidd4_task2,outputsjoin))
# X_in ==> (128 batch, 28 steps, 128 hidden)


# cell
##########################################

# basic LSTM Cell.
# lstm_cell_join = tf.contrib.rnn.BasicLSTMCell(n_hidden1_join, forget_bias=1, state_is_tuple=True)
lstm_cell_11 = tf.contrib.rnn.BasicLSTMCell(n_hidden1_task1, forget_bias=1, state_is_tuple=True)
lstm_cell_12 = tf.contrib.rnn.BasicLSTMCell(n_hidden1_task1, forget_bias=1, state_is_tuple=True)

lstm_cell_21 = tf.contrib.rnn.BasicLSTMCell(n_hidden1_task2, forget_bias=1, state_is_tuple=True)
lstm_cell_22 = tf.contrib.rnn.BasicLSTMCell(n_hidden1_task2, forget_bias=1, state_is_tuple=True)

lstm_cell1 = tf.contrib.rnn.MultiRNNCell([lstm_cell_11, lstm_cell_12], state_is_tuple=True)
lstm_cell2 = tf.contrib.rnn.MultiRNNCell([lstm_cell_21, lstm_cell_22], state_is_tuple=True)
# lstm cell is divided into two parts (c_state, h_state)

# init_statejoin = lstm_cell_join.zero_state(batch_size, dtype=tf.float32)
init_state1 = lstm_cell1.zero_state(batch_size, dtype=tf.float32)
init_state2 = lstm_cell2.zero_state(batch_size, dtype=tf.float32)
# You have 2 options for following step.

with tf.variable_scope('lstm1'):
	outputs1, final_state1 = tf.nn.dynamic_rnn(lstm_cell1, X_task1, initial_state=init_state1, time_major=False)
with tf.variable_scope('lstm2'):
	outputs2, final_state2 = tf.nn.dynamic_rnn(lstm_cell2, X_task2, initial_state=init_state2, time_major=False)


# hidden layer for output as the final results
#############################################
merge = tf.concat([outputs1, outputs2], axis=1)

# unpack to list [(batch, outputs)..] * steps
outputs1 = tf.unstack(tf.transpose(outputs1, [1, 0, 2]))    # states is the last outputs
outputs2 = tf.unstack(tf.transpose(outputs2, [1, 0, 2]))   
results1 = tf.matmul(outputs1[-1], weights['out_task1']) + biases['out_task1']
results2 = tf.matmul(outputs2[-1], weights['out_task2']) + biases['out_task2']

#the final results
#############################################
output_merge = tf.unstack(tf.transpose(merge, [1, 0, 2])) 
results_merge = tf.matmul(output_merge[-1], weights['out_task_join']) + biases['out_task_join']

#l2 optimisation
#############################################
lamena =lameda
lr=lr
l2 = lamena * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())  # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=results_merge, labels=y))+l2  # Softmax loss
Optimiser = tf.train.AdamOptimizer(lr).minimize(cost)
pred_result =tf.argmax(results_merge, 1)
softmaxed_logits = tf.nn.softmax(results_merge)
label_true =tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(results_merge, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



init = tf.global_variables_initializer()
local = tf.local_variables_initializer()
# Joint Training
# Calculation (Session) Code
#############################################


# open the session

with tf.Session() as sess:
    # merged = tf.merge_all_summaries()
    # writer = tf.train.SummaryWriter("logs/", sess.graph)
    sess.run(init)
    sess.run(local)
    saver = tf.train.Saver()
    step = 0
    start = time.clock()

    while step < iterations:# 1500 iterations
        for i in range(n_group):
            #print (train_label[1].shape)
            sess.run(Optimiser, feed_dict={
                xjoin:train_join[i],
                x1: train_alpha[i],
                x2: train_beta[i],
                y: train_label[i],
                })

        if sess.run(accuracy, feed_dict={
                xjoin:feature_testing_join,
                x1: feature_testing_task1,
                x2: feature_testing_task2,
                y: label_testing,
        	})>0.99:
            print(
            "The lamda is :", lamena, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ",
            sess.run(accuracy, feed_dict={
                xjoin:feature_testing_join,
                x1: feature_testing_task1,
                x2: feature_testing_task2,
                y: label_testing,
            }))
            break
            
        if step % 10 == 0:
            hh=sess.run(accuracy, feed_dict={
                xjoin:feature_testing_join,
                x1: feature_testing_task1,
                x2: feature_testing_task2,
                y: label_testing,
            })
            print("The lamda is :",lamena,", Learning rate:",lr,", The step is:",step,", The accuracy is:", hh)



            print("The cost is :",sess.run(cost, feed_dict={
                xjoin:feature_testing_join,
                x1: feature_testing_task1,
                x2: feature_testing_task2,
                y: label_testing,
            }))
            
        step += 1
    endtime=time.clock()
    print "run time:", endtime-start

