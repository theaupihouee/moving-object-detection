from __future__ import division, print_function
# -*- coding: utf-8 -*-
import cv2 as cv
import math
import datetime 
import random
import numpy as np
import os
from tqdm import tqdm

## [capture]
video_path = 'input_video_opti.mp4'
output_folder = 'C:/Users/pihou/Desktop/hw7_opti'
capture = cv.VideoCapture(video_path) #Path of the video
frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH)) #get frame width of the video
frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)) #get frame height of the video
frame_rate = int(capture.get(cv.CAP_PROP_FPS)) # get number of frames per second
num_frame =int(capture.get(cv.CAP_PROP_FRAME_COUNT)) #get total number of frames
fourcc = int(capture.get(cv.CAP_PROP_FOURCC)) #get fourcc code of the video

#Resizing the video and convert the video to a matrix M, where each column contains pixel information
#You can increase target_width below to get video with better quality, if the code does not take forever to run in your computer
target_width = 300 #width of the video after preprocessing

##Resize 
##from a frame
target_height = int(frame_height*target_width/frame_width)
size = [target_height, target_width, 3] #size of the tensor of each frame
i = 0
Total = target_width*target_height*3 #number of pixels per frame
M = np.zeros((Total,num_frame), dtype = 'uint8')
while i<num_frame:
    ret, frame = capture.read()
    if frame is None:
        break        
    else:
        frame = cv.resize(frame,(target_width,target_height))
        M[:,i] = frame.reshape((1,Total))
        i = i + 1
num_frame = i
M = M[:,:num_frame]
All = Total*num_frame
print('Finish capturing and resizing the video!')

##Apply Stochastic Gradient Descent to solve Frobenius norm minimization
##The following part solves the problem min ||xy^T - M||_F^2, and we recover the low rank matrix to be xy^T
##Parameters
r = 1 #Rank of the low-rank matrix
num_epoch = 0.5 #Total number of epochs, a positive integer or anything between 0 and 1
batch_size = 1 #Batch size of each iteration, a positive integer
learning_rate = 0.01 #Initial learningrate of the SGD, recommended to be between 1e-1 and 1e-2
damp = 0.1 #damping coefficient of the SGD, recommended to be between 0.1 and 0.5
damp_epoch = 0.2 #number of epochs between each damping
if num_epoch<1:
    num_batch = int(All*num_epoch/batch_size) #number of batchs per epoch
else:
    num_batch = int(All/batch_size) #number of batchs per epoch
damp_iterates = int(damp_epoch*All/batch_size) #number of iterations between each damping

##Start the timer
t1 = datetime.datetime.now()

##Random initialization
x = 16*np.random.random((Total,r))
y = 16*np.random.random((num_frame,r))

#SGD update
print('Start running SGD for {} epochs'.format(num_epoch))
epoch_ceil = int(math.ceil(num_epoch))
for i in range(epoch_ceil):
    p = np.random.permutation(All) #random permutation of the M
    print('epoch {} of SGD started'.format(i+1))
    for j in tqdm(range(num_batch)):
        x0 = x
        y0 = y
        for t in range(j*batch_size, (j+1)*batch_size):
            ##Calculate the gradient of the (a,b)th term at (x0,y0)
            cur = p[t]
            b = cur % num_frame
            a = math.floor(cur/num_frame)
            x[a,0] -= learning_rate*y0[b,0]*(y0[b,0]*x0[a,0] - M[a,b])
            y[b,0] -= learning_rate*x0[a,0]*(x0[a,0]*y0[b,0] - M[a,b])
        if j % damp_iterates == damp_iterates - 1:
            learning_rate = learning_rate*damp
t2 = datetime.datetime.now()
print('running time for SGD is {} seconds'.format(t2-t1))
L = np.dot(x,np.transpose(y))#xy^T
L = np.around(L)
L = np.minimum(L,255)
S = np.abs(M - L) #the sparse matrix
L = L.astype('uint8')
S = S.astype('uint8')
## Print running time of the SGD
t2 = datetime.datetime.now()
print('running time for smooth matrix completion is {} s'.format(t2-t1))

# Define the codec and create VideoWriter object

fourcc = cv.VideoWriter_fourcc('m','p','4','v') 
outd_path = os.path.join(output_folder,'rescaled_video.mp4')
outl_path = os.path.join(output_folder,'L.mp4')
outs_path = os.path.join(output_folder,'S.mp4')


f_outD = cv.VideoWriter(outd_path, fourcc, frame_rate, (target_width, target_height))
f_outL = cv.VideoWriter(outl_path, fourcc, frame_rate, (target_width, target_height))
f_outS = cv.VideoWriter(outs_path, fourcc, frame_rate, (target_width, target_height))
i = 0

while(i<num_frame):
    frame = np.reshape(np.asarray(M[:,i]),size)
    f_outD.write(frame)
    frame = np.reshape(np.asarray(L[:,i]),size)
    f_outL.write(frame)
    frame = np.reshape(np.asarray(S[:,i]),size)
    f_outS.write(frame)
    i = i + 1

f_outD.release()
f_outL.release()
f_outS.release()
