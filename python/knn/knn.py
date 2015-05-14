# -*- coding: utf-8 -*-
"""
使用KNN分类器的手写识别系统 只能识别数字0到9。
需要识别的数字使用图形处理软件，处理成具有相同的色彩和大小：宽髙是32像素X32像素的黑白图像，这里已经将将图像转换为文本格式。
训练数据中每个数字大概有200个样本，
程序中将图像样本格式化处理为向量，即一个把一个32x32的二进制图像矩阵转换为一个1x1024的向量
"""

from numpy import *
import os

def img_to_vector(filename):
    row_length = 32
    col_length = 32
    image_vector = zeros((1, row_length * col_length))
    image_file = open(filename)
    for row in range(row_length):
        line = image_file.readline()
        for col in range(col_length):
            image_vector[0, row * col_length + col] = int(line[col])
    return image_vector

def read_dataset(dataset_path):
    datalist = os.listdir(dataset_path)
    dataset_count = len(datalist)
    
    data_x = zeros((dataset_count, 1024))
    data_y = []
    for i in range(dataset_count):
        filename = datalist[i]
        
        # get train_x 
        data_x[i, :] = img_to_vector(dataset_path + filename)
        
        # get label from file name,  such as 1_17.txt return 1
        label = int(filename.split('_')[0])
        data_y.append(label)
    return data_x, data_y
        
# load data set
def load_dataset():
    ## Getting training set and test set
    dataset_path = 'digits/'
    train_x, train_y = read_dataset(dataset_path + 'trainingDigits/')
    test_x, test_y = read_dataset(dataset_path + 'testDigits/')
    return train_x, train_y, test_x, test_y
 
# classify using kNN
def knn_classify(test_x, train_x, train_y, k):
    train_count = train_x.shape[0] 

    ## step 1: calculate Euclidean distance
    diff = tile(test_x, (train_count, 1)) - train_x
    squared_diff = diff ** 2 
    squared_dist = sum(squared_diff, axis=1)
    distance = squared_dist ** 0.5

    ## step 2: sort the distance
    sorted_distance = argsort(distance)

    class_count_map = {} 
    for i in xrange(k):
        ## step 3: choose the min k distance
        vote_label = train_y[sorted_distance[i]]

        ## step 4: count the times labels occur
        class_count_map[vote_label] = class_count_map.get(vote_label, 0) + 1

    ## step 5: the max voted class will return
    max_count, max_index = 0, -1
    for key, value in class_count_map.items():
        if value > max_count:
            max_count, max_index = value, key
    return max_index 
   
def predict_hand_writing():
    
    ## step 1: load data
    train_x, train_y, test_x, test_y = load_dataset()
    
    ## step 2: predicting
    test_count = test_x.shape[0]
    match_count = 0
    for i in range(test_count):
        predict_y = knn_classify(test_x[i], train_x, train_y, 3)
        if predict_y == test_y[i]:
            match_count += 1
    accuracy = float(match_count) / test_count
    
    print('The classify accuracy is %.2f%%' % (accuracy * 100))
    
predict_hand_writing()        