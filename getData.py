import numpy as np
import gzip
import os
import platform
import pickle
from PIL import Image
import skimage.io as io
from array import *


class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'crop_part1':
            self.UTKDataSetConstruct(isIID)
        else:
            pass


    def UTKDataSetConstruct(self, isIID):
        #批量读取图片
        from skimage import data_dir
        #str= './data/UTKFace/*.jpg'
        A = os.listdir('./data/crop_part1')
        data = [i.split(',') for i in A]
        #coll = io.ImageCollection(str)
        # 读取图片转成灰度格式
        arr1=[]
        arr2=[]
        for na in data:
            #print(na)
            path1='./data/crop_part1/'
            str_convert = ''.join(na)
            path=path1+str_convert
            img =Image.open(path).convert("RGB")
            tensor=torchvision.transforms.functional.to_tensor(img)
    
            arr1.append(tensor)

        for filename in A:
            #print(filename)
            p1,p2,p3,p4 = (filename.split('_'))
            label=[int(p1),int(p2),int(p3)]
            arr2.append(label)
            
        data_image = np.asarray(arr1,dtype='float')
        data_label = np.asarray(arr2,dtype='float')
        train_images = data_image[0:6000]
        train_labels = data_label[0:6000]
        test_images = data_image[6001:9001]
        test_labels = data_label[6001:9001]
        
 

        #train_images = train_images.astype(np.float32)
        #train_images = np.multiply(train_images, 1.0 / 255.0)
        #test_images = test_images.astype(np.float32)
        #test_images = np.multiply(test_images, 1.0 / 255.0)

   

        self.train_data = train_images
        self.train_label = train_labels
        self.test_data = test_images
        self.test_label = test_labels
    

if __name__=="__main__":
    'test data set'
    #UTKDataSet = GetDataSet('crop_part1', True) # test NON-IID
    #if type(UTKDataSet.train_data) is np.ndarray and type(UTKDataSet.test_data) is np.ndarray and \
     #       type(UTKDataSet.train_label) is np.ndarray and type(UTKDataSet.test_label) is np.ndarray:
     #   print('the type of data is numpy ndarray')
    #else:
     #   print('the type of data is not numpy ndarray')
    #print('the shape of the train data set is {}'.format(UTKDataSet.train_data.shape))
    #print('the shape of the test data set is {}'.format(UTKtDataSet.test_data.shape))
    print(mnistDataSet.train_label)
    print(mnistDataSet.test_label)
 

