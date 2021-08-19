import numpy as np
import gzip
import os
import platform
import pickle
from PIL import Image
import skimage.io as io
from array import *
from torchvision import transforms
from skimage import data_dir
import torch
from PIL import Image
import matplotlib.pyplot as plt


class GetDataSet(object):
    def __init__(self, dataSetName):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None
        self._index_in_train_epoch = 0

        A = os.listdir('./data/crop_part1')
        data = [i.split(',') for i in A]
        arr1=[]
        arr2=[]
        loader = transforms.Compose([transforms.ToTensor()]) 
        unloader = transforms.ToPILImage()
        for na in data:
            path1='./data/crop_part1/'
            str_convert = ''.join(na)
            path=path1+str_convert
            img =Image.open(path) .convert('L')
            #tensor=img_to_tensor(img)   
            arr = []
            for i in range(200):
                for j in range(200):
                    pixel = 1.0 - float(img.getpixel((j, i)))
                    arr.append(pixel)
            arr1.append(np.array(arr))
            #img = Image.open(path).convert('RGB')
            #img = loader(img).unsqueeze(0)
            #arr1.append(img.to(device, torch.float))
        t1=torch.Tensor(arr1)
        t1=t1.unsqueeze(0)
        t1=t1.unsqueeze(0)
        #print(t1.shape)
       

        for filename in A:
            p1,p2,p3,p4 = (filename.split('_'))
            label=[int(p1),int(p2),int(p3)]
            arr2.append(label)
            
        #data_image = np.asarray(arr1,dtype='float'
        #data_label = np.asarray(arr2,dtype='float')
          
        #data_image = arr1
        data_image= t1
        data_label = arr2
        
        train_images = data_image[0:6000]
        train_labels = data_label[0:6000]
        test_images = data_image[6001:9001]
        test_labels = data_label[6001:9001]
        self.train_data = train_images
        self.train_label = train_labels
        self.test_data = test_images
        self.test_label = test_labels
    
        #print(data_image)
if __name__=="__main__":
    'test data set'
    UTKDataSet = GetDataSet('crop_part1') 
    #print(UTKDataSet.train_data[1])
    #print(UTKDataSet.test_data)
    
