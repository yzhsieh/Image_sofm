import tensorflow
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import json
import numpy as np
###
d_r = 0.5
featureNUM = 1024
encoding_dim = 200
db_path = './CorelDB2/'
feature_path = './CNN_feature.txt'
x_train = []
x_test = []
input_list = []
###

class inputImage():
    def __init__(self,weight,name,cate,tran=0):
        self.weight = weight
        self.name = name
        self.cate = cate
        self.dis = None  ## for BMU to get Images
        self.tran = tran
    def getWeight(self):
        return self.weight
    
    def getName(self):
        return [self.cate, self.name]
    
    def updateWeight(self, arr):
        global weightChangeFlag
        if len(arr) != len(self.weight) and weightChangeFlag == 0:
            print("Change Weight from {} to {}".format(len(self.weight), len(arr)))
            weightChangeFlag = 1
        self.weight = arr


def load_feature():
    file = open(feature_path,'r')
    raw = json.load(file)
    for cate in raw:
        for img in raw[cate]:
            tmp = inputImage(raw[cate][img],img,cate)
            input_list.append(tmp)
            # PCAlist.append([a/1 for a in raw[cate][img]])
            x_train.append(np.array(raw[cate][img]))


def train():
    input_img = Input(shape=(80, 120, 3))  # adapt this if using `channels_first` image data format
    # input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


if __name__ == '__main__': 
    # MNIST()
    print("Loading feature")
    load_feature()
    print(" - done")
    x_train = np.array(x_train)
    x_test = np.array(x_train)
    print(np.array(x_train).shape)
    train()