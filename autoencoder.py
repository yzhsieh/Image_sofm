import tensorflow
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.datasets import mnist
import json
import numpy as np
import PIL
from PIL import Image
import sys
###
cmd = 'train'
d_r = 0.5
featureNUM = 1024
encoding_dim = 200
db_path = './CorelDB2/'
feature_path = './CNN_feature.txt'
# feature_path = './CNN_feature_light.txt'
decoded_img_path = './decoded_img/'
x_train = []
x_test = []
input_list = []
###
ACT = 'tanh'
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
        print('.',end='',flush=True)
        for img in raw[cate]:
            tmp = inputImage(raw[cate][img],img,cate)
            input_list.append(tmp)
            # PCAlist.append([a/1 for a in raw[cate][img]])
            x_train.append(np.array(raw[cate][img]))


def train():
    input_img = Input(shape=(80, 120, 3))  # adapt this if using `channels_first` image data format
    # input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(64, (3, 3), activation=ACT, padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation=ACT, padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    encoder = Model(input_img, encoded)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(32, (3, 3), activation=ACT, padding='same')(encoded)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation=ACT, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)

    decoder_layer = autoencoder.layers[-1]

    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.compile(optimizer='sgd', loss='mse')
    earlystopping = EarlyStopping(monitor='loss', patience = 20, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(filepath='./IDontKonwWhatThisIs',
                                 verbose=1,
                                 save_best_only=True,
                                 # save_weights_only=True,
                                 monitor='loss',
                                 mode='auto')
    autoencoder.fit(x_train, x_train,
                epochs=250,
                batch_size=64,
                callbacks=[earlystopping,checkpoint])
    # autoencoder.fit(x=np.array(x_train), y=np.array(x_train), 
                    # batch_size=128, epochs=5, verbose=1, callbacks=None, 
                    # validation_split=0.0, validation_data=None, shuffle=0, class_weight=None, 
                    # sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    decoded_imgs = autoencoder.predict(x_test)
    
    # print(encoded_imgs)
    # print(encoded_imgs.shape)
    print('-----------------')
    print(decoded_imgs)
    print(decoded_imgs.shape)
    print("Saving decoded imgs")
    idx = 1
    for it in decoded_imgs:
    	tmp = np.array(it)
    	tmp = tmp*255
    	print(tmp)
    	tmp = np.array(tmp, dtype='uint8')
    	img = PIL.Image.fromarray(tmp, 'RGB')
    	img.save(decoded_img_path + '{}.jpg'.format(idx))
    	idx += 1

def MNIST():
    global x_train, x_test
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

if __name__ == '__main__': 
    # MNIST()
    print("Loading feature")
    load_feature()
    print(" - done")
    x_train = np.array(x_train)
    x_test = np.array(x_train)
    print(np.array(x_train).shape)
    train()