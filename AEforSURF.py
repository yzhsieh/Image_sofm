from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import json
import numpy as np
###
db_path = './CorelDB2/'
feature_path = './SURF_feature.txt'
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
            tmp = inputImage([a/1 for a in raw[cate][img]],img,cate)
            input_list.append(tmp)
            # PCAlist.append([a/1 for a in raw[cate][img]])
            x_train.append(np.array([a/1 for a in raw[cate][img]]))
def train():
    # this is the size of our encoded representations
    encoding_dim = 128  # 1024 -> 128

    # this is our input placeholder
    input_img = Input(shape=(1024,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(1024, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x=np.array(x_train), y=np.array(x_train), 
                    batch_size=128, epochs=5, verbose=1, callbacks=None, 
                    validation_split=0.0, validation_data=None, shuffle=0, class_weight=None, 
                    sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    
    encoder.save("AEforSURF.h5")
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    print(encoded_imgs)
    print(encoded_imgs.shape)

    print('-----------------')
    print(decoded_imgs)
    print(decoded_imgs.shape)
    

    # encoder.compile(optimizer='SGD', loss='mean_squared_error')
    
    # encoder.fit(x=np.array(x_train), y=np.array(encoded_imgs), 
    #                 batch_size=128, epochs=15, verbose=1, callbacks=None, 
    #                 validation_split=0.0, validation_data=None, shuffle=0, class_weight=None, 
    #                 sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)



def MNIST():
    global x_train, x_test
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

if __name__ == '__main__': 
    # MNIST()
    load_feature()
    x_train = np.array(x_train)
    x_test = np.array(x_train)
    print(np.array(x_train).shape)
    train()