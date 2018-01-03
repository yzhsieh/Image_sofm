import tensorflow
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import backend as K
from keras.datasets import mnist
import json
import numpy as np
import PIL
from PIL import Image
import sys, os
###
cmd = 'train'
d_r = 0.5
featureNUM = 1024
encoding_dim = 200
db_path = './CorelDB2/'
feature_path = './CNN_feature.txt'
# feature_path = './CNN_feature_light.txt'
decoded_img_path = './decoded_img/'
model_path = './model.h5'
autoencoder_model_path = './autoencoder.h5'
encoder_model_path = './encoder.h5'
x_train = []
x_test = []
input_list = []
epochNUM = 500
loss_hist = []
###
ACT = 'tanh'
PADDING = 'same'
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
	global loss_hist
	input_img = Input(shape=(80, 120, 3))  # adapt this if using `channels_first` image data format
	# input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

	x = Conv2D(64, (3, 3), activation=ACT, padding=PADDING)(input_img)
	x = MaxPooling2D((2, 2), padding=PADDING)(x)
	# x = Conv2D(64, (3, 3), activation='relu', padding=PADDING)(x)
	# x = MaxPooling2D((2, 2), padding=PADDING)(x)
	x = Conv2D(32, (3, 3), activation=ACT, padding=PADDING)(x)
	x = MaxPooling2D((2, 2), padding=PADDING)(x)
	x = Conv2D(8, (3, 3), activation=ACT, padding=PADDING)(x)
	encoded = MaxPooling2D((2, 2), padding=PADDING)(x)
	encoder = Model(input_img, encoded)
	# at this point the representation is (4, 4, 8) i.e. 128-dimensional
	x = Conv2D(8, (3, 3), activation=ACT, padding=PADDING)(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(32, (3, 3), activation=ACT, padding=PADDING)(x)
	# x = UpSampling2D((2, 2))(x)
	# x = Conv2D(64, (3, 3), activation='relu', padding=PADDING)(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(64, (3, 3), activation=ACT, padding=PADDING)(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(3, (3, 3), activation='tanh', padding=PADDING)(x)
	autoencoder = Model(input_img, decoded)

	# encoded_input = Input(shape=((20,30,32)))
	decoder_layer = autoencoder.layers[-1]
	# decoder = Model(encoded_input, decoder_layer(encoded_input))
	# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	autoencoder.compile(optimizer='sgd', loss='mse')
	if not os.path.isfile(encoder_model_path):
		print("save initial encoder model")
		encoder.save(encoder_model_path)
	else:
		print("initial encoder model exists")
	if not os.path.isfile(autoencoder_model_path):
		print("save initial autoencoder model")
		encoder.save(autoencoder_model_path)
	else:
		print("initial autoencoder model exists")
	autoencoder.summary()
	earlystopping = EarlyStopping(monitor='loss', patience = 3, verbose=1, mode='auto')
	checkpoint = ModelCheckpoint(filepath='./autoencoder_checkpoint.h5',
								 verbose=1,
								 save_best_only=True,
								 # save_weights_only=True,
								 monitor='loss',
								 mode='auto')

	for iterNOW in range(epochNUM//10):
		### load nodel ###
		encoder = load_model(encoder_model_path)
		autoencoder = load_model(autoencoder_model_path)

		print(">>>>>iterNOW : {}".format(iterNOW))
		hist = autoencoder.fit(x_train, x_train,
						epochs=10,
						batch_size=64,
						)
		### save model###
		print("Saving models")
		encoder.save(encoder_model_path)
		autoencoder.save(autoencoder_model_path)
		# decoder.save("decoder.h5")
		loss_hist.extend(np.array(hist.history['loss']))
		if len(loss_hist) > 10:
			print(' , '.join([str(a)[:8] for a in loss_hist[-10:]]))
		else:
			print(loss_hist)
		print("========================")


def MNIST():
	global x_train, x_test
	(x_train, _), (x_test, _) = mnist.load_data()

	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
	x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

def load(save_pics = 1):
	# global x_train, x_test
	# autoencoder = load_model(autoencoder_model_path)
	autoencoder = load_model(autoencoder_model_path)
	autoencoder.compile(optimizer='sgd', loss='mse')
	autoencoder.summary()
	decoded_imgs = autoencoder.predict(x_test)
	print('-----------------')
	# print(decoded_imgs)
	print(decoded_imgs.shape)
	if save_pics == 1:
		print("Saving decoded imgs")
		idx = 1
		for it in decoded_imgs:
			print('\r Processing : ({}/{})'.format(idx,len(decoded_imgs)), end='')
			tmp = np.array(it)
			tmp = tmp*255
			# print(tmp)
			tmp = np.array(tmp, dtype='uint8')
			img = PIL.Image.fromarray(tmp, 'RGB')
			img.save(decoded_img_path + '{}.jpg'.format(idx))
			idx += 1
	print("ohhhh it seems good so far")
	print("Start to save encode things")
	encoder = load_model(encoder_model_path)
	encoder.summary()
	encoded_imgs = encoder.predict(x_test)
	print(encoded_imgs)
	print(encoded_imgs.shape)
if __name__ == '__main__': 
	### init
	if len(sys.argv) != 1:
		cmd = sys.argv[1]
	if not os.path.exists(decoded_img_path):
		print("crate new directory : {}".format(decoded_img_path))
		os.makedirs(decoded_img_path)
	###
	if cmd == "train":
		print("#################")
		print("## Start train ##")
		print("#################")
		# MNIST()
		print("Loading feature")
		load_feature()
		print(" - done")
		x_train = np.array(x_train)
		x_test = np.array(x_train)
		print(np.array(x_train).shape)
		train()
	elif cmd == "load":
		print("#################")
		print("## Start test  ##")
		print("#################")
		print("Loading feature")
		load_feature()
		print(" - done")
		x_train = np.array(x_train)
		x_test = np.array(x_train)
		load(save_pics=0)
	else:
		print("ERROR!!!")
		print("command : {} not found".format(cmd))
	print("ALL Done!!!!!")