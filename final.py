import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
import PIL
from PIL import Image
import math
import time
import json
import csv
from operator import attrgetter
import sys
### test for numba
from numba import autojit
from numba import vectorize
from numba import cuda
import tensorflow
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import backend as K
from keras.datasets import mnist
import json
import os


DEBUG = 0
featureNUM = 1200
inputNUM = 4
nodeNUM = 400
epochNUM = 300
# PCAcomponentsNUM = 1024
output_path = './output/'
model_path = './CNN_model_50.txt'
feature_path = './weights.txt'
data_clusterNUM = 64
same_threshold = 50
# PCAlist = []
# PCAcomponentsNUM = 1024
weightChangeFlag = 0
###
### global variables
node_list = list()
input_list = list()
MAPradius = (nodeNUM ** 0.5) / 2
radius = (nodeNUM ** 0.5) / 2
lr0 = 0.95   # initial learning rate
lr = lr0   # learning rate
tc = epochNUM / math.log10(MAPradius)   #time constant (lumbda)
length = int(nodeNUM ** 0.5)
init_time = time.time()
PCAlist = []
###
weightChangeFlag = 0
encode_epochNUM = 500
# feature_path = './CNN_feature_light.txt'
###

db_path = './CorelDB2/'
encode_feature_path = './CNN_feature.txt'
encoder_model_path = './encoder.h5'
weight_path = './weights.txt'
autoencoder_model_path = './autoencoder.h5'
encoding_dim = 200
x_train = []
x_test = []
operation = 'train'
d_r = 0.5
loss_hist = []
ACT = 'tanh'
PADDING = 'same'
decoded_img_path = './decoded_img/'


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
		# if len(arr) != len(self.weight) and weightChangeFlag == 0:
			# print("Change Weight from {} to {}".format(len(self.weight), len(arr)))
			# weightChangeFlag = 1
		self.weight = arr


def load_feature():
	file = open(encode_feature_path,'r')
	raw = json.load(file)
	for cate in raw:
		print('.',end='',flush=True)
		for img in raw[cate]:
			tmp = inputImage(raw[cate][img],img,cate)
			input_list.append(tmp)
			# PCAlist.append([a/1 for a in raw[cate][img]])
			x_train.append(np.array(raw[cate][img]))

def save_encoded_weight():
	odict = {}
	file = open(weight_path, 'w')
	now = None
	tmpdict ={}
	for it in input_list:
		print(">>>>>>NOW cate : {}, name : {}".format(it.cate, it.name))
		if now != it.cate:
			if len(tmpdict) == 0:
				print(" - initial")
				now = it.cate
				tmpdict[it.name] = it.weight
			else:
				print(" - new cate : {}".format(it.cate))
				odict[now] = tmpdict
				tmpdict = {}
				now = it.cate
				tmpdict[it.name] = it.weight
		else:
			print(" - same cate")
			tmpdict[it.name] = it.weight
	print("Dumping to weights.txt")
	json.dump(odict, file)

def encode_train():
	global loss_hist
	input_img = Input(shape=(80, 120, 3))  # adapt this if using `channels_first` image data format
	# input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

	x = Conv2D(64, (3, 3), activation=ACT, padding=PADDING)(input_img)
	x = MaxPooling2D((2, 2), padding=PADDING)(x)
	x = Conv2D(32, (3, 3), activation=ACT, padding=PADDING)(x)
	x = MaxPooling2D((2, 2), padding=PADDING)(x)
	x = Conv2D(8, (3, 3), activation=ACT, padding=PADDING)(x)
	encoded = MaxPooling2D((2, 2), padding=PADDING)(x)
	encoder = Model(input_img, encoded)


	# at this point the representation is (4, 4, 8) i.e. 128-dimensional
	x = Conv2D(8, (3, 3), activation=ACT, padding=PADDING)(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(32, (3, 3), activation=ACT, padding=PADDING)(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(64, (3, 3), activation=ACT, padding=PADDING)(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(3, (3, 3), activation='sigmoid', padding=PADDING)(x)
	autoencoder = Model(input_img, decoded)

	# encoded_input = Input(shape=((20,30,32)))
	decoder_layer = autoencoder.layers[-1]
	# decoder = Model(encoded_input, decoder_layer(encoded_input))
	# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	autoencoder.compile(optimizer='sgd', loss='mse')

	'''
	if not os.path.isfile(encoder_model_path):
		print("save initial encoder model")
		encoder.save(encoder_model_path)
	else:
		print("initial encoder model exists")
	if not os.path.isfile(autoencoder_model_path):
		print("save initial autoencoder model")
		encoder.save(autoencoder_model_path)
		compile_flag = 1
	else:
		print("initial autoencoder model exists")
		compile_flag = 0
	'''


	autoencoder.summary()
	earlystopping = EarlyStopping(monitor='loss', patience = 3, verbose=1, mode='auto')
	checkpoint = ModelCheckpoint(filepath='./autoencoder_checkpoint.h5',
								 verbose=1,
								 save_best_only=True,
								 # save_weights_only=True,
								 monitor='loss',
								 mode='auto')

	for iterNOW in range(encode_epochNUM//10):
		### load nodel ###
		if os.path.isfile(encoder_model_path):
			encoder = load_model(encoder_model_path)
		if os.path.isfile(autoencoder_model_path):
			autoencoder = load_model(autoencoder_model_path)
		# if compile_flag == 1:
			# autoencoder.compile(optimizer='sgd', loss='mse')
			# compile_flag = 0

		print(">>>>>iterNOW : {}".format(iterNOW))
		hist = autoencoder.fit(x_train, x_train,
						epochs=10,
						batch_size=64,
						shuffle=False,
						validation_split=0.1
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

def load(save_pics = 1, show_weight = 0):
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
	if show_weight == 1:
		print("Start to save encode things")
		encoder = load_model(encoder_model_path)
		encoder.summary()
		encoded_imgs = encoder.predict(x_test)
		print(encoded_imgs)
		print(encoded_imgs.shape)

def save_weight():
	print("Start to save encode things")
	encoder = load_model(encoder_model_path)
	encoder.summary()
	idx = 1
	length = len(input_list)
	for item in input_list:
		print('\r - processing : {} ({}/{})'.format(item.name, idx, length), end='')
		idx += 1
		weight = item.getWeight()
		weight = np.reshape(weight, (1 ,80,120,3))
		rnt = encoder.predict(weight)
		rnt = np.reshape(rnt, (1200))
		rnt = np.array(rnt, dtype=float)
		rnt = np.round(rnt, decimals=6)
		# print(rnt)
		rnt = np.array(rnt, dtype=float).tolist()
		# rnt = [ '%.6f' % a for a in rnt ]
		# rnt = [ float(a) for a in rnt]
		item.updateWeight(rnt)
	print("Saving weights to file")
	save_encoded_weight()

def encode_test(path):
	im = Image.open(path)
	arr = np.array(im) / 255
	arr = np.reshape(arr, (1,80,120,3))
	autoencoder = load_model(autoencoder_model_path)
	decoded_img = autoencoder.predict(arr)
	tmp = np.array(decoded_img)
	tmp = tmp*255
	tmp = np.array(tmp, dtype='uint8')
	tmp = np.reshape(tmp, (80,120,3))	
	img = PIL.Image.fromarray(tmp, 'RGB')
	img.save(path + '_out.jpg')
	print("Predict {} Done".format(path))

class node():
	def __init__(self, x, y, id):
		self.weight = list()
		for i in range(featureNUM):
			self.weight.append(random.randint(0,1000) / 1000) # set to 0~1
		self.posX = int(x)
		self.posY = int(y)
		self.id = int(id)
		self.cluster = []
		self.category = int(id)

	def getWeight(self):
		return self.weight

	def get_distance(self,dist_list):
		rnt = 0
		# print("DEBUG : len of dist_list = {}, weight = {}".format(len(dist_list),len(self.weight)))
		for i in range(featureNUM):
			rnt += (dist_list[i] - self.weight[i])**2
		rnt = rnt ** 0.5
		return rnt

	def get_position(self):
		return [self.posX, self.posY]

	def cal_neighbourhood(self,pos):
		return ((self.posX - pos[0])**2 + (self.posY - pos[1])**2 ) ** 0.5

	def printinfo(self):
		print("position : {},{}".format(self.posX,self.posY))
		print("weight : ",' , '.join([str(tmp) for tmp in self.weight]))

	def printCluster(self,path):
		f, axarr = plt.subplots(8,8, figsize=(10,12))
		for i in range(8):
			for j in range(8):
				ptr = self.cluster[i*8 + j]
				im = Image.open(db_path + ptr[0] + '/' + ptr[1])
				axarr[i, j].imshow(np.array(im))
				axarr[i, j].axis('off')
				axarr[i, j].set_title(ptr[0] + '\n' + ptr[1])
		f.savefig(path)

def printCluster(nptr,path = './out.png'):
	id = nptr.category
	print_list = []
	for item in node_list:
		if item.category == id:
			for img in item.cluster:
				if img not in print_list:
					print_list.append(img)
	print("   - found images : {}".format(len(print_list)))
	n = len(print_list)
	if n > 80 :
		ny = 12
		nx = n//12 + 1
	else:
		ny = 8
		nx = n//8 + 1
	f, axarr = plt.subplots(nx, ny, figsize=(10,12))
	plt.suptitle("\n cluster ID : {}\n # of img : {}".format(id, len(print_list)),fontsize=22)
	for i in range(nx):
		for j in range(ny):
			if i*ny+j > len(print_list) -1:
				break
			ptr = print_list[i*ny + j]
			im = Image.open(db_path + ptr[0] + '/' + ptr[1])
			axarr[i, j].imshow(np.array(im))
			axarr[i, j].axis('off')
			# axarr[i, j].set_title(ptr[0] + '\n' + ptr[1])
	f.savefig(path)



def save2pic(nx, ny,path = './out.png'):
	# arr = np.rollaxis(arr,2)    
	f, axarr = plt.subplots(nx, ny, figsize=(10,12))
	for i in range(nx):
		for j in range(ny):
			ptr = input_list[i*ny + j]
			im = Image.open(db_path + ptr.cate + '/' + ptr.name)
			axarr[i, j].imshow(np.array(im))
			axarr[i, j].axis('off')
			axarr[i, j].set_title(ptr.cate + '\n' + ptr.name)
	f.savefig(path)


def save_node_pic(t):
	f, axarr = plt.subplots(20, 20, figsize=(20,20))
	for i in range(20):
		for j in range(20):
			ptr = node_list[i*20 + j]
			near = None
			for it in input_list:
				if near == None:
					near = it
				elif ptr.get_distance(near.weight) > ptr.get_distance(it.weight):
					near = it
			im = Image.open(db_path + it.cate + '/' + it.name)								
			axarr[i, j].imshow(np.array(im))
			axarr[i, j].axis('off')
			axarr[i, j].set_title(ptr.cate + '\n' + ptr.name)
	f.savefig('./node_stat_{}.jpg'.format(t))


def readPic(path = './out.png'):
	im = Image.open(path)
	im.convert('RGB')
	arr = np.array(im)
	return arr

def init_train():
	## initialize node
	pos = 0
	edge = nodeNUM ** 0.5
	for i in range(nodeNUM):
		node_list.append(node(pos%edge, pos//edge, pos))
		pos += 1

	## initialize inputs
	file = open(feature_path,'r')
	raw = json.load(file)
	for cate in raw:
		for img in raw[cate]:
			# tmp = inputImage([a/(256) for a in raw[cate][img]],img,cate)
			tmp = inputImage(raw[cate][img],img,cate)
			input_list.append(tmp)
			# PCAlist.append([a/(256) for a in raw[cate][img]])
	print("input length :",len(input_list))
	print("feature length :",len(input_list[0].weight))

	## initialize radius
	radius = (nodeNUM ** 0.5) / 2

def init_test():
	## initialize node
	pos = 0
	edge = nodeNUM ** 0.5
	for i in range(nodeNUM):
		node_list.append(node(pos%edge,pos//edge,pos))
		pos += 1

	## initialize inputs
	file = open(feature_path,'r')
	raw = json.load(file)
	for cate in raw:
		for img in raw[cate]:
			tmp = inputImage([a/1 for a in raw[cate][img]],img,cate)
			input_list.append(tmp)
			PCAlist.append([a/1 for a in raw[cate][img]])
			

	## initialize radius
	radius = (nodeNUM ** 0.5) / 2



@autojit
def cal_BMU(weight):
	dist_list = list()
	for n in node_list:
		dist_list.append(n.get_distance(weight))
	return node_list[np.argmin(dist_list)]

@autojit
def cal_theta(dist,t):
	return math.exp(-dist**2/(2*cal_sigma(t)**2))

@autojit
def cal_sigma(t):
	return MAPradius*math.exp(-t/tc)

def save_node_model(path = 'node_model.txt'):
	file = open(path, 'w')
	for item in node_list:
		posX = str(item.posX)
		posY = str(item.posY)
		id = str(item.id)
		weight = '#'.join([str(a) for a in item.weight])
		file.write(id + ',' + posX + ',' + posY + ',' + weight + '\n')

def load_node_model():
	print('Loading model')
	file = open(model_path, 'r')
	tmp = csv.reader(file)
	row_count = sum(1 for row in tmp)
	# row_featNUM = 0
	if row_count != nodeNUM:
		print("ERROR!!! number of nodes is not match")
		print("# of node in program: {},  # of node in model.txt : {}".format(nodeNUM, row_count))
	# if row_featNUM != featureNUM:
		# print("ERROR!!! number of features is not match")
		# print("# of node in program: {},  # of node in model.txt : {}".format(nodeNUM, row_count))
	file = open(model_path, 'r')
	for row in csv.reader(file):
		id = int(row[0])
		node_list[id].id = id
		node_list[id].posX = int(row[1])
		node_list[id].posY = int(row[2])
		tmpweight = [float(a) for a in row[3].split('#')]
		node_list[id].weight = tmpweight
	print(' - Done')

@autojit()
def train(radius, lr, tc):
	# global radius, lr, tc
	for times in range(epochNUM):
		print("\niteration : ",times)
		print("time elapsed : ",time.time() - init_time)
		for i in input_list:
			print("\rnow : ({}/{})".format(input_list.index(i) + 1,len(input_list)),end='',flush=True)
			if input_list.index(i) == 100:
				print("time used : ", time.time() - init_time)
			BMU = cal_BMU(i.getWeight())
			for n in node_list:
				if n == BMU:
					continue
				distance = n.cal_neighbourhood(BMU.get_position())
				if distance < radius:
					# print(n.id)
					for idx in range(featureNUM):   # update
						n.weight[idx] = n.weight[idx] + cal_theta(distance,times) * lr * ((i.getWeight()[idx]) - n.weight[idx])
					# print(n.weight)
		# save_node_pic(times)
		lr = lr0 * math.exp(-times/tc)
		radius = MAPradius * math.exp(-times/tc)
		print("Saving node model.....")
		save_node_model('./CNN_models/CNN_model_{}.txt'.format(times))
		# print('\a',end='',flush=True)


def test(test_path):
	print("Processing test data : ", test_path[2:])
	global input_list
	stat = loadimg(test_path)
	BMU = cal_BMU(stat)
	print("BMU id : ",BMU.id)
	# for img in input_list:
		# img.dis = BMU.get_distance(img.getWeight())
	## sort the intput list
	# input_list = sorted(input_list, key=attrgetter('dis'))
	save2pic(8, 8,'out_' + test_path[2:-4] + '.png')
	printCluster(BMU, 'out_' + test_path[2:-4] + '.png')
	print(" - Done")

def loadimg(path):
	encoder = load_model(encoder_model_path)
	im = Image.open(path)
	arr = np.array(im)
	weight = np.reshape(arr, (1 ,80,120,3))
	rnt = encoder.predict(weight)
	rnt = np.reshape(rnt, (1200))
	rnt = np.array(rnt, dtype=float)
	# rnt = np.round(rnt, decimals=6)
	#######################
	return rnt

@autojit
def same(list1,list2):
	sameNUM = 0
	for i in list1:
		for j in list2:
			if(i==j):
				sameNUM = sameNUM +1
				continue
	# print("\n",sameNUM)
	return sameNUM


@autojit
def matching(input_list):
	global node_list
	print("Matching")
	tmptime = time.time()    
	for i in node_list:
		if DEBUG:
			if i.id == 2:
				break
		print("\r - now : {}/{}   time : {}".format(i.id, len(node_list), time.time() - tmptime),end='')
		tmptime = time.time()
		# node_cluster = []
		for j in input_list:
			j.dis = i.get_distance(j.getWeight())
		input_list = sorted(input_list, key=attrgetter('dis'))
		for idx in range(data_clusterNUM):
			i.cluster.append(input_list[idx].getName())
	### for debug ###
	if DEBUG:
		for i in node_list:
			i.cluster = node_list[0].cluster
	#################
	print("\n - Matching done")
	print("Node clustering")
	for i in node_list:
		for j in node_list[i.id+1:]:
			if i.category == j.category:
				continue
			print("\r - Processing node {} and node {}".format(i.id,j.id),end='')
			sameNUM = same(i.cluster,j.cluster)
			if(sameNUM>same_threshold):
				ti = i.category
				tj = j.category
				i.category = min(i.category, j.category)
				j.category = i.category
				print("\n change cate id : {} and {} to {}".format(ti, tj, i.category))
				print(" - same NUM : {}".format(sameNUM))
	print(" - Done")

def printClusterinfo():
	clslist = []
	for it in node_list:
		if it.category not in clslist:
			clslist.append(it.category)
	print("Number of cluster : ",len(clslist))
	print(','.join([str(a) for a in clslist]))

def put_back_PCA():
	global input_list
	for idx in range(len(input_list)):
		input_list[idx].updateWeight(PCAlist[idx])

def myPCA():
	global PCAlist
	print("Origin length : {}".format(len(PCAlist[0])))
	pca=PCA(n_components=PCAcomponentsNUM)
	PCAlist = pca.fit_transform(PCAlist)
	print("new length : {}".format(len(PCAlist[0])))



if __name__ == '__main__':
	if len(sys.argv) != 1:
		operation = sys.argv[1]

	if not os.path.exists(decoded_img_path):
		print("crate new directory : {}".format(decoded_img_path))
		os.makedirs(decoded_img_path)


	if operation == 'auto_train':
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
		encode_train()
	elif operation == "auto_load":
		print("#################")
		print("## Start load  ##")
		print("#################")
		print("Loading feature")
		load_feature()
		print(" - done")
		x_train = np.array(x_train)
		x_test = np.array(x_train)
		load(save_pics=1)
	elif operation == 'auto_weight':
		print("#################")
		print("## Save Weight ##")
		print("#################")
		print("Loading feature")
		load_feature()
		print(" - done")
		x_train = np.array(x_train)
		x_test = np.array(x_train)
		save_weight()
	elif operation == 'auto_test':
		print("#################")
		print("## Save  Test  ##")
		print("#################")
		encode_test('./test/1.jpg')
		encode_test('./test/2.jpg')
		encode_test('./test/3.jpg')
		encode_test('./test/4.jpg')
		encode_test('./test/5.jpg')
	elif operation == 'train':
		print("#################")
		print("## Start train ##")
		print("#################")
		print("Initialize")
		init_time = time.time()
		init_train()
		print(" - Done")
		# print("Procressing PCA")
		# myPCA()
		# put_back_PCA()
		# print(" - Done")
		print("Start to train")
		train(radius, lr, tc)
	elif operation == 'test':
		print("################")
		print("## Start test ##")
		print("################")
		if DEBUG:
			print("## DEBUG is on ##")
		print("Initialize")
		init_time = time.time()
		init_test()
		print(" - Done")
		# print("Procressing PCA")
		# myPCA()
		# put_back_PCA()
		# print(" - Done")
		load_node_model()
		matching(input_list)
		printClusterinfo()
		test('./1.jpg')
		test('./2.jpg')
		test('./3.jpg')
		test('./4.jpg')
		test('./5.jpg')
		test('./6.jpg')
		test('./7.jpg')
		test('./8.jpg')
		test('./9.jpg')
		test('./10.jpg')


	else:
		print("ERROR!!!")
		print("command : {} not found".format(operation))
	print("ALL Done!!!!!")



