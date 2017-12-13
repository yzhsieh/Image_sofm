import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import PIL
import math
import time
import json
import csv
### test for numba
from numba import autojit
from numba import vectorize
from numba import cuda

### parameters
featureNUM = 256*4
inputNUM = 4
nodeNUM = 400
epochNUM = 300
output_path = './output/'
feature_path = './RGB_gray_feature.txt'
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
###


class inputImage():
    def __init__(self,weight,name,cate):
        self.weight = weight
        self.name = name
        self.cate = cate

    def getWeight(self):
        return self.weight


class node():
    def __init__(self, x, y, id):
        self.weight = list()
        for i in range(featureNUM):
            self.weight.append(random.randint(0,1000) / 1000) # set to 0~1
        self.posX = int(x)
        self.posY = int(y)
        self.id = int(id)

    def getWeight(self):
        return self.weight

    def get_distance(self,dist_list):
        rnt = 0
        for i in range(len(dist_list)):
            rnt += (dist_list[i] - self.weight[i])**2
            # print(rnt)
        rnt = rnt ** 0.5
        return rnt

    def get_position(self):
        return [self.posX, self.posY]

    def cal_neighbourhood(self,pos):
        return ((self.posX - pos[0])**2 + (self.posY - pos[1])**2 ) ** 0.5

    def printinfo(self):
        print("position : {},{}".format(self.posX,self.posY))
        print("weight : ",' , '.join([str(tmp) for tmp in self.weight]))


def init_color():
    # mymap = np.zeros((width,heigh,3)).tolist()
    mymap = np.zeros((width,heigh,3),dtype=int)
    for i in range(width):
        for j in range(heigh):
            for k in range(3):
                mymap[i][j][k] = random.randint(0,255)
    return mymap

def save2pic(nx, ny,path = './out.png'):
    # arr = np.rollaxis(arr,2)    
    f, axarr = plt.subplots(nx, ny)
    for i in range(nx):
        for j in range(ny):
            axarr[i, j].imshow(input_list[i*ny + j], cmap='gray')
            axarr[i, j].set_title(str(i*ny + j))

def save_node_pic(t):
    tmp = np.zeros((length,length,3), dtype=int)
    for i in node_list:
        # print( i.getRGB(0))
        tmp[i.posX][i.posY] = i.getRGB(0)
    tmp = np.array(tmp,dtype='uint8')
    img = PIL.Image.fromarray(tmp, 'RGB')
    if type(t) == int:
        img.save(output_path + 'iter' + str(t+1) + '.jpg')
    else:
        img.save(output_path + t + '.jpg')


def readPic(path = './out.png'):
    im = Image.open(path)
    im.convert('RGB')
    arr = np.array(im)
    return arr

def init():
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
            tmp = inputImage([a/(80*120) for a in raw[cate][img]],img,cate)
            input_list.append(tmp)
            

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

def save_model(path = 'model.txt'):
    file = open(path, 'w')
    for item in node_list:
        posX = str(item.posX)
        posY = str(item.posY)
        id = str(item.id)
        weight = '#'.join([str(a) for a in item.weight])
        file.write(id + ',' + posX + ',' + posY + ',' + weight + '\n')

def load_model():
    print('Loading model')
    file = open('model.txt', 'r')
    tmp = csv.reader(file)
    row_count = sum(1 for row in tmp)
    if row_count != nodeNUM:
        print("ERROR!!! number of nodes is not match")
        print("# of node in program: {},  # of node in model.txt : {}".format(nodeNUM, row_count))

    file = open('model.txt', 'r')
    for row in csv.reader(file):
        id = int(row[0])
        node_list[id].id = id
        node_list[id].posX = int(row[1])
        node_list[id].posY = int(row[2])
        tmpweight = [float(a) for a in row[3].split('#')]
        node_list[id].weight = tmpweight
    print(' - Done')

@autojit
def train():
    global radius, lr, tc
    for times in range(epochNUM):
        print("\niteration : ",times)
        print("time elapsed : ",time.time() - init_time)
        for i in input_list:
            print("\rnow : ({}/{})".format(input_list.index(i) + 1,len(input_list)),end='',flush=True)
            BMU = cal_BMU(i.getRGB())
            for n in node_list:
                if n == BMU:
                    continue
                distance = n.cal_neighbourhood(BMU.get_position())
                if distance < radius:
                    # print(n.id)
                    for idx in range(featureNUM):   # update
                        n.weight[idx] = n.weight[idx] + cal_theta(distance,times) * lr * ((i.getRGB()[idx]) - n.weight[idx])
                    # print(n.weight)
        save_node_pic(times)
        lr = lr0 * math.exp(-times/tc)
        radius = MAPradius * math.exp(-times/tc)
        # print('\a',end='',flush=True)

@autojit()
def train_gray(radius, lr, tc):
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
        save_model('./test')
        # print('\a',end='',flush=True)


if __name__ == '__main__':
    print("Initialize")
    init_time = time.time()
    init()
    print(" - Done")
    train_gray(radius, lr, tc)