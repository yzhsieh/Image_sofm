import random
import numpy as np
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import math
import time
import json
import csv
from operator import attrgetter
from numba import autojit

### parameters
featureNUM = 256 * 4 
nodeNUM = 400
output_path = './output/'
model_path = './model_RGBG.txt'
db_path = './CorelDB2/'
input_path = './RGB_gray_feature.txt'
### global variables
node_list = list()
input_list = list()
MAPradius = (nodeNUM ** 0.5) / 2
radius = (nodeNUM ** 0.5) / 2
lr0 = 0.1   # initial learning rate
lr = 0.1   # learning rate
length = int(nodeNUM ** 0.5)
init_time = time.time()
###
class inputImage():
    def __init__(self,weight,name,cate):
        self.weight = weight
        self.name = name
        self.cate = cate
        self.dis = None  ## for BMU to get Images
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
    file = open(input_path,'r')
    raw = json.load(file)
    for cate in raw:
        for img in raw[cate]:
            tmp = inputImage([a/(80*120) for a in raw[cate][img]],img,cate)
            # print(len(tmp.weight))
            input_list.append(tmp)
            

    ## initialize radius
    radius = (nodeNUM ** 0.5) / 2

def cal_BMU(weight):
    dist_list = list()
    for n in node_list:
        dist_list.append(n.get_distance(weight))
    return node_list[np.argmin(dist_list)]

def cal_theta(dist,t):
    return math.exp(-dist**2/(2*cal_sigma(t)**2))

def cal_sigma(t):
    return MAPradius*math.exp(-t/tc)

def save_model():
    file = open('model.txt', 'w')
    for item in node_list:
        posX = str(item.posX)
        posY = str(item.posY)
        id = str(item.id)
        weight = '#'.join([str(a) for a in item.weight])
        file.write(id + ',' + posX + ',' + posY + ',' + weight + '\n')

def load_model():
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

@autojit
def show_node():
    print("Processing : showing node")
    f, axarr = plt.subplots(20, 20, figsize=(16,16))
    print(" - Start")
    for n in node_list:
        print('\r Processing : ', n.id, end='',flush=True)
        aim = math.inf
        imgptr = None
        for img in input_list:
            if aim > n.get_distance(img.weight):
                aim =n.get_distance(img.weight)
                imgptr = img
        im = Image.open(db_path + imgptr.cate + '/' + imgptr.name)
        axarr[n.posX, n.posY].imshow(np.array(im))
        axarr[n.posX, n.posY].axis('off')
        axarr[n.posX, n.posY].set_title(n.id)
    f.savefig('./node_result.png')

def cal_node_similiarity():
    print("Calculating node similiarity")
    heat_arr = np.zeros((nodeNUM, nodeNUM)).tolist()
    for i in range(nodeNUM):
        # print('\r now : ',i+1)
        for j in range(nodeNUM):
            heat_arr[i][j] = node_list[i].get_distance(node_list[j].weight)
    
    file = open('./node_similiarity.csv', 'w')
    for i in heat_arr:
        tmp = ','.join([str(a) for a in i]) 
        file.write(tmp + '\n')
    # plt.imshow(heat_arr, cmap='hot')
    # plt.show()
    
    print(' - Done')
    
        

def test(test_path):
    print("Processing test data : ", test_path[2:])
    global input_list
    stat = loadimg(test_path)
    BMU = cal_BMU(stat)
    print("BMU id : ",BMU.id)
    for img in input_list:
        img.dis = BMU.get_distance(img.getWeight())
    ## sort the intput list
    input_list = sorted(input_list, key=attrgetter('dis'))
    save2pic(8, 8,'out_' + test_path[2:-4] + '.png')
    print(" - Done")

def loadimg(path):
    im = Image.open(path)
    arr = np.array(im)
    ### create dict
    ### for 3 color + gray
    stat = np.zeros((featureNUM),dtype=int).tolist()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            tmp = int(sum(arr[i][j]) / 3)
            stat[tmp] += 1
            #R
            tmp = arr[i][j][0]
            stat[tmp + 256] += 1
            #G
            tmp = arr[i][j][1]
            stat[tmp + 256*2] += 1
            #B
            tmp = arr[i][j][2]
            stat[tmp + 256*3] += 1
    return stat
if __name__ == '__main__':
    print("Initialize")
    init_time = time.time()
    init()
    print(" - Done")
    load_model()
    # show_node()
    # cal_node_similiarity()
    test('./168087.jpg')
    test('./326050.jpg')
    test('./618013.jpg')
    # save2pic(5,5)
    # train_gray()