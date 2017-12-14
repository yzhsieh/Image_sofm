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
DEBUG = 0
featureNUM = 256 * 4 
nodeNUM = 400
output_path = './output/'
model_path = './new_model.txt'
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

data_clusterNUM = 64
same_threshold = 30



###
class inputImage():
    def __init__(self,weight,name,cate):
        self.weight = weight
        self.name = name
        self.cate = cate
        self.dis = None  ## for BMU to get Images
    def getWeight(self):
        return self.weight
    
    def getName(self):
        return [self.cate, self.name]


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
    # for img in input_list:
        # img.dis = BMU.get_distance(img.getWeight())
    ## sort the intput list
    # input_list = sorted(input_list, key=attrgetter('dis'))
    # save2pic(8, 8,'out_' + test_path[2:-4] + '.png')
    # BMU.printCluster('out_' + test_path[2:-4] + '.png')
    printCluster(BMU, 'out_' + test_path[2:-4] + '.png')
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

@autojit
def same(list1,list2):
    sameNUM = 0
    for i in list1:
        for j in list2:
            if(i==j):
                sameNUM = sameNUM +1
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
    print(" - Done")





if __name__ == '__main__':
    if DEBUG:
        print(">> DEBUG mode is on <<")
    print("Initialize")
    init_time = time.time()
    init()
    print(" - Done")
    load_model()
    matching(input_list)
    # show_node()
    # cal_node_similiarity()
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
    # save2pic(5,5)
    # train_gray()