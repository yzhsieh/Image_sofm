from sklearn.decomposition import PCA 
import json
import numpy as np
import time
###
feature_path = './all_pixel_feature.txt'
rawlist = []
input_list = []
class inputImage():
    def __init__(self,weight,name,cate):
        self.weight = weight
        self.name = name
        self.cate = cate

    def getWeight(self):
        return self.weight


def init():
    ## initialize inputs
    file = open(feature_path,'r')
    raw = json.load(file)
    for cate in raw:
        for img in raw[cate]:
            ctmp = ([float(a/(80*120)) for a in raw[cate][img]], img, cate)
            tmp = [float(a/(80*120)) for a in raw[cate][img]]
            rawlist.append(tmp)

def put_back():
    for cate in raw

if __name__ == '__main__':
    init_time = time.time()
    print("Loading feature")
    init()
    print(" - Done")
    pca=PCA(n_components='mle')
    newData = pca.fit_transform(rawlist)
    print(newData)
    print("Time elapsed : {}".format(time.time() - init_time))
