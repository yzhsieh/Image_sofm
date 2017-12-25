import numpy as np
from matplotlib import pyplot as plt
import csv
import json
from PIL import Image
from numba import autojit
import time
import cv2
### user defined libraries ###
import SURF

def SURF(path, CVthreshold = 10000):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=CVthreshold, upright=True, extended=False)
    (kps, descs) = surf.detectAndCompute(gray, None)
    while len(kps)<16:
        CVthreshold = CVthreshold - 500
        # print("change thres to : ",CVthreshold)                
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=CVthreshold, upright=True, extended=False)
        (kps, descs) = surf.detectAndCompute(gray, None)
    tmp = []
    for idx in range(16):
        tmp.extend(descs[idx])
    # print(len(tmp))
    return tmp

def load_SURF():
    global CVthreshold
    file = open('./filename_list.txt', 'r')
    all_data = json.load(file)
    cnt = 1
    for cate in all_data:
        print("Processing : {} ({}/{})".format(cate,cnt,len(all_data)))
        cnt += 1
        # if cnt == 5:
            # break
        tmpdict = {}
        ccnt = 1            
        for img in all_data[cate]:
            CVthreshold = 10000
            print("\r└─ Processing : {} ({}/{})".format(img,ccnt,len(all_data[cate])),end='')
            ccnt += 1
            tmp = SURF.SURF(dir_name + cate + '/' + img)
            tmpdict[img] = np.array(tmp,dtype=float).tolist()
            gray_dict[cate] = tmpdict
        print()
    print("Calculate done, saving file......")
    file = open("./SURF_feature.txt", 'w')
    json.dump(gray_dict, file)
    print("ALL Done!!!")