import json
import PIL
from PIL import Image
import numpy as np
import colorsys
###
dir_name = './CorelDB2/'
###
raw_data = dict()
###
def extractRGB():
    for a in all_name:
        for b in all_name[a]:
            print(b)

def demo():
    arrR = arr * 1
    arrG = arr * 1
    arrB = arr * 1
    print(arr.shape)
    for i in range(80):
        for j in range(120):
            arrR[i][j][1] = arrR[i][j][2] = 0
            arrG[i][j][0] = arrG[i][j][2] = 0
            arrB[i][j][0] = arrB[i][j][1] = 0
    oimg = Image.fromarray(arrR,'RGB')
    oimg.save('testR.png')
    oimg = Image.fromarray(arrG,'RGB')
    oimg.save('testG.png')
    oimg = Image.fromarray(arrB,'RGB')
    oimg.save('testB.png')

def cal_RGB_ave(arr):
    axis1 = arr.shape[0]
    axis2 = arr.shape[1]
    arrR = 0
    arrG = 0
    arrB = 0
    for i in range(axis1):
        for j in range(axis2):
            arrR += arr[i][j][0]
            arrG += arr[i][j][1]
            arrB += arr[i][j][2]
    mult = axis1 * axis2
    return [arrR/mult, arrG/mult, arrB/mult]

if __name__ == '__main__':
    nfile = open('filename_list.txt')
    all_name = json.load(nfile)
    # print(all_name)
    print(all_name['bld_castle'][1])
    im = PIL.Image.open(dir_name + 'bld_castle/' + all_name['bld_castle'][0])
    arr = np.array(im)
    print(arr.shape)
    print(cal_RGB_ave(arr))
    for item in all_name:
        for k in all_name[item]:
            print("Processing : ",item + '/' + k)
            im = Image.open(dir_name + item + '/' + k)
            arr = np.array(im)
            raw_data[k] = {'RGB':cal_RGB_ave(arr),"dir":item}
    print(raw_data)

    file = open('features.txt','w')
    json.dump(raw_data, file)
    # demo()
    # extractRGB()