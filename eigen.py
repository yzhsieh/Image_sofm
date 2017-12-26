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
###
dir_name = './CorelDB2/'
gray_dict = {}
###
CVthreshold = 10000

### temp
land = 0
straight = 0
###

def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = np.linalg.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs

def test_PCA(data, dims_rescaled_data=2):
    '''
    test by attempting to recover original data array from
    the eigenvectors of its covariance matrix & comparing that
    'recovered' array with the original data
    '''
    _ , _ , eigenvectors = PCA(data, dim_rescaled_data=2)
    data_recovered = np.dot(eigenvectors, m).T
    data_recovered += data_recovered.mean(axis=0)
    assert np.allclose(data, data_recovered)


def plot_pca(data):
    clr1 =  '#2026B2'
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    data_resc, data_orig, _ = PCA(data)
    ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)
    plt.show()

def test():
    test = [[1, 2, 3, 4, 6], [3, 4, 1, 6, 7], [5, 7, 8, 2, 4], [4, 1, 8, 6, 3], [9, 6, 7, 10, 4]]
    test = np.array(test, dtype=float)
    res, evals, evecs = PCA(test, 3)
    print(res)
    print('--------------------')
    print(evals)
    print('--------------------')
    print(evecs)    
    plot_pca(test)

def load_and_turn_gray_new():
    file = open('./filename_list.txt', 'r')
    all_data = json.load(file)
    cnt = 1
    for cate in all_data:
        print("Processing : {} ({}/{})".format(cate,cnt,len(all_data)))
        cnt += 1
        # if cnt == 5:
            # break
        tmpdict = {}            
        for img in all_data[cate]:
            # print("Processing : {}/{}".format(cate,img))
            im = Image.open(dir_name + cate + '/' + img)
            arr = np.array(im)
            ### create dict
            tmparr = np.zeros((arr.shape[:2]),dtype=int).tolist()
            stat = np.zeros((256),dtype=int).tolist()
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    tmp = int(sum(arr[i][j]) / 3)
                    stat[tmp] += 1
            tmpdict[img] = stat
            # print(len(tmparr))
        gray_dict[cate] = tmpdict
        # print(gray_dict)
    print("Calculate done, saving file......")
    file = open("./gray_feature.txt", 'w')
    json.dump(gray_dict, file)
    print("ALL Done!!!")
@autojit


def load_and_turn_RGB():
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
            print("\r Processing : {} ({}/{})".format(img,ccnt,len(all_data[cate])),end='')
            ccnt += 1
            im = Image.open(dir_name + cate + '/' + img)
            arr = np.array(im)
            ### create dict
            ### for 3 color + gray
            stat = np.zeros((256*4),dtype=int).tolist()
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
            tmpdict[img] = stat
        gray_dict[cate] = tmpdict
        print()
        # print(gray_dict)
    print("Calculate done, saving file......")
    file = open("./RGB_gray_feature.txt", 'w')
    json.dump(gray_dict, file)
    print("ALL Done!!!")

def load_brutal():
    global straight, land
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
            print("\r└─ Processing : {} ({}/{})".format(img,ccnt,len(all_data[cate])),end='')
            ccnt += 1
            im = Image.open(dir_name + cate + '/' + img)
            if im.size == (80, 120): 
                land += 1
                im = im.transpose(Image.ROTATE_90)
            elif im.size == (120, 80):
                straight += 1
            else:
                print("WRONG size : ",im.size)
            arr = np.array(im)
            ### create dict
            ### for 3 color + gray
            stat = np.zeros((256*4),dtype=int).tolist()
            tmpR = []
            tmpG = []
            tmpB = []
            tmpGray = []
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    #Gray
                    # tmp = int(sum(arr[i][j]) / 3)
                    # tmpGray.append(tmp)
                    #R
                    tmp = arr[i][j][0]
                    tmpR.append(tmp)
                    #G
                    tmp = arr[i][j][1]
                    tmpG.append(tmp)
                    #B
                    tmp = arr[i][j][2]
                    tmpB.append(tmp)
            tmpAll = []
            tmpAll.extend(tmpR)
            tmpAll.extend(tmpG)
            tmpAll.extend(tmpB)
            # tmpAll.extend(tmpGray)
            tmpdict[img] = np.array(tmpAll,dtype=int).tolist()
            # print(len(tmpAll))
        gray_dict[cate] = tmpdict
        print()
        # print(gray_dict)
    print("land = {}\nstraight = {}".format(land,straight))
    print("Calculate done, saving file......")
    file = open("./all_pixel_RGB_feature.txt", 'w')
    json.dump(gray_dict, file)
    print("ALL Done!!!")

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
            # image = cv2.imread(dir_name + cate + '/' + img)
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # surf = cv2.xfeatures2d.SURF_create(hessianThreshold=CVthreshold, upright=True, extended=False)
            # (kps, descs) = surf.detectAndCompute(gray, None)
            # while len(kps)<16:
                # CVthreshold = CVthreshold - 500
                # print("change thres to : ",CVthreshold)                
                # surf = cv2.xfeatures2d.SURF_create(hessianThreshold=CVthreshold, upright=True, extended=False)
                # (kps, descs) = surf.detectAndCompute(gray, None)
            # tmp = []
            # for idx in range(16):
                # tmp.extend(descs[idx])
            ### create dict
            ### for 3 color + gray
            tmpdict[img] = np.array(tmp,dtype=float).tolist()
            # print(len(tmpAll))
            gray_dict[cate] = tmpdict
        print()
        # print(gray_dict)
    print("Calculate done, saving file......")
    file = open("./SURF_feature.txt", 'w')
    json.dump(gray_dict, file)
    print("ALL Done!!!")


def load_CNN():
    global straight, land
    all_dict = {}
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
            print("\r└─ Processing : {} ({}/{})".format(img,ccnt,len(all_data[cate])),end='')
            ccnt += 1
            im = Image.open(dir_name + cate + '/' + img)
            if im.size == (80, 120): 
                land += 1
                im = im.transpose(Image.ROTATE_90)
                tran = 1
            elif im.size == (120, 80):
                straight += 1
                tran = 0
            else:
                print("WRONG size : ",im.size)
            arr = np.array(im)
            # print(arr.shape)
            ### create dict
            ### for 3 color + gray
            # tmpdict[img] = np.array(arr,dtype=int).tolist()
            tmpdict[img] = arr.tolist()
            # print(len(tmpAll))
        all_dict[cate] = tmpdict
        print()
        # print(gray_dict)
    print("land = {}\nstraight = {}".format(land,straight))
    print("Calculate done, saving file......")
    file = open("./CNN_feature.txt", 'w')
    json.dump(all_dict, file)
    print("ALL Done!!!")


if __name__ == '__main__':
    init_time = time.time()
    load_CNN()
    # load_brutal()
    print("DONE!!")
    print("Time elapsed : {}".format(time.time() - init_time))
