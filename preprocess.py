import os
import numpy as np
def get_train_names(path):
    names = []
    file_names = next(os.walk(path))[2]
    for i in file_names:
        if("_val" in i):
            continue
        else:
            fp = open(path + i)
            for line in fp:
                names.append(line.split()[0])
    
    return set(names)

def get_test_names(path):
    names = []
    file_names = next(os.walk(path))[2]
    for i in file_names:
        if("_val" in i):
            fp = open(path + i)
            for line in fp:
                names.append(line.split()[0])
    
    return set(names)

def multiple_lables(x_train, y_train):
    y_train = list(y_train)
    index_x = {}
    for i in range(len(y_train)):
        if (len(y_train[i]) >  1):
            arr = y_train[i][1:]
            y_train[i] = [y_train[i][0]]
            index_x[i] = arr
    imgs = []
    nos = []
        
    for i in index_x:    
        arr = index_x[i]
        img = x_train[i]
        for j in arr:
            nos.append([j])
            imgs.append(img)


    imgs = np.array(imgs)
    nos = np.array(nos)        
            
    x_train = np.append(imgs, x_train ,axis=0)
    y_train = np.append(nos, y_train , axis=0)
    return x_train, y_train