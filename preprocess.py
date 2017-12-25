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
   
    return x_train, y_train


def reshape_array(np_array):
    l = [len(np_array)]
    dimen = np_array[0].shape
    print(shape)
    shp = tuple( l + list(dimen))
    zeros = np.zeros(shp)
    for i in range(l[0]):
        zeros[i] = np_array[i]
    return zeros


def reshape_array1(np_array):
    l = []
    for i in np_array:
        l.append(i)
    
    return np.array(l)
