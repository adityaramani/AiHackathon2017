import json
import pandas as pd
import preprocess
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

# TRAIN_SIZE = 2000
# TEST_SIZE = 200
BASE_PATH = "./VOC2010/"
IMAGE_SIZE = (128, 128)

keys = ['class', 'difficult']

def new_bboxes(box):
    for i in box:
        for j in i:
            if(j in keys):
                continue
            else :
                i[j] = i[j]

def reshape_array(np_array):
    l = [len(np_array)]
    dimen = np_array[0].shape
    shp = tuple( l + list(dimen))
    zeros = np.zeros(shp)
    for i in range(l[0]):
        zeros[i] = np_array[i]
    
    return zeros

def base_model(x_train, num_classes ):
    model = Sequential() 
    model.add(Conv2D(32, (3, 3), padding='same', input_shape= x_train.shape[1:]))
    model.add(Activation('relu'))

    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9 ,nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

def image_to_array(name):
    img = Image.open(BASE_PATH+"Preprocessed/"+name+'.jpg').convert('RGB')
    img =  np.array(img)
    return img

def one_hot(boxes):    
    index = []
    for i in boxes:
        index.append(i['class'])
    arr = [0 for i in range(len(class_count.keys()))]
    for i in index:
        arr[class_mappings[i]] = 1
    
    return arr
def class_mem(boxes):
    index = []
    for i in boxes:
        index.append(class_mappings[i['class']])
    return index #change

images = []
with open('image-data.json', encoding='utf-8') as data_file:
    for line in data_file:
        images.append(json.loads(line))


with open('class-count.json', encoding='utf-8') as data_file:
    class_count = json.load(data_file)


with open('class-mappings.json', encoding='utf-8') as data_file:
    class_mappings = json.load(data_file)
# print(class_mappings)

train_names = preprocess.get_train_names('./VOC2010/ImageSets/Main/')
test_names = preprocess.get_test_names('./VOC2010/ImageSets/Main/')
df = pd.DataFrame(images)
df['file_name'] = df['filepath'].apply(lambda x : x[-5: -16: -1][::-1])


sample = df.sample(frac = 0.5)

# print(len(train_names))
# print(len(test_names))


# test = sample[sample['file_name'].isin(list(train_names))]
# train = sample[sample['file_name'].isin(list(test_names))]

sample['new_bboxes'] = sample['bboxes'].apply(new_bboxes)

'''
sample['array_repr'] = sample['file_name'].apply(image_to_array)
sample['class_membs'] = sample['bboxes'].apply(class_mem)

# print(sample)
train, test = train_test_split(sample, test_size=0.2)

x_train = reshape_array(np.array(train['array_repr']))
y_train = train['class_membs']

x_test = reshape_array(np.array(test['array_repr']))
y_test = test['class_membs']




x_train ,y_train = preprocess.multiple_lables(x_train, y_train)
x_test ,y_test = preprocess.multiple_lables(x_test,  y_test)


y_train = to_categorical(y_train, len(list(class_mappings.keys())))
y_test = to_categorical(y_test, len(list(class_mappings.keys())))


cnn_n = base_model(x_train,  len(list(class_count.keys())))

print(cnn_n.summary())

cnn_n.fit(x_train, y_train, validation_data=(x_test,y_test))
'''