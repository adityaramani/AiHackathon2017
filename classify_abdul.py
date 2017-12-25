import json
import math
import pandas as pd
import preprocess
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from preprocess import reshape_array,reshape_array1
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
import keras
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from sklearn.metrics import confusion_matrix

IMAGE_SIZE = (50, 50)
BASE_PATH = "./VOC2010/"



keys = ['class', 'difficult']

WINDOW_SIZES = [ (100,100), (50,50)]

def pad_image(image):
    w = image.shape[0]
    h = image.shape[1]
    heightSpace = 256 - h
    widthSpace = 256 - w
    leftSpace = heightSpace//2
    rightSpace = heightSpace - leftSpace
    topSpace = widthSpace//2
    bottomSpace = widthSpace - topSpace
    border=cv2.copyMakeBorder(image, top=topSpace , bottom=bottomSpace, left=leftSpace, right=leftSpace, borderType= cv2.BORDER_CONSTANT )
    border  = cv2.resize(image,(256,256))
    return border


def sliding_window(image):
    best_predictions = []
    for size in WINDOW_SIZES:
        predictions = []
        for top in range(0, image.shape[0] - size[0] +1, size[0]):
            for left in range(0, image.shape[1] - size[1] +1, size[1]):
                box = (top , left, top + size[0], left + size[1] )
                cropped_image = image[box[0]:box[2], box[1]: box[3]]
                padded_image = pad_image(cropped_image)
                val = classify(padded_image) , list(box)
                predictions.append(val)
        
        predictions.sort(key = lambda x : x[0])
        best_predictions.append(predictions[0])
    best_predictions.sort(key = lambda x : x[0])
    print(best_predictions[0])
    return best_predictions[0]
            
def classify(im):
    ar = np.array([im])
    prediction = cnn_n.predict(ar)
    prediction = prediction[0]
    prediction = list(prediction)
    max_index= prediction.index(max(prediction))
    max_val = max(prediction)
    class_name = mapping_list[max_index]
    return [max_val , class_name]

def display_image(x):
    cv2.imshow("name",x)
    cv2.waitKey(1)


def LisTOfClass(mapping):
	className = []
	for k in sorted(mapping, key=mapping.get):
		className.append(k)
	return className

def GetImageClass(l):
    className = LisTOfClass(class_mappings)
    l = list(l)
    index = l.index(max(l))
    return className[index]	

def newBoxes(box,width,height):
    newBox=[]
    for i in box:
        newInnerDict ={}
        newInnerDict['y2'] = int(math.ceil((256.0/height)*i['y2']))
        newInnerDict['x2'] = int(math.ceil((256.0/width)*i['x2']))
        newInnerDict['y1'] = int(math.ceil((256.0/height)*i['y1']))
        newInnerDict['x1'] = int(math.ceil((256.0/width)*i['x1']))
        newInnerDict['class'] = i['class']
        newInnerDict['difficult'] = i['difficult']
        newBox.append(newInnerDict)

    return newBox


def cood_main(x):
    i  = x[0]
    return (i['x1'], i['y1'] , i['x2'] , i['y2'] )

def class_main(x): 
    i  = x[0]
    return (i['class'] )


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


def show_image(box, image):
    image= image.copy() 
    box= box[2:]
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle((box[0],box[2]),box[1],box[3]  ,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()  


def base_model(x_train, num_classes ):
    model = Sequential() 
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
    model.add(Dropout(0.2))

    model.add(Conv2D(32,(3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3, 3),padding='same',activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64,(3, 3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    
    model.add(Conv2D(128,(3, 3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(32,activation='softmax',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    opt=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


images = []

with open('image-data.json', encoding='utf-8') as data_file:
    for line in data_file:
        images.append(json.loads(line))


with open('class-count.json', encoding='utf-8') as data_file:
    class_count = json.load(data_file)


with open('class-mappings.json', encoding='utf-8') as data_file:
    class_mappings = json.load(data_file)

train_names = preprocess.get_train_names('./VOC2010/ImageSets/Main/')
test_names = preprocess.get_test_names('./VOC2010/ImageSets/Main/')
df = pd.DataFrame(images)

df['file_name'] = df['filepath'].apply(lambda x : x[-5: -16: -1][::-1])
df['new_boxes'] = df.apply(lambda x: newBoxes(x['bboxes'],x['width'],x['height']),axis=1)
df['cood_main'] = df['new_boxes'].apply(cood_main)
df['class_main'] = df['new_boxes'].apply(class_main)




sample1 = df.sample(frac = 0.3)


sample  = sample1[sample1['class_main'] != 'person']
# print(sample)
sample['array_repr'] = sample['file_name'].apply(image_to_array)
sample['class_membs'] = sample['bboxes'].apply(class_mem)


train, test = train_test_split(sample, test_size=0.2)

x_train = reshape_array1(np.array(train['array_repr']))

y_train = train['class_membs']

x_test = reshape_array1(np.array(test['array_repr']))
y_test = test['class_membs']

x_train ,y_train = preprocess.multiple_lables(x_train, y_train)
x_test ,y_test = preprocess.multiple_lables(x_test,  y_test)

y_train = to_categorical(y_train, len(list(class_mappings.keys())))
y_test = to_categorical(y_test, len(list(class_mappings.keys())))


cnn_n = base_model(x_train,  len(list(class_count.keys())))
cnn_n.fit(x_train, y_train ,epochs=10)

cnn_n.save('CNN_Model_2.h5')

# cnn_n = load_model('CNN_Model_2.h5')
# print(x_test)
# score = cnn_n.evaluate(x_test, y_test)
	
#prediction = cnn_n.predict(x_test)

#prediction= np.array(prediction)
#np.save("prediction.npy",prediction)
#np.save("y_test",y_test)

# print(y_train)

mapping_list = LisTOfClass(class_mappings)
# print(mapping_list)
# print(class_mappings)
#print(test['class_main'])
'''


test['predicted_probs'] = [i for i in range(prediction.shape[0])]
test['predicted_probs'] = test['predicted_probs'].apply(lambda x : prediction[x] )
test['predicted_class'] = test['predicted_probs'].apply(GetImageClass)


mat = confusion_matrix(y_test,prediction , class_mappings)



print(x_train.shape)
print(x_test.shape)
print(mapping_list)
# print(test["class_membs"], test['predicted_boxes'] )
'''
    
test['predicted_boxes']  = test['array_repr'].apply(sliding_window)

boxes = np.array(test['predicted_boxes'])

print("dispaly")

for i in range(len(boxes)):
    show_image(boxes[i] , x_test[i])

    

