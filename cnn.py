import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = 'train/train'
TEST_DIR = 'test/test'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = f'healthyvsunhealthy-{LR}-2conv-basic'

def label_img(img):
    word_label = img[0]
    if word_label == 'h':
        return [1, 0, 0, 0]
    elif word_label == 'b':
        return [0, 1, 0, 0]
    elif word_label == 'v':
        return [0, 0, 1, 0]
    elif word_label == 'l':
        return [0, 0, 0, 1]


def create_train_data():
    training_data = []
    shapes = set()  # Set to store shapes of all elements
    
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        # Append the shape of the image to the set
        shapes.add(img.shape)
        
        training_data.append([np.array(img), np.array(label)])
    
    shuffle(training_data)
    
    # Check the data types of the elements in training_data
    training_data_types = [type(item[0]) for item in training_data]  # Check the type of the first element in each pair
    unique_training_data_types = np.unique(training_data_types)
    if len(unique_training_data_types) > 1:
        print("Data types are different at indices:")
        for i, item in enumerate(training_data_types):
            print(f"Index {i}: {item}")
    else:
        print("Data types are consistent throughout the array.")

    return training_data


    



def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    np.save('test_data.npy', np.array(testing_data))  # Convert list to numpy array before saving
    return testing_data

train_data = create_train_data()


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 4, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists(f'{MODEL_NAME}.meta'):
    model.load(MODEL_NAME)
    print('Model loaded!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=40, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
