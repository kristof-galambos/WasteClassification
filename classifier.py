
import cv2
import os
#from neuralnet import NeuralNet
import random

# read in and prepare the data
train_features = []
train_labels = []
basepath_train_o = 'DATASET/TRAIN/O'
basepath_train_r = 'DATASET/TRAIN/R'
dim = (185, 250)


#append 1000 O pictures
def append_1000_o():
    for i, filepath in enumerate(os.listdir(basepath_train_o)):
        img = cv2.imread(basepath_train_o + '/' + filepath)
        img_resized = cv2.resize(img, dim)
        train_features.append(img_resized)
        if i == 999:
            break

#append 1000 R pictures
def append_1000_r():
    for i, filepath in enumerate(os.listdir(basepath_train_r)):
        img = cv2.imread(basepath_train_r + '/' + filepath)
        img_resized = cv2.resize(img, dim)
        train_features.append(img_resized)
        if i == 999:
            break

#create train labels with 1000 0-s and 1000 1-s
def create_train_labels():
    for i in range(0, 1000):
        train_labels.append(0)
    for i in range(0, 1000):
        train_labels.append(1)

def create_2_random_list():
    c = list(zip(train_features, train_labels))
    random.shuffle(c)
    train_features, train_labels = zip(*c)



append_1000_o()
append_1000_r()
create_train_labels()
create_2_random_list()
print(train_labels)
print(train_features)

#legyen egy lista train_features(képek), train_labels(0 ha O,1 ha R), test_features (képek)
#meg kell keverni, hogy ne egymás után jöjjenek az O és R képek
#n x width x 3 legyen az összes kép

# # use the neural net for training and prediction
# nn = NeuralNet()
# nn.train(train_features, train_labels)
# predicted_labels = nn.predict(test_features)
