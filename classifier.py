
import cv2
import os
#from neuralnet import NeuralNet
import random

# read in and prepare the data
train_features = []
train_labels = []
test_features = []
test_labels = []
basepath_train_o = 'DATASET/TRAIN/O'
basepath_train_r = 'DATASET/TRAIN/R'
basepath_test_o = 'DATASET/TEST/O'
basepath_test_r = 'DATASET/TEST/R'

dim = (185, 250)


#append 1000 train O pictures
def train_append_1000_o():
    for i, filepath in enumerate(os.listdir(basepath_train_o)):
        img = cv2.imread(basepath_train_o + '/' + filepath)
        img_resized = cv2.resize(img, dim)
        train_features.append(img_resized)
        if i == 999:
            break

#append 1000 R pictures
def train_append_1000_r():
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

def test_append_1000_o():
    for i, filepath in enumerate(os.listdir(basepath_test_o)):
        img = cv2.imread(basepath_test_o + '/' + filepath)
        img_resized = cv2.resize(img, dim)
        test_features.append(img_resized)
        if i == 999:
            break

def test_append_1000_r():
    for i, filepath in enumerate(os.listdir(basepath_test_r)):
        img = cv2.imread(basepath_test_r + '/' + filepath)
        img_resized = cv2.resize(img, dim)
        train_features.append(img_resized)
        if i == 999:
            break

def create_test_labels():
    for i in range(0, 1000):
        test_labels.append(0)
    for i in range(0, 1000):
        test_labels.append(1)

def create_2_random_train_list(train_features, train_labels):
    c = list(zip(train_features, train_labels))
    random.shuffle(c)
    train_features, train_labels = zip(*c)
    return train_features, train_labels

def create_2_random_test_list(test_features, test_labels):
    c = list(zip(test_features, test_labels))
    random.shuffle(c)
    test_features, test_labels = zip(*c)
    return test_features, test_labels


train_append_1000_o()
train_append_1000_r()
test_append_1000_r()
test_append_1000_o()
create_test_labels()
create_train_labels()
train_features_shuffled, train_labels_shuffled = create_2_random_train_list(train_features, train_labels)
test_features_shuffled, test_labels_shuffled = create_2_random_test_list(test_features, test_labels)

# to test if code works
cv2.imshow('anyad', train_features_shuffled[100])
print(train_labels_shuffled[100])

cv2.imshow('anyad', test_features_shuffled[100])
print(test_labels_shuffled[100])

#legyen egy lista train_features(képek), train_labels(0 ha O,1 ha R), test_features (képek)
#meg kell keverni, hogy ne egymás után jöjjenek az O és R képek
#n x width x 3 legyen az összes kép

# # use the neural net for training and prediction
# nn = NeuralNet()
# nn.train(train_features, train_labels)
# predicted_labels = nn.predict(test_features)
