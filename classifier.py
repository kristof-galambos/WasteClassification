
import cv2
import os
from neuralnet import NeuralNet

# read in and prepare the data
train = []
basepath = 'DATASET/TRAIN/O'
for i, filepath in enumerate(os.listdir(basepath)):
    # print(basepath + '/' + filepath)
    img = cv2.imread(basepath + '/' + filepath)
    # print(type(img))
    # cv2.imshow('hello', img)
    train.append(img)
    # break
    if i==999:
        break


#legyen egy lista train_features(képek), train_labels(0 ha O,1 ha R), test_features (képek)
#meg kell keverni, hogy ne egymás után jöjjenek az O és R képek

cv2.imshow('image', train[500])

# # use the neural net for training and prediction
# nn = NeuralNet()
# nn.train(train_features, train_labels)
# predicted_labels = nn.predict(test_features)
