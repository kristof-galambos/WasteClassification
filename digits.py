
import numpy as np
import keras
# from keras.utils import to_categorical
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from neuralnet import NeuralNet, ConvNet

############### getting the smaller dataset ##############################
digits = load_digits()
X = digits.data.reshape(-1, 8, 8)
y = digits.target

X = np.array([X[i] for i in range(len(X)) if y[i] == 0 or y[i] == 1])
y = np.array([y[i] for i in range(len(y)) if y[i] == 0 or y[i] == 1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# y_train = to_categorical(y_train)
print(X_train.shape)
print(y_train.shape)

nn = NeuralNet()
nn.train(X_train, y_train)
y_pred = nn.predict(X_test)
print('The accuracy score is: {:.4f}'.format(nn.get_accuracy_score(y_test)))
print('The confusion matrix is:')
nn.print_confusion_matrix(y_test)


# ############################## getting the larger dataset ######################
# first_class = 8
# second_class = 4
# (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# X_train = np.array([X_train[i] for i in range(len(X_train)) if y_train[i] == first_class or y_train[i] == second_class])
# y_train = np.array([y_train[i] for i in range(len(y_train)) if y_train[i] == first_class or y_train[i] == second_class])
# X_test = np.array([X_test[i] for i in range(len(X_test)) if y_test[i] == first_class or y_test[i] == second_class])
# y_test = np.array([y_test[i] for i in range(len(y_test)) if y_test[i] == first_class or y_test[i] == second_class])
# y_train[y_train == first_class] = 0
# y_train[y_train == second_class] = 1
# y_test[y_test == first_class] = 0
# y_test[y_test == second_class] = 1

# cnn = ConvNet()
# cnn.train(X_train, y_train)
# y_pred = cnn.predict(X_test)
# print('The accuracy score is: {:.4f}'.format(cnn.get_accuracy_score(y_test)))
# print('The confusion matrix is:')
# cnn.print_confusion_matrix(y_test)