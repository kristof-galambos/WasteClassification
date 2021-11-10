
import numpy as np
# from keras.utils import to_categorical
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from neuralnet import NeuralNet

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
print(nn.get_accuracy_score(y_test))
nn.print_confusion_matrix(y_test)