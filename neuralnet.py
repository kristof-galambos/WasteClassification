
import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.datasets.samples_generator import make_blobs
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Dropout, Flatten
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.python.keras.layers.normalization import BatchNormalization
# from tensorflow.python.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
# from keras.utils import to_categorical



class NeuralNet():
    
    def __init__(self):
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None
        self.predictions = None
    
    def train(self, train_features, train_labels):
        self.train_features = np.array(train_features)
        self.train_labels = np.array(train_labels)
        # assume train_features has shape n x height x width x 3
        # print(self.train_features.shape)
        n = self.train_features.shape[0]
        height = self.train_features.shape[1]
        width = self.train_features.shape[2]
        try:
            colour_channels = self.train_features.shape[3] # this is probably 3
        except:
            colour_channels = 1
        dimensions = height * width * colour_channels
        self.train_features = np.array([x.flatten() for x in self.train_features]) # need to flatten for dense network
        
        model = Sequential()
        # print(n)
        model.add(Dense(n, input_dim=dimensions, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        model.fit(self.train_features, self.train_labels, epochs=200, verbose=1)
        
        self.model = model
    
    def predict(self, test_features):
        self.test_features = np.array([x.flatten() for x in test_features]) 
        predictions = self.model.predict(self.test_features)
        predictions = predictions.flatten().astype(int)
        self.predictions = predictions
        return predictions
    
    def get_accuracy_score(self, true_labels):
        if self.predictions is None:
            raise Exception("Call the predict function first!")
        acc = accuracy_score(true_labels, self.predictions)
        return acc
    
    def print_confusion_matrix(self, true_labels):
        if self.predictions is None:
            raise Exception("Call the predict function first!")
        conf_mat = confusion_matrix(true_labels, self.predictions)
        print(conf_mat)
        
        
        
class ConvNet(NeuralNet):
    
    def __init__(self):
        super().__init__()
        
    def train(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        # assume train_features has shape n x height x width x 3
        n = self.train_features.shape[0]
        height = self.train_features.shape[1]
        width = self.train_features.shape[2]
        colour_channels = self.train_features.shape[3] # this is probably 3
        dimensions = height * width * colour_channels
        # don't need to flatten for convolutional network
        
        model = Sequential()
        model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(height, width, colour_channels)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        
        model.add(Dense(2, activation='softmax'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        
        self.model = model