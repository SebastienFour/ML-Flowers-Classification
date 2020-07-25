# Importing modules for CNN
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

#preprocess
from keras.preprocessing.image import ImageDataGenerator

# Data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
from cv2 import cv2              
import numpy as np
from tqdm import tqdm                 
from random import shuffle  
import random as rn
import os

#Checking for GPUs
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

#Formating data functions
def assign_labels(image, flower_type):
    return flower_type

def convertToRGB(img):
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

def prep_train_dataset(flower_type, directory):
    for image in tqdm(os.listdir(directory)):
        label = assign_labels(image, flower_type)
        path = os.path.join(directory, image)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        IMG_ARRAY_DICT.append(np.array(image))
        LABEL_DICT.append(str(label))

#Displaying images function
def show_random_images():
    fig,ax=plt.subplots(3,2)
    fig.set_size_inches(15,12)
    for i in range(3):
        for j in range (2):
            l=rn.randint(0,len(LABEL_DICT))
            ax[i,j].imshow(IMG_ARRAY_DICT[l])
            ax[i,j].set_title(LABEL_DICT[l])

    fig.suptitle('Random BGR flowers', fontsize=16)
    plt.tight_layout()
    plt.show()

#Locating Dataset
FLOWER_DAISY_DIR='D:/Datasets/Flowers/daisy'
FLOWER_SUNFLOWER_DIR='D:/Datasets/Flowers/sunflower'
FLOWER_TULIP_DIR='D:/Datasets/Flowers/tulip'
FLOWER_DANDI_DIR='D:/Datasets/Flowers/dandelion'
FLOWER_ROSE_DIR='D:/Datasets/Flowers/rose'

#Preparing the data
IMG_ARRAY_DICT=[]
LABEL_DICT=[]

IMG_WIDTH = 150
IMG_HEIGHT = 150

DIR_LIST = [FLOWER_DAISY_DIR, FLOWER_SUNFLOWER_DIR, FLOWER_TULIP_DIR, FLOWER_DANDI_DIR, FLOWER_ROSE_DIR]

class_names = os.listdir('D:/Datasets/Flowers')
labels = ['daisy', 'dandelion', 'rose','sunflower','tulip']
print(class_names)

#Assining labels and counting the number of images for each labels
#Daisy
prep_train_dataset('Daisy',FLOWER_DAISY_DIR)
print("Label :", LABEL_DICT[0])
print("Total images of Daisy :", LABEL_DICT.count('Daisy'), "\n")

#Sunflower
prep_train_dataset('Sunflower', FLOWER_SUNFLOWER_DIR)
print("\nLabel :", LABEL_DICT[770])
print("Total images of Sunflowers :", LABEL_DICT.count('Sunflower'), "\n")

#Tuplip
prep_train_dataset('Tuplip', FLOWER_TULIP_DIR)
print("\nLabel :", LABEL_DICT[1504])
print("Total images of Tuplip :", LABEL_DICT.count('Tuplip'), "\n")

#Dandelion
prep_train_dataset('Dandelion', FLOWER_DANDI_DIR)
print("\nLabel :", LABEL_DICT[2488])
print("Total images of Dandelion :", LABEL_DICT.count('Dandelion'), "\n")

#Rose
prep_train_dataset('Rose', FLOWER_ROSE_DIR)
print("\nLabel :", LABEL_DICT[3540])
print("Total images of Rose :", LABEL_DICT.count('Rose'))
print("\nTotal count of images :", len(IMG_ARRAY_DICT), "\n")

#Verifiying the shape of an image array and showing a set of random flowers with labels
print("Image array shape", IMG_ARRAY_DICT[0].shape)
show_random_images()

#Label encoding
lab = LabelEncoder()
Y = lab.fit_transform(LABEL_DICT)
Y = to_categorical(Y, 5)
IMG_ARRAY_DICT = np.array(IMG_ARRAY_DICT)
IMG_ARRAY_DICT = IMG_ARRAY_DICT/255

#Splitting data in training set and validation set
x_train, x_test, y_train, y_test = train_test_split(IMG_ARRAY_DICT, Y, test_size=0.25, random_state=42)

#Setting random seeds
np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

#Building the model
def simple_model():
    
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(5, activation = "softmax"))
    
    #Compiling the model
    opt = tf.keras.optimizers.Adam(1e-4)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['acc'])

    return model

#Models summaries
model_simple = simple_model()
model_simple.summary()

#Checkpoint callback usage
checkpoint_path = ('C:\\Users\\sebas\\Desktop\\Project Telespazio\\Saved_Model\\Checkpoints')
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1, period=5)
