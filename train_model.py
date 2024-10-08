import os
import keras
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.applications.resnet import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from sklearn.decomposition import PCA
from model_arch import *
from random import shuffle

from keras import backend as K
from tensorflow.core.protobuf import rewriter_config_pb2
from keras import initializers

data_path = ""
save_path = ""
if not os.path.exists(save_path):
    os.makedirs(save_path)

best_acc = 0
if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    batch_size = 32
    epochs = 10000

    try:
        data_dir = data_path
        mit_list = np.load(data_dir+"mit.npy")
        mit_list = mit_list/255.0
        Meso_list = np.load(data_dir+"Meso.npy")
        y_list = np.load(data_dir+"y.npy")
        Meso_list = np.reshape(Meso_list, (len(y_list), 300, 1))
    except IOError:
        print("No data")
        sys.exit(0)

    model = network_with_attention(300, 25, 3, 2)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    test_num = len(y_list)/20
    all_data = [i for i in range(len(y_list))]
    shuffle(all_data)
    test = all_data[:test_num]
    train = []
    for i in range(len(y_list)):
        if i in test:
            continue
        train.append(i)

    x_train_mit = mit_list[train]
    x_train_meso = Meso_list[train]
    y_train = y_list[train]

    x_test_mit = mit_list[test]
    x_test_meso = Meso_list[test]
    y_test = y_list[test]

    best_acc = 0
    def saveModel(epoch, logs):
        score, acc = model.evaluate([x_test_mit, x_test_meso], y_test, batch_size=batch_size)
        global best_acc
        if acc > best_acc:
            print("Save model, acc=", acc)
            best_acc = acc
            model.save(save_path + 'model_Meso.h5')
    callbacks = [LambdaCallback(on_epoch_end=saveModel)]

    model.fit([x_train_mit, x_train_meso], y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, callbacks=callbacks) 
