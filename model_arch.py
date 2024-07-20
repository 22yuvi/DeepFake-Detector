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

from keras import backend as K
from tensorflow.core.protobuf import rewriter_config_pb2
from keras import initializers

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU

IMGWIDTH = 256

class Classifier:
    def __init__(self):
        self.model = 0
    
    def predict(self, x):
        if x.size == 0:
            return []
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)

class MesoInception4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)
            
            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)
            
            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding='same', activation='relu')(x3)
            
            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding='same', activation='relu')(x4)

            y = Concatenate(axis = -1)([x1, x2, x3, x4])
            
            return y
        return func
    
    def init_model(self):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        
        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU()(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return KerasModel(inputs = x, outputs = y)

class X_plus_Layer(Layer):
    def __init__(self, **kwargs):
        super(X_plus_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', initializer='zeros', trainable=True)
        super(X_plus_Layer, self).build(input_shape)

    def call(self, inpt_x):
        ''' block-level temporal attention '''
        x, A = inpt_x
        x_diag = x
        for i in range(25-1):
            x_diag = K.concatenate([x_diag, x], axis=2)
        x_diag_channals = x_diag
        x_diag_channals = K.expand_dims(x_diag, axis=3)
        x_diag = K.expand_dims(x_diag, axis=3)
        for i in range(3-1):
            x_diag_channals = K.concatenate([x_diag_channals, x_diag], axis=3)
        
        x_mask = x
        width = 25
        for w in range(width-1):
            x_mask = K.concatenate([x_mask, x], axis=2)
        x_mask_channals = x_mask
        x_mask_channals = K.expand_dims(x_mask, axis=3)
        x_mask = K.expand_dims(x_mask, axis=3)
        for i in range(3-1):
            x_mask_channals = K.concatenate([x_mask_channals, x_mask], axis=3)

        a_part = multiply([x_diag, A])
        a_part = self.alpha * a_part
        b_part = self.beta * x_mask_channals
        ans = Add()([a_part, b_part])
        return ans

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 300, 25, 3)


class AttentionMapLayer(Layer):
    def __init__(self, channels, **kwargs):
        self.channels = channels
        super(AttentionMapLayer, self).__init__(**kwargs)
        roi_map_value = np.zeros((1, 300, 25))
        for i in range(300):
            for j in [3, 8, 12, 14, 17, 19]:
                roi_map_value[0, i, j] = 1
        self.roi_map = K.variable(roi_map_value)

    def call(self, inputs):
        s_o, t_o, ipt = inputs

        height = 300
        width = 25

        ''' adaptive spatial attention '''
        s_o = K.l2_normalize(s_o, axis=1)
        s_map = K.expand_dims(s_o, axis=1)
        s_o = K.expand_dims(s_o, axis=1)
        for h in range(height-1):
            s_map = K.concatenate([s_map, s_o], axis=1)

        ''' frame-level temporal attention '''
        t_o = K.l2_normalize(t_o, axis=1)
        t_map = K.expand_dims(t_o, axis=2)
        t_o = K.expand_dims(t_o, axis=2)
        for w in range(width-1):
            t_map = K.concatenate([t_map, t_o], axis=2)

        ''' Prior Spatial Attention '''
        a_o = multiply([s_map, t_map])
        a_o = Add()([a_o, self.roi_map])

        a_map = K.expand_dims(a_o, axis=3)
        a_o = K.expand_dims(a_o, axis=3)
        for c in range(self.channels-1):
            a_map = K.concatenate([a_map, a_o], axis=3)

        out = multiply([a_map, ipt])
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[2]

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=2, name=bn_name)(x)
    return x

def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def network_with_attention(height, width, channels, classes):

    inpt = Input(shape=(height, width, channels))

    # adaptive spatial attention
    x = Conv2D(64, (15,1), padding='valid', strides=(1,1), activation='relu')(inpt)
    x = BatchNormalization(axis=2)(x)
    x = MaxPooling2D(pool_size=(15,1), strides=(1,1), padding='same')(x)
    x = Flatten()(x)
    spatial_output = Dense(width)(x)

    # frame-level temperal attention
    y = Reshape((height, -1))(inpt)
    y = LSTM(height)(y)
    temperal_output = Dense(height)(y)

    # ST-MAP layer
    z = AttentionMapLayer(channels)([spatial_output, temperal_output, inpt])

    # x_plu: block-level temporal attention
    inpt_x = Input(shape=(height, 1))
    z = X_plus_Layer()([inpt_x, z])

    # ---------------- ResNet 18 -----------------
    res = ZeroPadding2D((3, 3))(z)

    # conv1
    res = Conv2d_BN(res, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    res = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(res)

    # conv2_x
    res = identity_Block(res, nb_filter=64, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=64, kernel_size=(3, 3))

    # conv3_x
    res = identity_Block(res, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=128, kernel_size=(3, 3))

    # conv4_x
    res = identity_Block(res, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=256, kernel_size=(3, 3))

    # conv5_x
    res = identity_Block(res, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=512, kernel_size=(3, 3))

    res = AveragePooling2D(pool_size=(7, 1))(res)
    res = Flatten()(res)
    res = Dense(1, name='resnet_result')(res)
    class_result = Activation('sigmoid')(res)

    model = Model(inputs=[inpt, inpt_x], outputs=class_result)
    return model

def to_be_2d(y):
    ny = np.zeros((len(y), 2))
    for i in range(len(y)):
        if y[i] == 0:
            ny[i,0] = 1
        elif y[i] == 1:
            ny[i,1] = 1
    return ny
