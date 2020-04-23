import tensorflow as tf 
#import tensorflow.keras as keras
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Activation, LSTM, Dense, Input, Flatten, GlobalAveragePooling1D


def Conv1DLSTM(in_shape, num_classes):
    def LFLBBlock(x, filters, pool_param = 4, maxpool = True):
        x = Conv1D(filters, 4, padding='same', kernel_regularizer =tf.keras.regularizers.l2(0.005))(x)
        x = BatchNormalization()(x)
        x = Activation('relu') (x)
        if maxpool:
            x = MaxPooling1D(pool_size = pool_param, strides = pool_param)(x)
        return x
    
    inputs = Input(shape =in_shape)
    x = LFLBBlock(inputs, 32,  2)
    x = LFLBBlock(x, 64)
    x = LFLBBlock(x, 64)
    x = LFLBBlock(x, 128, maxpool = False)
    x = GlobalAveragePooling1D()(x)
    #x = LSTM(32, dropout=0.3, recurrent_dropout=0.3)(x)
    x = Dense(1024, kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = Dense(num_classes, activation = 'softmax')(x)

    model = Model(inputs, x, name = 'conv1DLSTMNet')
    return model
# usage:
#
# x = conv_bn_relu(params) (x)
