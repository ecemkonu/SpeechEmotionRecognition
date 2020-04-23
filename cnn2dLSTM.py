import tensorflow as tf 
#import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, LSTM, Dense, Flatten, Input, Reshape, SpatialDropout2D


def Conv2DLSTM(in_shape, num_classes):
    def lflbblock(x, filters, pool_kernel, pool_stride):
        x = Conv2D(filters = filters, kernel_size =3, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation('elu') (x)
        x = SpatialDropout2D(0.5)(x)
        x = MaxPooling2D(pool_size = pool_kernel, strides = pool_stride)(x)
        return x
    width, height, _ = in_shape
    width = int(width/128)

    height = int(height / 128)
    inputs = Input(shape =in_shape)
    x = lflbblock(inputs, 64, 2, 2)
    x = lflbblock(x, 64, 4, 4)
    x = lflbblock(x, 128, 4, 4)
    x = lflbblock(x, 128, 4, 4)
    x = Reshape((-1 ,128))(x)
    x = LSTM(256)(x)
    x = Dense(num_classes, activation = 'softmax')(x)

    model = Model(inputs, x, name = 'conv2DLSTMNet')
    return model

