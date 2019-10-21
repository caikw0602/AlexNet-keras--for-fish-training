from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, convolutional
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.regularizers import l2

def Alex_Net(IMG_SIZE, class_num=16):

    model = Sequential()

    model.add(Conv2D(96, (11, 11), strides=(4, 4), padding="valid",
                     input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), kernel_regularizer=l2(0.0002)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='default'))
    model.add(Conv2D(256, (5, 5), padding="valid", kernel_regularizer=l2(0.0002)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='default'))
    model.add(Conv2D(384, (3, 3), padding="valid", kernel_regularizer=l2(0.0002)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))

    model.add(convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='default'))
    model.add(Conv2D(384, (3, 3), padding="valid", kernel_regularizer=l2(0.0002)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))

    model.add(convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='default'))
    model.add(Conv2D(256, (3, 3), padding="valid", kernel_regularizer=l2(0.0002)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096, kernel_regularizer=l2(0.0002)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    model.add(Dense(4096, kernel_regularizer=l2(0.0002)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    model.add(Dense(class_num, kernel_regularizer=l2(0.0002)))
    model.add(Activation("softmax"))

    return model
