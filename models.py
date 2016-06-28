from utils import loss_utils
from settings import IMG_ROWS, IMG_COLS

from keras.models import Sequential, Model,model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, merge, UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.optimizers import Momentum


def load_keras_model(structure, weights):

    model = Sequential()
    model.add(Convolution2D(4, 4, 4, border_mode='same', init='he_normal',
                            input_shape=(1, IMG_ROWS, IMG_COLS)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Convolution2D(8, 4, 4, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('softmax'))

    if structure and weights:

        model = model_from_json(open(structure).read())
        model.load_weights(weights)

    # TODO: try using the dice coef here, to get predicitions that are closer
    # to the actual leaderboard
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

def load_tflearn_model(weights=None):

    model = input_data(shape=[None, IMG_ROWS, IMG_COLS, 1])

    model = conv_2d(model, 32, 6, activation='relu')
    model = max_pool_2d(model, 2, strides=2)

    model = conv_2d(model, 64, 5, activation='relu')
    model = max_pool_2d(model, 2, strides=2)

    model = conv_2d(model, 128, 3, activation='relu')

    model = conv_2d(model, 128, 3, activation='relu')
    model = max_pool_2d(model, 2, strides=2)

    model = fully_connected(model, 4096, activation='relu')
    model = dropout(model, 0.5)
    model = fully_connected(model, 4096, activation='relu')
    model = dropout(model, 0.5)
    model = fully_connected(model, 2, activation='softmax')

    sgd = Momentum(learning_rate=1e-3, lr_decay=1e-6, momentum=0.9)
    model = regression(model, optimizer=sgd,
                       loss='categorical_crossentropy')

    model = tflearn.DNN(model, checkpoint_path='convnet_tf_vgg.tfl.ckpt',
                        max_checkpoints=1, tensorboard_verbose=3)

    if weights:
        model.load(weights)

    return model


def reference_model():
    '''
    This is just a reference model used by that Kaggle dude that gets supposedly +0.57 dice coefficient
    2 minutes per epoch on this model is a normal GPU -- takes likes 1.5 days haha -- use with TensorFlow backend
    '''
    inputs = Input((1, IMG_ROWS, IMG_COLS))
    conv1 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4],
                mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3],
                mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2],
                mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1],
                mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5),
                  loss=loss_utils.dice_coef_loss, metrics=[loss_utils.dice_coef])

    return model
