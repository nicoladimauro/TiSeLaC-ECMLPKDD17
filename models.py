import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Merge, Reshape, RepeatVector
from keras.layers import Activation, Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.layers.merge import Concatenate, Add
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import keras


def build_conv_models(n_models=10, input_shape=(23, 1)):

    input_list = []
    conv_1_list = []
    conv_2_list = []
    output_list = []

    for t in range(n_models):

        inputs = Input(shape=input_shape)
        layers = [inputs]
        input_list.append(inputs)

        conv_1 = Conv1D(filters=8, kernel_size=3, input_shape=input_shape,
                        activation='relu', padding='same')
        x = conv_1(inputs)
        layers.append(x)
        conv_1_list.append(x)

        maxp_1 = MaxPooling1D(pool_size=2,  padding='same')
        x = maxp_1(x)
        layers.append(x)

        conv_2 = Conv1D(filters=4, kernel_size=3,
                        activation='relu', padding='same')
        x = conv_2(x)
        layers.append(x)
        conv_2_list.append(x)

        maxp_2 = MaxPooling1D(pool_size=2,  padding='same')
        x = maxp_2(x)
        layers.append(x)

        f = Flatten()
        x = f(x)
        layers.append(x)
        output_list.append(x)

    conc_conv_1 = Concatenate()
    x = conc_conv_1(conv_1_list)

    all_conv_1 = Conv1D(filters=8, kernel_size=10,
                        dilation_rate=23,
                        activation='relu', padding='same')
    x = all_conv_1(x)

    f = Flatten()
    x = f(x)
    output_list.append(x)

    conc_conv_2 = Concatenate()
    x = conc_conv_2(conv_2_list)

    all_conv_1 = Conv1D(filters=4, kernel_size=10,
                        dilation_rate=12,
                        activation='relu', padding='same')
    x = all_conv_1(x)

    f = Flatten()
    x = f(x)
    output_list.append(x)

    conc_output = Concatenate()
    x = conc_output(output_list)

    output = x

    model = Model(inputs=input_list, outputs=output)

    return model, layers


