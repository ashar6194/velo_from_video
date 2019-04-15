from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
from keras import optimizers
from keras.activations import softmax
from keras.applications.vgg16 import VGG16
import pickle
import keras

from sklearn.metrics import confusion_matrix

from layers import *


def build_ddp_vgg(input_shape, num_classes=54):
  model_vgg16_conv = VGG16(include_top=False)
  inp = Input(shape=input_shape)
  output_vgg16_conv = model_vgg16_conv(inp)
  x = Flatten(name='flatten')(output_vgg16_conv)
  x = Dense(1024, kernel_initializer='glorot_normal', activation='relu')(x)
  x = Dropout(rate=0.2)(x)
  x_out = Dense(256, kernel_initializer='glorot_normal', activation='relu')(x)
  x_out = Dense(num_classes, kernel_initializer='glorot_normal')(x)
  model = Model(inputs=inp, outputs=x_out, name='VGG Model')
  print (model.summary())

  return model


def build_model_basic(input_shape, num_classes=54):
  inp = Input(shape=input_shape)

  x = conv_elu(inp, 24, (3, 3))
  x = conv_elu(x, 24, (3, 3), strides=(2, 2), padding='valid')

  x = conv_elu(x, 36, (3, 3))
  x = conv_elu(x, 36, (3, 3), strides=(2, 2), padding='valid')

  x = conv_elu(x, 48, (3, 3))
  x = conv_elu(x, 48, (3, 3), strides=(2, 2), padding='valid')

  x = Dropout(rate=0.5)(x)

  x = conv_elu(x, 64, (3, 3), strides=(1, 1), padding='valid')
  x = conv_elu(x, 64, (3, 3), strides=(1, 1), padding='valid')

  x = Flatten()(x)
  x = Dense(100, kernel_initializer='glorot_uniform', activation='relu')(x)
  x = Dense(50, kernel_initializer='glorot_uniform', activation='relu')(x)
  x = Dense(10, kernel_initializer='glorot_uniform', activation='relu')(x)
  x_out = Dense(1, kernel_initializer='glorot_uniform')(x)

  model = Model(inputs=inp, outputs=x_out, name='Alexnet Model')

  return model


def build_minivgg_basic(input_shape, num_classes=54):
  inp = Input(shape=input_shape)

  x = conv_act(inp, 64, (3, 3))
  x = MaxPooling2D()(x)
  x = conv_act(x, 128, (3, 3))
  x = MaxPooling2D()(x)
  x = conv_act(x, 256, (3, 3))
  x = conv_act(x, 512, (3, 3))
  x = MaxPooling2D()(x)
  x = conv_act(x, 1024, (3, 3))
  x = Flatten()(x)
  x = Dense(1024, kernel_initializer='glorot_uniform', activation='relu')(x)
  x = Dropout(rate=0.2)(x)
  x = Dense(256, kernel_initializer='glorot_uniform', activation='relu')(x)
  x_out = Dense(num_classes, kernel_initializer='glorot_uniform')(x)
  model = Model(inputs=inp, outputs=x_out, name='Mini VGGnet Model')
  return model


def identity(layer):
  print('Layer output:\n', layer)
  return layer


def direct_huber_loss(y_true, y_pred):
  sigma_sq = 0.8
  alpha = 0.01

  error = y_true - y_pred
  cond = tf.keras.backend.abs(error) < sigma_sq
  squared_loss = 0.5 * tf.keras.backend.square(error) * sigma_sq
  linear_loss = (tf.keras.backend.abs(error) - (0.5 / sigma_sq))
  res_loss = K.mean(tf.where(cond, squared_loss, linear_loss))

  reg_term = K.sum(K.abs(y_pred))
  return ((1 - alpha) * res_loss) + (alpha * reg_term)


def compile_network(model):
  optim_adam = optimizers.Adam(lr=0.0001, decay=0.0005)
  model.compile(loss='mse', optimizer=optim_adam)


if __name__== '__main__':
  print (keras.__version__)
  # custom_L1_loss(1, 2)

  datastats = pickle.load(open('datastats.pkl', 'rb'))
  train_mean = datastats['mean']
  train_std = datastats['std']
  pkl_array = pickle.load(open('pose_centroids.pkl', 'rb'))
  gt_bases = np.reshape(pkl_array, (100, -1))
  new_bases = (gt_bases - train_mean) / train_std
  print (gt_bases.shape, gt_bases[1, ], new_bases[1, ])