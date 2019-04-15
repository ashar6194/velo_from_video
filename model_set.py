from keras.models import Model
import tensorflow as tf
from keras import optimizers
from keras.layers import ELU

from layers import *


def build_model_basic(input_shape):
  inp = Input(shape=input_shape)

  x = conv_act(inp, 24, (3, 3))
  x = conv_act(x, 24, (3, 3), strides=(2, 2), padding='valid')

  x = conv_act(x, 36, (3, 3))
  x = conv_act(x, 36, (3, 3), strides=(2, 2), padding='valid')

  x = conv_act(x, 48, (3, 3))
  x = conv_act(x, 48, (3, 3), strides=(2, 2), padding='valid')

  x = Dropout(rate=0.5)(x)

  x = conv_act(x, 64, (3, 3), strides=(1, 1), padding='valid')
  x = conv_act(x, 64, (3, 3), strides=(1, 1), padding='valid')

  x = Flatten()(x)
  x = Dense(100, kernel_initializer='glorot_uniform', activation='relu')(x)
  x = Dense(50, kernel_initializer='glorot_uniform', activation='relu')(x)
  x = Dense(10, kernel_initializer='glorot_uniform', activation='relu')(x)
  x_out = Dense(1, kernel_initializer='glorot_uniform')(x)

  model = Model(inputs=inp, outputs=x_out, name='Alexnet Model')

  return model


def build_model_elu(input_shape):
  inp = Input(shape=input_shape)

  x = conv(inp, 24, (3, 3))
  x = ELU()(x)
  x = conv(x, 24, (3, 3), strides=(2, 2), padding='valid')
  x = ELU()(x)

  x = conv(x, 36, (3, 3))
  x = ELU()(x)
  x = conv(x, 36, (3, 3), strides=(2, 2), padding='valid')
  x = ELU()(x)

  x = conv(x, 48, (3, 3))
  x = ELU()(x)
  x = conv(x, 48, (3, 3), strides=(2, 2), padding='valid')
  x = ELU()(x)

  x = Dropout(rate=0.5)(x)

  x = conv(x, 64, (3, 3), strides=(1, 1), padding='valid')
  x = ELU()(x)
  x = conv(x, 64, (3, 3), strides=(1, 1), padding='valid')
  x = ELU()(x)

  x = Flatten()(x)
  x = Dense(100, kernel_initializer='glorot_uniform', activation=None)(x)
  x = ELU()(x)
  x = Dense(50, kernel_initializer='glorot_uniform', activation=None)(x)
  x = ELU()(x)
  x = Dense(10, kernel_initializer='glorot_uniform', activation=None)(x)
  x = ELU()(x)
  x_out = Dense(1, kernel_initializer='glorot_uniform')(x)

  model = Model(inputs=inp, outputs=x_out, name='Alexnet Model')

  return model


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


if __name__ == '__main__':
  a = 1